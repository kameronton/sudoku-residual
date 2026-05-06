#!/usr/bin/env python3
"""Mean-ablate specialized heads and measure substructure suppression gaps.

For each configured head H_S, replace the head's projected output at the read
position with its mean projected output over held-out puzzles. Then measure

    mean[ablated_logit(cell, digit) - clean_logit(cell, digit)]

over empty cells in S where digit is already present in the same row/col/box
instance. Positive gaps mean the ablation removed suppression.

Defaults use human-readable 1-indexed labels:
  L4H6 on columns 4-6, control columns 1-3
  L5H8 on rows 7-9,    control rows 1-3
  L6H3 on box 5,       control box 1

Usage:
    uv run python plots/scripts/fig_ablation_head_substructure_suppression.py
    uv run python plots/scripts/fig_ablation_head_substructure_suppression.py --n-puzzles 500
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from tqdm.auto import tqdm

from sudoku.activations import load_checkpoint
from sudoku.data import PAD_TOKEN
from sudoku.probes.session import ProbeSession

DEFAULT_CACHE = "results/3M-backtracking-packing/activations.npz"
CSV_OUT = Path("plots/data/fig_ablation_head_substructure_suppression.csv")
NPZ_OUT = Path("plots/data/fig_ablation_head_substructure_suppression.npz")


@dataclass(frozen=True)
class HeadSpec:
    label: str
    layer: int          # 0-indexed
    head: int           # 0-indexed
    stype: str          # rows, cols, or boxes
    target: tuple[int, ...]
    control: tuple[int, ...]


DEFAULT_SPECS = (
    HeadSpec("L4H6_cols4-6", layer=3, head=5, stype="cols", target=(3, 4, 5), control=(0, 1, 2)),
    HeadSpec("L5H8_rows7-9", layer=4, head=7, stype="rows", target=(6, 7, 8), control=(0, 1, 2)),
    HeadSpec("L6H3_box5", layer=5, head=2, stype="boxes", target=(4,), control=(0,)),
)


def _ln(x, scale, bias, eps=1e-5):
    m = x.mean(-1, keepdims=True)
    v = x.var(-1, keepdims=True)
    return (x - m) / jnp.sqrt(v + eps) * scale + bias


def _pad_batch(sequences: np.ndarray, rows: np.ndarray, max_seq_len: int) -> np.ndarray:
    tokens = np.full((len(rows), max_seq_len), PAD_TOKEN, dtype=np.int32)
    for i, row in enumerate(rows):
        seq = sequences[int(row)]
        length = min(len(seq), max_seq_len)
        tokens[i, :length] = seq[:length]
    return tokens


def _cells_in(stype: str, instance: int) -> list[int]:
    if stype == "rows":
        return [instance * 9 + c for c in range(9)]
    if stype == "cols":
        return [r * 9 + instance for r in range(9)]
    if stype == "boxes":
        r0, c0 = 3 * (instance // 3), 3 * (instance % 3)
        return [(r0 + dr) * 9 + (c0 + dc) for dr in range(3) for dc in range(3)]
    raise ValueError(f"unknown substructure type {stype!r}")


def _grid_at_clues(seq, n_clues: int) -> np.ndarray:
    grid = np.zeros(81, dtype=np.int8)
    for tok in seq[:n_clues]:
        tok = int(tok)
        if 0 <= tok <= 728:
            grid[tok // 9] = (tok % 9) + 1
    return grid


def _pair_tokens(grid: np.ndarray, stype: str, instances: tuple[int, ...]) -> list[int]:
    tokens: list[int] = []
    for inst in instances:
        cells = _cells_in(stype, inst)
        present = sorted({int(grid[cell]) for cell in cells if grid[cell] != 0})
        if not present:
            continue
        for cell in cells:
            if grid[cell] != 0:
                continue
            for digit in present:
                tokens.append(cell * 9 + (digit - 1))
    return tokens


def _make_forward_fns(params, model):
    cfg = model.config
    n_layers = cfg.n_layers
    n_heads = cfg.n_heads
    d_model = cfg.d_model
    head_dim = d_model // n_heads
    dtype = cfg.jax_dtype

    def forward(params, tokens, query_pos, *, ablate_layer: int, ablate_head: int, mean_contrib):
        bsz, seq_len = tokens.shape
        b_idx = jnp.arange(bsz)
        pos_ids = jnp.arange(seq_len)[None, :]
        x = params["token_emb"]["embedding"][tokens]
        if cfg.use_pos_emb:
            x = x + params["pos_emb"]["embedding"][pos_ids]

        gathered_contrib = None
        for li in range(n_layers):
            blk = params[f"block_{li}"]
            attn = blk["CausalSelfAttention_0"]

            h = _ln(x, blk["LayerNorm_0"]["scale"], blk["LayerNorm_0"]["bias"])
            qkv = h @ attn["qkv"]["kernel"] + attn["qkv"]["bias"]
            q = qkv[..., :d_model].reshape(bsz, seq_len, n_heads, head_dim).transpose(0, 2, 1, 3)
            k = qkv[..., d_model:2 * d_model].reshape(bsz, seq_len, n_heads, head_dim).transpose(0, 2, 1, 3)
            v = qkv[..., 2 * d_model:].reshape(bsz, seq_len, n_heads, head_dim).transpose(0, 2, 1, 3)

            scores = (q @ k.transpose(0, 1, 3, 2)) * (head_dim ** -0.5)
            mask = jnp.tril(jnp.ones((seq_len, seq_len), dtype=bool))
            scores = jnp.where(mask[None, None], scores, jnp.finfo(dtype).min)
            z = jax.nn.softmax(scores, axis=-1) @ v

            w_o_h = attn["proj"]["kernel"].reshape(n_heads, head_dim, d_model)
            if li == ablate_layer:
                z_q = z[b_idx, ablate_head, query_pos]
                contrib_q = jnp.einsum("bd,de->be", z_q, w_o_h[ablate_head])
                gathered_contrib = contrib_q

            z_cat = z.transpose(0, 2, 1, 3).reshape(bsz, seq_len, d_model)
            attn_out = z_cat @ attn["proj"]["kernel"] + attn["proj"]["bias"]
            if li == ablate_layer:
                delta = mean_contrib[None, :] - gathered_contrib
                attn_out = attn_out.at[b_idx, query_pos].add(delta.astype(attn_out.dtype))

            x = x + attn_out
            h2 = _ln(x, blk["LayerNorm_1"]["scale"], blk["LayerNorm_1"]["bias"])
            h2 = jax.nn.gelu(h2 @ blk["Dense_0"]["kernel"] + blk["Dense_0"]["bias"])
            x = x + h2 @ blk["Dense_1"]["kernel"] + blk["Dense_1"]["bias"]

        x = _ln(x, params["LayerNorm_0"]["scale"], params["LayerNorm_0"]["bias"])
        logits = x @ params["lm_head"]["kernel"] + params["lm_head"]["bias"]
        return logits[b_idx, query_pos, :729], gathered_contrib

    return jax.jit(forward, static_argnames=("ablate_layer", "ablate_head"))


def _heldout_rows(session: ProbeSession, step: int, n_puzzles: int | None, seed: int) -> np.ndarray:
    idx = session.index.at_step(step).first_per_puzzle()
    _, test_mask = session.split(idx, seed=seed)
    rows = idx.puzzle_idx[test_mask]
    seq_pos = idx.seq_pos[test_mask]

    keep = []
    for row, pos in zip(rows, seq_pos):
        row = int(row)
        pos = int(pos)
        seq = session.sequences[row]
        if pos < len(seq) and int(seq[pos]) != PAD_TOKEN:
            keep.append(row)
    rows = np.array(keep, dtype=np.int32)

    if n_puzzles is not None and len(rows) > n_puzzles:
        rng = np.random.default_rng(seed)
        rows = np.sort(rng.choice(rows, size=n_puzzles, replace=False))
    return rows


def _query_positions(session: ProbeSession, rows: np.ndarray, step: int) -> np.ndarray:
    pos = np.array([int(session.n_clues[int(row)]) + step for row in rows], dtype=np.int32)
    return pos


def _mean_contrib(forward_fn, params, tokens_all, qpos_all, spec: HeadSpec, batch_size: int) -> np.ndarray:
    sums = None
    count = 0
    dummy_mean = jnp.zeros(params["lm_head"]["kernel"].shape[0], dtype=jnp.float32)

    for start in tqdm(range(0, len(tokens_all), batch_size), desc=f"{spec.label} mean"):
        end = min(start + batch_size, len(tokens_all))
        _, contrib = forward_fn(
            params,
            jnp.asarray(tokens_all[start:end]),
            jnp.asarray(qpos_all[start:end]),
            ablate_layer=spec.layer,
            ablate_head=spec.head,
            mean_contrib=dummy_mean,
        )
        contrib_np = np.asarray(contrib, dtype=np.float64)
        sums = contrib_np.sum(axis=0) if sums is None else sums + contrib_np.sum(axis=0)
        count += contrib_np.shape[0]
    return (sums / max(count, 1)).astype(np.float32)


def _score_spec(
    forward_fn,
    clean_logits_fn,
    params,
    session: ProbeSession,
    rows: np.ndarray,
    tokens_all: np.ndarray,
    qpos_all: np.ndarray,
    spec: HeadSpec,
    mean_contrib: np.ndarray,
    batch_size: int,
) -> list[dict]:
    sums = {"target": 0.0, "control": 0.0}
    sums_sq = {"target": 0.0, "control": 0.0}
    counts = {"target": 0, "control": 0}
    puzzle_counts = {"target": 0, "control": 0}

    for start in tqdm(range(0, len(rows), batch_size), desc=f"{spec.label} ablate"):
        end = min(start + batch_size, len(rows))
        ablated_logits, _ = forward_fn(
            params,
            jnp.asarray(tokens_all[start:end]),
            jnp.asarray(qpos_all[start:end]),
            ablate_layer=spec.layer,
            ablate_head=spec.head,
            mean_contrib=jnp.asarray(mean_contrib),
        )
        ablated = np.asarray(ablated_logits)

        clean_logits = np.asarray(
            clean_logits_fn(params, jnp.asarray(tokens_all[start:end]), jnp.asarray(qpos_all[start:end]))
        )

        for local_i, row in enumerate(rows[start:end]):
            grid = _grid_at_clues(session.sequences[int(row)], int(session.n_clues[int(row)]))
            token_groups = {
                "target": _pair_tokens(grid, spec.stype, spec.target),
                "control": _pair_tokens(grid, spec.stype, spec.control),
            }
            diff = ablated[local_i] - clean_logits[local_i]
            for group, toks in token_groups.items():
                if not toks:
                    continue
                vals = diff[np.asarray(toks, dtype=np.int32)]
                sums[group] += float(vals.sum())
                sums_sq[group] += float(np.square(vals).sum())
                counts[group] += int(vals.size)
                puzzle_counts[group] += 1

    rows_out = []
    for group in ("target", "control"):
        n = counts[group]
        mean = sums[group] / n if n else np.nan
        var = (sums_sq[group] / n - mean * mean) if n else np.nan
        stderr = np.sqrt(max(var, 0.0) / n) if n else np.nan
        rows_out.append({
            "head": spec.label,
            "layer": spec.layer + 1,
            "head_index": spec.head + 1,
            "substructure_type": spec.stype,
            "group": group,
            "instances": " ".join(str(i + 1) for i in (spec.target if group == "target" else spec.control)),
            "mean_gap": mean,
            "stderr_gap": stderr,
            "n_pairs": n,
            "n_puzzles_with_pairs": puzzle_counts[group],
        })
    return rows_out


def _make_clean_logits_fn(params, model):
    @jax.jit
    def clean_logits(params, tokens, query_pos):
        logits = model.apply({"params": params}, tokens)
        b_idx = jnp.arange(tokens.shape[0])
        return logits[b_idx, query_pos, :729]

    return clean_logits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", default=DEFAULT_CACHE)
    parser.add_argument("--step", type=int, default=0, help="read position offset from SEP")
    parser.add_argument("--n-puzzles", type=int, default=None, help="optional held-out puzzle subsample")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--csv-out", type=Path, default=CSV_OUT)
    parser.add_argument("--npz-out", type=Path, default=NPZ_OUT)
    args = parser.parse_args()

    session = ProbeSession(args.cache, max_step=args.step, act_type="post_mlp")
    params, model = load_checkpoint(args.cache.replace("activations.npz", "checkpoint"))
    if not session.has_sep:
        raise ValueError("This script expects a SEP/END_CLUES token so step 0 is well-defined.")

    rows = _heldout_rows(session, args.step, args.n_puzzles, args.seed)
    if len(rows) == 0:
        raise ValueError("No held-out rows found for this step.")
    qpos = _query_positions(session, rows, args.step)
    tokens = _pad_batch(session.sequences, rows, model.config.max_seq_len)

    valid = np.array(
        [pos < model.config.max_seq_len and tokens[i, pos] != PAD_TOKEN for i, pos in enumerate(qpos)],
        dtype=bool,
    )
    rows, qpos, tokens = rows[valid], qpos[valid], tokens[valid]

    clean_logits_fn = _make_clean_logits_fn(params, model)
    forward_fn = _make_forward_fns(params, model)

    all_rows = []
    mean_vectors = {}
    for spec in DEFAULT_SPECS:
        if spec.layer >= model.config.n_layers or spec.head >= model.config.n_heads:
            print(f"Skipping {spec.label}: model has {model.config.n_layers} layers, {model.config.n_heads} heads")
            continue
        mean_contrib = _mean_contrib(forward_fn, params, tokens, qpos, spec, args.batch_size)
        mean_vectors[spec.label] = mean_contrib
        all_rows.extend(
            _score_spec(
                forward_fn,
                clean_logits_fn,
                params,
                session,
                rows,
                tokens,
                qpos,
                spec,
                mean_contrib,
                args.batch_size,
            )
        )

    args.csv_out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.csv_out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"Saved {args.csv_out}")

    args.npz_out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.npz_out,
        rows=rows,
        qpos=qpos,
        **{f"mean_contrib_{k}": v for k, v in mean_vectors.items()},
    )
    print(f"Saved {args.npz_out}")

    for row in all_rows:
        print(
            f"{row['head']} {row['group']:>7}: gap={row['mean_gap']:.4f}"
            f" +/- {row['stderr_gap']:.4f}  pairs={row['n_pairs']}"
        )


if __name__ == "__main__":
    main()
