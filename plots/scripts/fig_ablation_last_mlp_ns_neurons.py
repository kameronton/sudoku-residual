#!/usr/bin/env python3
"""Mean-ablate final-MLP naked-single neurons for the model's next NS guess.

Protocol:
  1. Load the cell -> neuron map from fig_neuron_candidate_sensitivity.csv.
  2. Use one puzzle split to estimate each mapped neuron's mean GELU activation
     in the final MLP.
  3. Use a disjoint split to find states with at least two naked singles where
     the model's next generated token is one of those naked singles.
  4. Re-run inference while replacing the final-MLP neuron(s) mapped to the
     predicted cell with their mean activations.
  5. Report the mean drop in the predicted NS logit/probability, and the mean
     drop in the logits of the other naked singles in the same states.

Usage:
    uv run python plots/scripts/fig_ablation_last_mlp_ns_neurons.py
    uv run python plots/scripts/fig_ablation_last_mlp_ns_neurons.py --mean-puzzles 1000 --eval-puzzles 1000
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from sudoku.activations import load_checkpoint
from sudoku.data import PAD_TOKEN, decode_fill
from sudoku.data_bt import PAD_TOKEN_BT, POP_TOKEN, PUSH_TOKEN, SUCCESS_TOKEN
from sudoku.probes.session import ActivationIndex, ProbeSession

DEFAULT_CACHE = "results/3M-backtracking-packing/activations.npz"
DEFAULT_NEURONS = "plots/data/fig_neuron_candidate_sensitivity.csv"
CSV_OUT = Path("plots/data/fig_ablation_last_mlp_ns_neurons.csv")
NPZ_OUT = Path("plots/data/fig_ablation_last_mlp_ns_neurons.npz")

BATCH = 64
_PAD = {PAD_TOKEN, PAD_TOKEN_BT}
_CONTROL = {PUSH_TOKEN, POP_TOKEN, SUCCESS_TOKEN}


def _seq_len(seq) -> int:
    for i, tok in enumerate(seq):
        if int(tok) in _PAD:
            return i
    return len(seq)


def _candidates(board: dict[int, int]) -> list[int]:
    row = [0] * 9
    col = [0] * 9
    box = [0] * 9
    for cell, digit in board.items():
        r, c = divmod(cell, 9)
        bit = 1 << (digit - 1)
        row[r] |= bit
        col[c] |= bit
        box[(r // 3) * 3 + c // 3] |= bit

    full = (1 << 9) - 1
    out = [0] * 81
    for cell in range(81):
        if cell in board:
            continue
        r, c = divmod(cell, 9)
        out[cell] = full & ~row[r] & ~col[c] & ~box[(r // 3) * 3 + c // 3]
    return out


def _load_cell_neurons(path: str, min_gap: float | None) -> dict[int, list[int]]:
    df = pd.read_csv(path)
    if min_gap is not None and "gap" in df.columns:
        df = df[df["gap"] >= min_gap]
    cell_to_neurons: dict[int, list[int]] = defaultdict(list)
    for _, row in df.iterrows():
        cell_to_neurons[int(row["cell"])].append(int(row["neuron"]))
    return {cell: sorted(set(neurons)) for cell, neurons in cell_to_neurons.items()}


def _valid_position_states(session: ProbeSession, puzzle_ids: np.ndarray) -> ActivationIndex:
    puzzle_idx = []
    seq_pos = []
    for pi in tqdm(puzzle_ids, desc="Collecting mean states"):
        pi = int(pi)
        seq = session.sequences[pi]
        slen = _seq_len(seq)
        nc = int(session.n_clues[pi])
        anchor = nc if session.has_sep else nc - 1
        depth = 0
        for sp in range(anchor, slen):
            tok = int(seq[sp])
            if tok == PUSH_TOKEN:
                depth += 1
            elif tok == POP_TOKEN:
                depth = max(depth - 1, 0)
            if tok in _CONTROL or depth > 0:
                continue
            puzzle_idx.append(pi)
            seq_pos.append(sp)

    n = len(puzzle_idx)
    return ActivationIndex(
        puzzle_idx=np.asarray(puzzle_idx, dtype=np.int32),
        seq_pos=np.asarray(seq_pos, dtype=np.int32),
        step=np.zeros(n, dtype=np.int32),
        n_filled=np.zeros(n, dtype=np.int16),
        tokens=np.zeros(n, dtype=np.int16),
    )


def _ln_np(x: np.ndarray, scale: np.ndarray, bias: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    return (x - x.mean(-1, keepdims=True)) / np.sqrt(x.var(-1, keepdims=True) + eps) * scale + bias


def _gelu_np(x: np.ndarray) -> np.ndarray:
    return np.asarray(jax.nn.gelu(jnp.asarray(x)), dtype=np.float32)


def _collect_neuron_means(
    session_attn: ProbeSession,
    params,
    layer: int,
    idx: ActivationIndex,
    neurons: np.ndarray,
    batch_size: int,
) -> np.ndarray:
    blk = params[f"block_{layer}"]
    w_in = np.asarray(blk["Dense_0"]["kernel"], dtype=np.float32)
    b_in = np.asarray(blk["Dense_0"]["bias"], dtype=np.float32)
    ln_scale = np.asarray(blk["LayerNorm_1"]["scale"], dtype=np.float32)
    ln_bias = np.asarray(blk["LayerNorm_1"]["bias"], dtype=np.float32)

    sums = np.zeros(len(neurons), dtype=np.float64)
    count = 0
    for lo in tqdm(range(0, len(idx), batch_size), desc="Mean activations"):
        hi = min(lo + batch_size, len(idx))
        x = session_attn.acts(idx[lo:hi], layer=layer).astype(np.float32)
        h = _gelu_np(_ln_np(x, ln_scale, ln_bias) @ w_in[:, neurons] + b_in[neurons])
        sums += h.sum(axis=0)
        count += h.shape[0]
    return (sums / max(count, 1)).astype(np.float32)


def _scan_eval_states(
    session: ProbeSession,
    puzzle_ids: np.ndarray,
    cell_to_neurons: dict[int, list[int]],
    max_states: int,
) -> list[dict]:
    states = []
    for pi in tqdm(puzzle_ids, desc="Scanning eval puzzles"):
        pi = int(pi)
        seq = session.sequences[pi]
        slen = _seq_len(seq)
        nc = int(session.n_clues[pi])
        anchor = nc if session.has_sep else nc - 1

        board: dict[int, int] = {}
        for tok in seq[:anchor]:
            tok = int(tok)
            if 0 <= tok <= 728:
                r, c, digit = decode_fill(tok)
                board[r * 9 + c] = digit

        stack: list[dict[int, int]] = []
        depth = 0
        for sp in range(anchor, slen - 1):
            tok = int(seq[sp])
            if 0 <= tok <= 728:
                r, c, digit = decode_fill(tok)
                board[r * 9 + c] = digit
            elif tok == PUSH_TOKEN:
                stack.append(dict(board))
                depth += 1
            elif tok == POP_TOKEN and stack:
                board = stack.pop()
                depth = max(depth - 1, 0)

            next_tok = int(seq[sp + 1])
            if tok in _CONTROL or depth > 0 or next_tok in _CONTROL:
                continue
            if not (0 <= next_tok <= 728):
                continue

            cands = _candidates(board)
            ns = [(cell, mask.bit_length()) for cell, mask in enumerate(cands) if mask and not (mask & (mask - 1))]
            if len(ns) < 2:
                continue

            ns_tokens = [cell * 9 + (digit - 1) for cell, digit in ns]
            if next_tok not in ns_tokens:
                continue
            pred_cell = next_tok // 9
            if pred_cell not in cell_to_neurons:
                continue
            other_tokens = [tok_ns for tok_ns in ns_tokens if tok_ns != next_tok]
            if not other_tokens:
                continue

            states.append({
                "pi": pi,
                "sp": sp,
                "pred_token": next_tok,
                "pred_cell": pred_cell,
                "other_tokens": other_tokens,
                "n_ns": len(ns_tokens),
            })
            break

        if len(states) >= max_states:
            break
    return states


def _pad_tokens(session: ProbeSession, states: list[dict], max_seq_len: int) -> tuple[np.ndarray, np.ndarray]:
    tokens = np.full((len(states), max_seq_len), PAD_TOKEN, dtype=np.int32)
    qpos = np.zeros(len(states), dtype=np.int32)
    for i, state in enumerate(states):
        seq = session.sequences[int(state["pi"])]
        length = min(len(seq), max_seq_len)
        tokens[i, :length] = seq[:length]
        qpos[i] = int(state["sp"])
    return tokens, qpos


def _ln(x, scale, bias, eps=1e-5):
    m = x.mean(-1, keepdims=True)
    v = x.var(-1, keepdims=True)
    return (x - m) / jnp.sqrt(v + eps) * scale + bias


def _make_logits_fn(params, model):
    cfg = model.config
    n_layers = cfg.n_layers
    n_heads = cfg.n_heads
    d_model = cfg.d_model
    head_dim = d_model // n_heads
    dtype = cfg.jax_dtype

    @jax.jit
    def logits_with_mlp_patch(params, tokens, query_pos, neuron_mask, neuron_values):
        bsz, seq_len = tokens.shape
        b_idx = jnp.arange(bsz)
        pos_ids = jnp.arange(seq_len)[None, :]
        x = params["token_emb"]["embedding"][tokens]
        if cfg.use_pos_emb:
            x = x + params["pos_emb"]["embedding"][pos_ids]

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

            z_cat = z.transpose(0, 2, 1, 3).reshape(bsz, seq_len, d_model)
            x = x + z_cat @ attn["proj"]["kernel"] + attn["proj"]["bias"]

            h2 = _ln(x, blk["LayerNorm_1"]["scale"], blk["LayerNorm_1"]["bias"])
            h2 = jax.nn.gelu(h2 @ blk["Dense_0"]["kernel"] + blk["Dense_0"]["bias"])
            if li == n_layers - 1:
                h2_q = h2[b_idx, query_pos]
                h2_q = jnp.where(neuron_mask, neuron_values, h2_q)
                h2 = h2.at[b_idx, query_pos].set(h2_q)
            x = x + h2 @ blk["Dense_1"]["kernel"] + blk["Dense_1"]["bias"]

        x = _ln(x, params["LayerNorm_0"]["scale"], params["LayerNorm_0"]["bias"])
        logits = x @ params["lm_head"]["kernel"] + params["lm_head"]["bias"]
        return logits[b_idx, query_pos, :729]

    return logits_with_mlp_patch


def _softmax_prob(logits: np.ndarray, token: int) -> float:
    m = float(np.max(logits))
    exps = np.exp(logits - m)
    return float(exps[token] / exps.sum())


def _stderr(vals: np.ndarray) -> float:
    if len(vals) < 2:
        return float("nan")
    return float(vals.std(ddof=1) / np.sqrt(len(vals)))


def _run_ablation(
    states: list[dict],
    tokens: np.ndarray,
    qpos: np.ndarray,
    params,
    model,
    cell_to_neurons: dict[int, list[int]],
    mean_by_neuron: dict[int, float],
    batch_size: int,
) -> tuple[list[dict], dict]:
    d_ff = model.config.d_ff
    logits_fn = _make_logits_fn(params, model)

    rows = []
    pred_logit_drops = []
    pred_prob_drops = []
    other_logit_drops = []

    for lo in tqdm(range(0, len(states), batch_size), desc="Ablating eval states"):
        hi = min(lo + batch_size, len(states))
        batch_states = states[lo:hi]
        mask = np.zeros((hi - lo, d_ff), dtype=bool)
        values = np.zeros((hi - lo, d_ff), dtype=np.float32)
        for i, state in enumerate(batch_states):
            for neuron in cell_to_neurons[int(state["pred_cell"])]:
                mask[i, neuron] = True
                values[i, neuron] = mean_by_neuron[neuron]

        clean = np.asarray(
            logits_fn(
                params,
                jnp.asarray(tokens[lo:hi]),
                jnp.asarray(qpos[lo:hi]),
                jnp.zeros_like(jnp.asarray(mask)),
                jnp.asarray(values),
            )
        )
        patched = np.asarray(
            logits_fn(
                params,
                jnp.asarray(tokens[lo:hi]),
                jnp.asarray(qpos[lo:hi]),
                jnp.asarray(mask),
                jnp.asarray(values),
            )
        )

        for i, state in enumerate(batch_states):
            pred = int(state["pred_token"])
            pred_logit_drop = float(clean[i, pred] - patched[i, pred])
            pred_prob_drop = _softmax_prob(clean[i], pred) - _softmax_prob(patched[i], pred)
            other = np.asarray(state["other_tokens"], dtype=np.int32)
            other_drops = clean[i, other] - patched[i, other]
            other_drop = float(other_drops.mean())

            pred_logit_drops.append(pred_logit_drop)
            pred_prob_drops.append(pred_prob_drop)
            other_logit_drops.extend(float(x) for x in other_drops)
            rows.append({
                "pi": int(state["pi"]),
                "sp": int(state["sp"]),
                "pred_token": pred,
                "pred_cell": int(state["pred_cell"]),
                "n_ns": int(state["n_ns"]),
                "n_other_ns": int(len(other)),
                "n_suppressed_neurons": len(cell_to_neurons[int(state["pred_cell"])]),
                "pred_logit_drop": pred_logit_drop,
                "pred_prob_drop": pred_prob_drop,
                "other_ns_logit_drop": other_drop,
            })

    pred_logit = np.asarray(pred_logit_drops, dtype=np.float32)
    pred_prob = np.asarray(pred_prob_drops, dtype=np.float32)
    other_logit = np.asarray(other_logit_drops, dtype=np.float32)
    summary = {
        "n_eval": len(states),
        "mean_pred_logit_drop": float(pred_logit.mean()),
        "stderr_pred_logit_drop": _stderr(pred_logit),
        "mean_pred_prob_drop": float(pred_prob.mean()),
        "stderr_pred_prob_drop": _stderr(pred_prob),
        "mean_other_ns_logit_drop": float(other_logit.mean()),
        "stderr_other_ns_logit_drop": _stderr(other_logit),
        "specificity_ratio": float(pred_logit.mean() / max(abs(other_logit.mean()), 1e-12)),
    }
    return rows, summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", default=DEFAULT_CACHE)
    parser.add_argument("--neurons", default=DEFAULT_NEURONS)
    parser.add_argument("--mean-puzzles", type=int, default=1000)
    parser.add_argument("--eval-puzzles", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=BATCH)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-gap", type=float, default=None,
                        help="optional minimum candidate-sensitivity gap for neuron map")
    parser.add_argument("--csv-out", type=Path, default=CSV_OUT)
    parser.add_argument("--npz-out", type=Path, default=NPZ_OUT)
    args = parser.parse_args()

    cell_to_neurons = _load_cell_neurons(args.neurons, args.min_gap)
    all_neurons = np.asarray(sorted({n for neurons in cell_to_neurons.values() for n in neurons}), dtype=np.int32)
    print(f"Loaded {len(all_neurons)} neurons covering {len(cell_to_neurons)} cells")

    session_attn = ProbeSession(args.cache, act_type="post_attn")
    session = ProbeSession(args.cache, act_type="post_mlp")
    params, model = load_checkpoint(args.cache.replace("activations.npz", "checkpoint"))
    last = model.config.n_layers - 1

    rng = np.random.default_rng(args.seed)
    puzzle_ids = np.arange(session.n_puzzles, dtype=np.int32)
    rng.shuffle(puzzle_ids)
    mean_puzzles = puzzle_ids[:args.mean_puzzles]
    eval_pool = puzzle_ids[args.mean_puzzles:]

    mean_idx = _valid_position_states(session, mean_puzzles)
    mean_vals = _collect_neuron_means(session_attn, params, last, mean_idx, all_neurons, args.batch_size)
    mean_by_neuron = {int(neuron): float(val) for neuron, val in zip(all_neurons, mean_vals)}

    states = _scan_eval_states(session, eval_pool, cell_to_neurons, args.eval_puzzles)
    if not states:
        raise ValueError("No evaluation states found.")
    tokens, qpos = _pad_tokens(session, states, model.config.max_seq_len)
    rows, summary = _run_ablation(
        states,
        tokens,
        qpos,
        params,
        model,
        cell_to_neurons,
        mean_by_neuron,
        args.batch_size,
    )

    args.csv_out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.csv_out, "w", newline="") as f:
        fields = list(rows[0].keys())
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved per-state data to {args.csv_out}")

    args.npz_out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.npz_out,
        neurons=all_neurons,
        mean_values=mean_vals,
        mean_puzzles=mean_puzzles,
        eval_puzzles=np.asarray([s["pi"] for s in states], dtype=np.int32),
        **{k: np.asarray(v) for k, v in summary.items()},
    )
    print(f"Saved summary arrays to {args.npz_out}")

    print("\nSummary")
    print(f"  eval states:               {summary['n_eval']}")
    print(
        f"  prediction logit drop:     {summary['mean_pred_logit_drop']:.4f}"
        f" +/- {summary['stderr_pred_logit_drop']:.4f}"
    )
    print(
        f"  prediction prob drop:      {summary['mean_pred_prob_drop']:.4f}"
        f" +/- {summary['stderr_pred_prob_drop']:.4f}"
    )
    print(
        f"  other-NS logit drop:       {summary['mean_other_ns_logit_drop']:.4f}"
        f" +/- {summary['stderr_other_ns_logit_drop']:.4f}"
    )
    print(f"  specificity ratio:         {summary['specificity_ratio']:.2f}")


if __name__ == "__main__":
    main()
