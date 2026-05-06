#!/usr/bin/env python3
"""
Structure-present coordinate patch for legal initial placements.

For initial held-out states B1 where the first generated token is a legal
placement (cell c, digit d), compare against B2, the immediately following board
state after that placement. For each row/col/box containing c, train
structure-present probes and patch B1 at one layer by copying the normalized
probe coordinate from B2:

    x'_B1 = x_B1 + sum_S ((w_hat[S,d] @ x_B2) - (w_hat[S,d] @ x_B1)) w_hat[S,d]

Then measure the clean B1 top-1 logit drop, patched top-1 validity, and patched
top-1 change rate across layers.

Outputs:
  plots/data/fig_ablation_structure_present_patch_dataset.pkl  - cached held-out pairs
  plots/data/fig_ablation_structure_present_patch_probes.pkl   - cached probe directions
  plots/data/fig_ablation_structure_present_patch.npz          - ablation results
  plots/data/fig_ablation_structure_present_patch.csv          - plotted per-layer data
  plots/figures/fig_ablation_structure_present_patch.pdf          - target logit drop by layer
"""

import argparse
import csv
import pickle
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from tqdm.auto import tqdm

from sudoku.activations import load_checkpoint
from sudoku.data import PAD_TOKEN, decode_fill
from sudoku.data_bt import PAD_TOKEN_BT, POP_TOKEN, PUSH_TOKEN, SUCCESS_TOKEN
from sudoku.probes.modes import STRUCTURE
from sudoku.probes.session import ActivationIndex, ProbeSession

sns.set_theme(style="ticks", context="paper")
plt.rcParams.update({
    "font.size": 10, "axes.titlesize": 10, "axes.labelsize": 10,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 9,
    "font.family": "serif", "pdf.fonttype": 42,
})

DEFAULT_CACHE = "results/3M-backtracking-packing/activations.npz"
DATASETS_CACHE = Path("plots/data/fig_ablation_structure_present_patch_dataset.pkl")
PROBES_CACHE = Path("plots/data/fig_ablation_structure_present_patch_probes.pkl")
RESULTS_OUT = Path("plots/data/fig_ablation_structure_present_patch.npz")
DATA_OUT = Path("plots/data/fig_ablation_structure_present_patch.csv")
FIGURE_OUT = Path("plots/figures/fig_ablation_structure_present_patch.pdf")

PROBE_STEP = 0
MAX_SAMPLES = 300
SEED = 42
EPS = 1e-8
_PAD = {PAD_TOKEN, PAD_TOKEN_BT}
_CONTROL = {PUSH_TOKEN, POP_TOKEN, SUCCESS_TOKEN}

SUBTYPES = (
    [("row", i) for i in range(9)]
    + [("col", i) for i in range(9)]
    + [("box", i) for i in range(9)]
)

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


def _valid_tokens(cands: list[int]) -> set[int]:
    tokens = set()
    for cell, mask in enumerate(cands):
        for d0 in range(9):
            if mask & (1 << d0):
                tokens.add(cell * 9 + d0)
    return tokens


def _substructures_for_cell(cell: int) -> tuple[tuple[str, int], tuple[str, int], tuple[str, int]]:
    r, c = divmod(cell, 9)
    return (
        ("row", r),
        ("col", c),
        ("box", (r // 3) * 3 + c // 3),
    )


def _make_index(samples: list[dict], key: str) -> ActivationIndex:
    n = len(samples)
    return ActivationIndex(
        puzzle_idx=np.array([s["pi"] for s in samples], dtype=np.int32),
        seq_pos=np.array([s[key] for s in samples], dtype=np.int32),
        step=np.zeros(n, dtype=np.int32),
        n_filled=np.zeros(n, dtype=np.int16),
        tokens=np.zeros(n, dtype=np.int16),
    )


def _replay_board(seq, pos: int) -> dict[int, int]:
    board: dict[int, int] = {}
    stack: list[dict[int, int]] = []
    for tok in seq[:pos + 1]:
        tok = int(tok)
        if 0 <= tok <= 728:
            r, c, digit = decode_fill(tok)
            board[r * 9 + c] = digit
        elif tok == PUSH_TOKEN:
            stack.append(dict(board))
        elif tok == POP_TOKEN and stack:
            board = stack.pop()
    return board


def build_dataset(
    session: ProbeSession,
    rng,
    max_n: int,
    allowed_pis: set[int],
) -> list[dict]:
    data: list[dict] = []

    for pi in tqdm(sorted(allowed_pis), desc="Scanning sequences"):
        seq = session.sequences[pi]
        slen = _seq_len(seq)
        anchor = int(session.n_clues[pi]) if session.has_sep else int(session.n_clues[pi]) - 1
        if anchor < 0 or anchor + 1 >= slen:
            continue

        next_tok = int(seq[anchor + 1])
        if next_tok in _CONTROL or not (0 <= next_tok <= 728):
            continue

        board = _replay_board(seq, anchor)
        cands = _candidates(board)

        nr, nc, nd = decode_fill(next_tok)
        cell = nr * 9 + nc
        valid_tokens = _valid_tokens(cands)
        if next_tok not in valid_tokens:
            continue

        data.append({
            "pi": pi,
            "pos_b1": anchor,
            "pos_b2": anchor + 1,
            "digit": nd,
            "valid_tokens": valid_tokens,
            "substructures": _substructures_for_cell(cell),
        })

    print(f"Found: {len(data):,} initial states with legal next placements")
    if len(data) > max_n:
        idx = rng.choice(len(data), max_n, replace=False)
        data = [data[i] for i in sorted(idx)]
    print(f"Kept:  {len(data):,} pairs")
    return data


def _load_dataset_cache() -> list[dict] | None:
    if not DATASETS_CACHE.exists():
        return None
    print(f"Loading dataset from {DATASETS_CACHE}")
    with open(DATASETS_CACHE, "rb") as f:
        payload = pickle.load(f)
    data = payload["data"] if isinstance(payload, dict) and "data" in payload else payload
    print(f"  pairs: {len(data):,}")
    return data


def _save_dataset_cache(data: list[dict]):
    with open(DATASETS_CACHE, "wb") as f:
        pickle.dump(data, f)
    print(f"Saved dataset to {DATASETS_CACHE}")


def train_present_probes(session: ProbeSession) -> dict:
    probe_idx = session.index.at_step(PROBE_STEP).first_per_puzzle()
    train_mask, _ = session.split(probe_idx)
    grids = session.grids(probe_idx)
    g_train = [grids[i] for i in np.where(train_mask)[0]]

    vecs = {}
    for layer in tqdm(range(session.n_layers), desc="Training structure-present probes"):
        X = session.acts(probe_idx, layer=layer)[train_mask]
        vecs[layer] = {}
        for stype, sidx in tqdm(SUBTYPES, desc=f"  L{layer}", leave=False):
            y = STRUCTURE.build_targets(g_train, stype, sidx).astype(int)
            clf = STRUCTURE.fit(X, y)
            for d0 in range(9):
                w = clf.estimators_[d0].coef_[0].astype(np.float32)
                vecs[layer][(stype, sidx, d0)] = w / (np.linalg.norm(w) + EPS)
    return vecs


def _probe_cache_meta(cache_path: str, session: ProbeSession) -> dict:
    return {
        "cache_path": str(Path(cache_path)),
        "act_type": "post_mlp",
        "probe_step": PROBE_STEP,
        "n_layers": session.n_layers,
        "target": "structure_present_unit_norm",
    }


def _load_probe_cache(cache_path: str, session: ProbeSession) -> dict | None:
    if not PROBES_CACHE.exists():
        return None
    print(f"Loading probes from {PROBES_CACHE}")
    with open(PROBES_CACHE, "rb") as f:
        payload = pickle.load(f)
    expected = _probe_cache_meta(cache_path, session)
    if not isinstance(payload, dict) or payload.get("meta") != expected:
        print("  Probe cache metadata did not match this run; retraining.")
        return None
    return payload["probe_vecs"]


def _save_probe_cache(probe_vecs: dict, cache_path: str, session: ProbeSession):
    payload = {
        "meta": _probe_cache_meta(cache_path, session),
        "probe_vecs": probe_vecs,
    }
    with open(PROBES_CACHE, "wb") as f:
        pickle.dump(payload, f)
    print(f"Saved probes to {PROBES_CACHE}")


def _padded_tokens(seq: list[int], max_seq_len: int) -> jnp.ndarray:
    if len(seq) > max_seq_len:
        raise ValueError(f"Sequence length {len(seq)} exceeds model max_seq_len={max_seq_len}")
    padded = seq + [PAD_TOKEN] * (max_seq_len - len(seq))
    return jnp.asarray(padded, dtype=jnp.int32)[None, :]


def _make_logits_fns(model_inst):
    @jax.jit
    def clean_logits(params, tokens, read_pos):
        logits = model_inst.apply({"params": params}, tokens)
        return logits[0, read_pos, :729]

    @partial(jax.jit, static_argnames=("layer",))
    def patched_logits(params, tokens, delta, patch_pos, *, layer: int):
        logits = model_inst.apply(
            {"params": params},
            tokens,
            patch={"layer": layer, "pos": patch_pos, "delta": delta},
        )
        return logits[0, patch_pos, :729]

    return clean_logits, patched_logits


def run_ablation(
    samples: list[dict],
    probe_vecs: dict,
    params,
    model_inst,
    session: ProbeSession,
) -> dict:
    n_layers = model_inst.config.n_layers
    clean_logits_fn, patched_logits_fn = _make_logits_fns(model_inst)
    idx_b1 = _make_index(samples, "pos_b1")
    idx_b2 = _make_index(samples, "pos_b2")

    n = len(samples)
    clean_b1_top1_logit = np.zeros(n, dtype=np.float32)
    patched_b1_top1_logit = np.zeros((n, n_layers), dtype=np.float32)
    logit_drop = np.zeros((n, n_layers), dtype=np.float32)
    patched_top1_valid = np.zeros((n, n_layers), dtype=bool)
    patched_top1_changed = np.zeros((n, n_layers), dtype=bool)

    order = sorted(range(n), key=lambda i: samples[i]["pi"])
    current_pi = -1
    tokens_pi = None

    acts_b1_by_layer = {}
    acts_b2_by_layer = {}
    for layer in range(n_layers):
        acts_b1_by_layer[layer] = session.acts(idx_b1, layer=layer).astype(np.float32)
        acts_b2_by_layer[layer] = session.acts(idx_b2, layer=layer).astype(np.float32)

    for rank in tqdm(order, desc="Ablation"):
        sample = samples[rank]
        pi = sample["pi"]
        pos = sample["pos_b1"]
        d0 = sample["digit"] - 1
        pos_jax = jnp.asarray(pos, dtype=jnp.int32)

        if pi != current_pi:
            tokens_pi = _padded_tokens(session.sequences[pi], model_inst.config.max_seq_len)
            current_pi = pi

        clean = np.array(clean_logits_fn(params, tokens_pi, pos_jax))
        top1_tok = int(np.argmax(clean))
        clean_b1_top1_logit[rank] = clean[top1_tok]

        for layer in range(n_layers):
            x1 = acts_b1_by_layer[layer][rank]
            x2 = acts_b2_by_layer[layer][rank]
            delta = np.zeros(model_inst.config.d_model, dtype=np.float32)
            for stype, sidx in sample["substructures"]:
                w = probe_vecs[layer][(stype, sidx, d0)].astype(np.float32)
                delta += (float(w @ x2) - float(w @ x1)) * w

            patched = np.array(
                patched_logits_fn(
                    params,
                    tokens_pi,
                    jnp.asarray(delta, dtype=jnp.float32),
                    pos_jax,
                    layer=layer,
                )
            )
            patched_b1_top1_logit[rank, layer] = patched[top1_tok]
            logit_drop[rank, layer] = clean_b1_top1_logit[rank] - patched[top1_tok]
            patched_top1_tok = int(np.argmax(patched))
            patched_top1_valid[rank, layer] = patched_top1_tok in sample["valid_tokens"]
            patched_top1_changed[rank, layer] = patched_top1_tok != top1_tok

    return {
        "n_total": n,
        "mean_clean_b1_top1_logit": float(clean_b1_top1_logit.mean()),
        "mean_patched_b1_top1_logit": patched_b1_top1_logit.mean(axis=0),
        "mean_logit_drop": logit_drop.mean(axis=0),
        "stderr_logit_drop": logit_drop.std(axis=0, ddof=1) / np.sqrt(max(n, 1)),
        "patched_top1_valid_rate": patched_top1_valid.mean(axis=0),
        "patched_top1_changed_rate": patched_top1_changed.mean(axis=0),
    }


def save_plot_data(results: dict, path: Path = DATA_OUT):
    fields = [
        "layer",
        "mean_logit_drop",
        "stderr_logit_drop",
        "mean_clean_b1_top1_logit",
        "mean_patched_b1_top1_logit",
        "patched_top1_valid_rate",
        "patched_top1_changed_rate",
        "n_total",
    ]
    n_layers = len(results["mean_logit_drop"])
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for layer in range(n_layers):
            writer.writerow({
                "layer": layer,
                "mean_logit_drop": results["mean_logit_drop"][layer],
                "stderr_logit_drop": results["stderr_logit_drop"][layer],
                "mean_clean_b1_top1_logit": results["mean_clean_b1_top1_logit"],
                "mean_patched_b1_top1_logit": results["mean_patched_b1_top1_logit"][layer],
                "patched_top1_valid_rate": results["patched_top1_valid_rate"][layer],
                "patched_top1_changed_rate": results["patched_top1_changed_rate"][layer],
                "n_total": results["n_total"],
            })
    print(f"Saved plot data to {path}")


def load_plot_data(path: Path) -> dict:
    with open(path, newline="") as f:
        rows = sorted(csv.DictReader(f), key=lambda r: int(r["layer"]))

    return {
        "n_total": int(rows[0]["n_total"]),
        "mean_clean_b1_top1_logit": float(rows[0].get("mean_clean_b1_top1_logit", "nan")),
        "mean_patched_b1_top1_logit": np.array([float(r.get("mean_patched_b1_top1_logit", "nan")) for r in rows]),
        "mean_logit_drop": np.array([float(r["mean_logit_drop"]) for r in rows]),
        "stderr_logit_drop": np.array([float(r["stderr_logit_drop"]) for r in rows]),
        "patched_top1_valid_rate": np.array([float(r["patched_top1_valid_rate"]) for r in rows]),
        "patched_top1_changed_rate": np.array([float(r["patched_top1_changed_rate"]) for r in rows]),
    }


def plot(results: dict, n_layers: int):
    x = np.arange(n_layers)
    fig, ax = plt.subplots(figsize=(4, 2.2))
    ax.errorbar(
        x,
        results["mean_logit_drop"],
        yerr=results["stderr_logit_drop"],
        fmt="o-",
        lw=2,
        capsize=2,
        color="tab:blue",
        label="top-1 logit drop",
    )
    ax.axhline(0, color="#888888", ls=":", lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels([f"L{layer + 1}" for layer in x])
    ax.set_xlabel("Patched layer")
    ax.set_ylabel("top-1 logit drop")
    # ax.set_title(
    #     f"Structure-present patch - legal initial placements\n"
    #     f"copy B2 row/col/box digit coordinates into B1 (n={results['n_total']})"
    # )
    ax.legend(frameon=False)
    sns.despine(fig)
    fig.tight_layout()
    fig.savefig(FIGURE_OUT, bbox_inches="tight")
    print(f"Saved {FIGURE_OUT}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", default=DEFAULT_CACHE)
    parser.add_argument("--rebuild", action="store_true",
                        help="ignore cached dataset and rebuild from scratch")
    parser.add_argument("--rebuild-probes", action="store_true",
                        help="ignore cached probe directions and retrain them")
    parser.add_argument("--max-samples", type=int, default=MAX_SAMPLES)
    parser.add_argument("--data", default=None,
                        help="load plotted per-layer CSV data instead of running ablation")
    args = parser.parse_args()

    if args.data:
        results = load_plot_data(Path(args.data))
        plot(results, len(results["mean_logit_drop"]))
        return

    rng = np.random.default_rng(SEED)
    ckpt = args.cache.replace("activations.npz", "checkpoint")

    session = ProbeSession(args.cache, act_type="post_mlp")
    params, model_inst = load_checkpoint(ckpt)
    n_layers = model_inst.config.n_layers

    probe_idx_split = session.index.at_step(PROBE_STEP).first_per_puzzle()
    _, test_mask = session.split(probe_idx_split)
    test_pis = set(probe_idx_split.puzzle_idx[test_mask].tolist())

    data = None if args.rebuild else _load_dataset_cache()
    if data is None:
        data = build_dataset(
            session,
            rng,
            args.max_samples,
            test_pis,
        )
        _save_dataset_cache(data)

    probe_vecs = None if args.rebuild_probes else _load_probe_cache(args.cache, session)
    if probe_vecs is None:
        probe_vecs = train_present_probes(session)
        _save_probe_cache(probe_vecs, args.cache, session)

    print(f"Running ablation on {len(data):,} pairs...")
    results = run_ablation(data, probe_vecs, params, model_inst, session)

    print(f"Mean clean B1 top-1 logit: {results['mean_clean_b1_top1_logit']:.3f}")
    for layer in range(n_layers):
        print(
            f"  L{layer}  drop={results['mean_logit_drop'][layer]:.3f}"
            f" +/- {results['stderr_logit_drop'][layer]:.3f}"
            f"  patched_logit={results['mean_patched_b1_top1_logit'][layer]:.3f}"
            f"  valid_top1={results['patched_top1_valid_rate'][layer]:.3f}"
            f"  changed_top1={results['patched_top1_changed_rate'][layer]:.3f}"
        )

    np.savez(RESULTS_OUT, **{k: np.asarray(v) for k, v in results.items()})
    print(f"Saved results to {RESULTS_OUT}")
    save_plot_data(results)
    plot(results, n_layers)


if __name__ == "__main__":
    main()
