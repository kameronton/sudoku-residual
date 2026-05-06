"""Figure: per-cell candidate-count sensitivity of final MLP neurons.

Scans every neuron in the last MLP layer and ranks by the gap between
mean preactivation at candidate_count=1 vs. the runner-up count.

Usage:
    uv run python plots/scripts/fig_neuron_candidate_sensitivity.py
    uv run python plots/scripts/fig_neuron_candidate_sensitivity.py --data plots/data/fig_neuron_candidate_sensitivity.csv
    uv run python plots/scripts/fig_neuron_candidate_sensitivity.py --boxplot 17 42   # cell 17, neuron 42
    uv run python plots/scripts/fig_neuron_candidate_sensitivity.py --layer 5 --max-samples 30000
"""

import argparse
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm
from jax.nn import gelu

from sudoku.activations import load_checkpoint
from sudoku.data_bt import PUSH_TOKEN, POP_TOKEN, SUCCESS_TOKEN
from sudoku.probes.session import ProbeSession

_CONTROL_TOKENS = {PUSH_TOKEN, POP_TOKEN, SUCCESS_TOKEN}

sns.set_theme(style="ticks", context="paper")

plt.rcParams.update({
    'font.size': 10,
    'axes.titlesize': 10,
    'axes.labelsize': 10,
    'xtick.labelsize': 8,     # Will become 8pt
    'ytick.labelsize': 8,
    'legend.fontsize': 9,     # Will become 9pt
    'font.family': 'serif',
    'pdf.fonttype': 42
})

DEFAULT_CACHE = "results/3M-backtracking-packing/activations.npz"
DEFAULT_CKPT  = "results/3M-backtracking-packing/checkpoint"
DATA_PATH = "plots/data/fig_neuron_candidate_sensitivity.csv"

BATCH = 512


# Peer cells for each of the 81 cells (row ∪ col ∪ box, excluding self)
_PEERS: list[list[int]] = []
for _cell in range(81):
    _r, _c = _cell // 9, _cell % 9
    _br, _bc = (_r // 3) * 3, (_c // 3) * 3
    _p: set[int] = set()
    for _j in range(9):
        _p.add(_r * 9 + _j)
        _p.add(_j * 9 + _c)
    for _dr in range(3):
        for _dc in range(3):
            _p.add((_br + _dr) * 9 + (_bc + _dc))
    _p.discard(_cell)
    _PEERS.append(sorted(_p))


def _apply_ln(x: np.ndarray, scale: np.ndarray, bias: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    x = x.astype(np.float32)
    mean = x.mean(-1, keepdims=True)
    var = x.var(-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps) * scale + bias


def _compute_candidate_counts(grids: list[str]) -> np.ndarray:
    """Return (N, 81) uint8 array: candidate count per empty cell, 0 for filled."""
    N = len(grids)
    arr = np.frombuffer("".join(grids).encode(), dtype=np.uint8).reshape(N, 81)
    arr = (arr - ord("0")).astype(np.uint8)
    counts = np.zeros((N, 81), dtype=np.uint8)
    for cell in range(81):
        is_empty = arr[:, cell] == 0
        if not is_empty.any():
            continue
        peer_vals = arr[:, _PEERS[cell]]  # (N, n_peers)
        n_used = np.zeros(N, dtype=np.uint8)
        for d in range(1, 10):
            n_used += (peer_vals == d).any(axis=1).view(np.uint8)
        counts[:, cell] = np.where(is_empty, 9 - n_used, 0)
    return counts


def _load_session_and_mlp(cache: str, ckpt: str, layer: int):
    session = ProbeSession(cache, act_type="post_attn")
    params, model = load_checkpoint(ckpt)
    if layer < 0:
        layer = model.config.n_layers + layer
    blk = params[f"block_{layer}"]
    W_in     = np.asarray(blk["Dense_0"]["kernel"], dtype=np.float32)
    b_in     = np.asarray(blk["Dense_0"]["bias"],   dtype=np.float32)
    ln_scale = np.asarray(blk["LayerNorm_1"]["scale"], dtype=np.float32)
    ln_bias  = np.asarray(blk["LayerNorm_1"]["bias"],  dtype=np.float32)
    return session, W_in, b_in, ln_scale, ln_bias, layer


def _neuron_acts(x: np.ndarray, W_in, b_in, ln_scale, ln_bias) -> np.ndarray:
    return np.asarray(gelu(_apply_ln(x, ln_scale, ln_bias) @ W_in + b_in), dtype = np.float32)


def compute_scan(args) -> pd.DataFrame:
    session, W_in, b_in, ln_scale, ln_bias, layer = _load_session_and_mlp(
        args.cache, args.ckpt, args.layer
    )
    d_ff = W_in.shape[1]
    print(f"Scanning layer {layer}, d_ff={d_ff}")

    idx = session.index
    idx = idx.filter(~np.isin(idx.tokens, list(_CONTROL_TOKENS)))
    if len(idx) > args.max_samples:
        rng = np.random.default_rng(args.seed)
        sel = np.sort(rng.choice(len(idx), args.max_samples, replace=False))
        idx = idx[sel]
    N = len(idx)
    print(f"Using {N:,} samples")

    print("Building board states...")
    grids = session.grids(idx)
    print("Computing candidate counts...")
    cand_counts = _compute_candidate_counts(grids)  # (N, 81)

    sum_acts = np.zeros((81, 10, d_ff), dtype=np.float64)
    n_acts   = np.zeros((81, 10), dtype=np.int64)

    for lo in tqdm(range(0, N, BATCH), desc="accumulating"):
        hi = min(lo + BATCH, N)
        x = session.acts(idx[lo:hi], layer=layer)
        h = _neuron_acts(x, W_in, b_in, ln_scale, ln_bias)  # (B, d_ff)
        cc = cand_counts[lo:hi]  # (B, 81)
        for c in range(0, 10):
            mask = cc == c  # (B, 81) bool
            n_acts[:, c] += mask.sum(axis=0)
            sum_acts[:, c] += mask.T.astype(np.float32) @ h  # (81, d_ff)

    # Compute per-(cell, neuron) gap
    mean_acts = np.where(
        n_acts[:, :, None] > 0,
        sum_acts / np.maximum(n_acts[:, :, None], 1),
        np.nan,
    ).astype(np.float32)

    rows = []
    for cell in range(81):
        if n_acts[cell, 1] < args.min_n:
            continue
        valid = [c for c in range(0, 10) if c != 1 and n_acts[cell, c] >= args.min_n]
        if not valid:
            continue
        other_means = np.stack([mean_acts[cell, c] for c in valid])  # (n_valid, d_ff)
        max_other = np.nanmax(other_means, axis=0)                    # (d_ff,)
        gap = mean_acts[cell, 1] - max_other                          # (d_ff,)

        for neuron in range(d_ff):
            g = float(gap[neuron])
            if np.isnan(g) or float(mean_acts[cell, 1, neuron]) <= 0 or g < 3.0:
                continue
            argmax_c = valid[int(np.argmax(other_means[:, neuron]))]
            rows.append({
                "cell": cell,
                "neuron": neuron,
                "mean_count1": float(mean_acts[cell, 1, neuron]),
                "max_mean_not1": float(max_other[neuron]),
                "gap": g,
                "argmax_count_not1": argmax_c,
                "n_count1": int(n_acts[cell, 1]),
            })

    df = pd.DataFrame(rows).sort_values("gap", ascending=False).reset_index(drop=True)
    return df


def print_top(df: pd.DataFrame, top: int = 20):
    cell_counts = df['cell'].value_counts()
    print(f"Total unique cells covered: {len(cell_counts)} / 81")
    print(f"Cells with duplicate neurons: {sum(cell_counts > 1)}\n")

    # 2. Print the exact neuron mapping per cell with Sudoku coordinates
    print("--- Exact Neuron Mapping ---")
    # Group the neurons by cell into lists
    cell_to_neurons = df.sort_values('cell').groupby('cell')['neuron'].apply(list)

    for cell, neurons in cell_to_neurons.items():
        # Convert flat index (0-80) to Row/Col (0-8)
        row = cell // 9
        col = cell % 9
        
        # Format a nice output string
        neuron_str = ", ".join([str(int(n)) for n in neurons])
        print(f"Cell {cell:<2} (R{row}C{col}): {len(neurons)} neuron(s) -> [{neuron_str}]")
    
    print(f"\n{'rank':>4}  {'cell':>4}  {'neuron':>6}  {'mean1':>8}  {'runner-up':>9}  {'gap':>8}  {'arg':>4}  {'n1':>6}")
    for rank, row in df.head(top).iterrows():
        r, c = int(row["cell"]) // 9, int(row["cell"]) % 9
        cell_label = f"({r},{c})"
        print(
            f"{rank+1:>4}  {cell_label:>6}  {int(row['neuron']):>6}  "
            f"{row['mean_count1']:>8.4f}  {row['max_mean_not1']:>9.4f}  "
            f"{row['gap']:>8.4f}  {int(row['argmax_count_not1']):>4}  {int(row['n_count1']):>6}"
        )


def plot_boxplot(args):
    cell   = args.boxplot[0]
    neuron = args.boxplot[1]

    session, W_in, b_in, ln_scale, ln_bias, layer = _load_session_and_mlp(
        args.cache, args.ckpt, args.layer
    )

    idx = session.index
    idx = idx.filter(~np.isin(idx.tokens, list(_CONTROL_TOKENS)))
    if len(idx) > args.max_samples:
        rng = np.random.default_rng(args.seed)
        sel = np.sort(rng.choice(len(idx), args.max_samples, replace=False))
        idx = idx[sel]
    N = len(idx)

    print("Building board states...")
    grids = session.grids(idx)
    print("Computing candidate counts...")
    cand_counts = _compute_candidate_counts(grids)[:, cell]  # (N,)

    neuron_acts = np.empty(N, dtype=np.float32)
    for lo in tqdm(range(0, N, BATCH), desc="collecting activations"):
        hi = min(lo + BATCH, N)
        x = session.acts(idx[lo:hi], layer=layer)
        h = _neuron_acts(x, W_in, b_in, ln_scale, ln_bias)
        neuron_acts[lo:hi] = h[:, neuron]

    by_count: dict[int, list[float]] = defaultdict(list)
    for i in range(N):
        c = int(cand_counts[i])
        if 0 <= c <= 9:
            by_count[c].append(float(neuron_acts[i]))

    present    = sorted(by_count.keys())
    data       = [by_count[c] for c in present]
    n_per      = [len(d) for d in data]
    xlabels    = ["filled" if c == 0 else str(c) for c in present]

    r, col = cell // 9, cell % 9
    fig, ax = plt.subplots(figsize=(5.5, 2.5))

    ax.boxplot(data, tick_labels=xlabels, 
               notch=False, 
               patch_artist=True,
               boxprops=dict(facecolor="white", edgecolor="black", alpha=0.85),
               medianprops=dict(color="tab:orange", linewidth=1.5),
               whiskerprops=dict(linewidth=0.8),
               capprops=dict(linewidth=0.8),
               flierprops=dict(visible=False),)

    rng = np.random.default_rng(0)
    for i, vals in enumerate(data):
        sample = rng.choice(vals, min(500, len(vals)), replace=False)
        # xs = rng.uniform(i + 0.75, i + 1.25, len(sample))
        gitter = rng.uniform(-0.08, 0.08, size = len(sample))
        ax.scatter(i + 1 + gitter, sample, s=5, color="black", alpha=0.15, linewidths=0, zorder=2)

    ax.set_xlabel("candidate count")
    ax.set_ylabel("GELU activation")
    ax.set_ylim(-0.2, 9)
    ax.set_title(f"Neuron {neuron} — cell ({r},{col})", fontsize=10)

    sns.despine(fig)
    fig.tight_layout()
    out = f"plots/figures/fig_neuron_{neuron}_cell_{cell}_boxplot.pdf"
    fig.savefig(out, bbox_inches="tight")
    print(f"Saved {out}")

import numpy as np
import pandas as pd
from sudoku.activations import load_checkpoint

def _get_cell_tokens(cell: int) -> list[int]:
    """
    Returns the token indices for the 9 digits in a given cell.
    Adjust this logic to match your exact Sudoku tokenizer vocabulary!
    """
    # Example: assuming placement tokens start at index 0 and are ordered sequentially
    return [cell * 9 + d for d in range(9)]


def compute_static_dla_metrics(filtered_df: pd.DataFrame, ckpt: str, layer: int) -> pd.DataFrame:
    """
    Computes the Static DLA geometry for a filtered list of neurons.
    filtered_df: The DataFrame containing 'cell' and 'neuron' columns (your 92 neurons).
    """
    # 1. Load weights
    params, model = load_checkpoint(ckpt)
    if layer < 0:
        layer = model.config.n_layers + layer
        
    # 1. Load the matrices and the LayerNorm scale
    W_out = np.asarray(params[f"block_{layer}"]["Dense_1"]["kernel"], dtype=np.float32)
    W_U = np.asarray(params["lm_head"]["kernel"], dtype=np.float32)
    ln_scale = np.asarray(params["LayerNorm_0"]["scale"], dtype=np.float32)
    
    # 2. FOLDING: Multiply the scale into the unembedding matrix upfront.
    # ln_scale is (d_model,), W_U is (d_model, vocab_size). 
    # This scales every row of W_U by the corresponding dimension's LN scale.
    W_U_folded = ln_scale[:, None] * W_U
    
    results = []
    
    for _, row in filtered_df.iterrows():
        neuron_idx = int(row["neuron"])
        target_cell = int(row["cell"])
        
        neuron_idx = int(row["neuron"])
        target_cell = int(row["cell"])
        
        # 3. Extract the neuron vector
        # Assuming W_out is shape (d_ff, d_model) based on standard Flax Dense layout
        neuron_vector = W_out[neuron_idx, :] 
        
        # 4. Mean-center the vector (simulating LayerNorm's mean subtraction)
        neuron_vector_centered = neuron_vector - np.mean(neuron_vector)
        
        # 5. Project to logits using the folded matrix
        logits = neuron_vector_centered @ W_U_folded # Shape: (vocab_size,)
        
        # 3. Separate target tokens from all other placement tokens
        target_tokens = _get_cell_tokens(target_cell)
        
        all_placement_tokens = []
        for c in range(81):
            all_placement_tokens.extend(_get_cell_tokens(c))
            
        other_tokens = list(set(all_placement_tokens) - set(target_tokens))
        
        # 4. Extract the target and other logits
        target_logits = logits[target_tokens]
        other_logits = logits[other_tokens]
        
        # 5. Calculate Metrics
        results.append({
            "cell": target_cell,
            "neuron": neuron_idx,
            "target_mean": np.mean(target_logits),
            "target_std": np.std(target_logits),
            "other_mean": np.mean(other_logits),
            "other_std": np.std(other_logits),
            "spatial_ratio": np.mean(np.abs(target_logits)) / (np.mean(np.abs(other_logits)) + 1e-6)
        })
        
    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", default=DEFAULT_CACHE)
    parser.add_argument("--ckpt",  default=DEFAULT_CKPT)
    parser.add_argument("--layer", type=int, default=-1, help="MLP layer (-1 = last)")
    parser.add_argument("--top",   type=int, default=20, help="top neurons to print")
    parser.add_argument("--min-n", type=int, default=30, dest="min_n",
                        help="minimum samples per bin to include a (cell, count) pair")
    parser.add_argument("--max-samples", type=int, default=50_000, dest="max_samples")
    parser.add_argument("--seed",  type=int, default=42)
    parser.add_argument("--data",  default=None, help="precomputed CSV — skip scan")
    parser.add_argument("--boxplot", type=int, nargs=2, metavar=("CELL", "NEURON"),
                        help="draw boxplot for this cell and neuron index")
    args = parser.parse_args()

    if args.boxplot:
        plot_boxplot(args)
        return

    if args.data:
        df = pd.read_csv(args.data)
    else:
        df = compute_scan(args)
        df.to_csv(DATA_PATH, index=False)
        print(f"Saved {DATA_PATH}")

    dla_metrics_df = compute_static_dla_metrics(df, ckpt=DEFAULT_CKPT, layer=-1)
    print(dla_metrics_df.describe())




if __name__ == "__main__":
    main()
