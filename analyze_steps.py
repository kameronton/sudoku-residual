"""Analyze solve-step positions per cell across Sudoku traces."""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from data import decode_fill, SEP_TOKEN, PAD_TOKEN


def load_sequences(path: str) -> np.ndarray:
    """Load and concatenate all sequence splits from NPZ."""
    npz = np.load(path)
    parts = []
    for key in ("sequences_train", "sequences_val", "sequences_test"):
        if key in npz:
            parts.append(npz[key])
    if parts:
        return np.concatenate(parts, axis=0)
    return npz["sequences"]


def compute_step_positions(sequences: np.ndarray) -> np.ndarray:
    """Return (N, 9, 9) array of solve-step positions. Clue cells = NaN."""
    N = sequences.shape[0]
    steps = np.full((N, 9, 9), np.nan)
    for i in range(N):
        seq = sequences[i]
        sep_idx = np.where(seq == SEP_TOKEN)[0]
        if len(sep_idx) == 0:
            continue
        sep_idx = sep_idx[0]
        for step, tok in enumerate(seq[sep_idx + 1:]):
            if tok == PAD_TOKEN:
                break
            row, col, _ = decode_fill(int(tok))
            steps[i, row, col] = step
    return steps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--traces_path", default="data/traces.npz")
    parser.add_argument("--output", default="step_histograms.png")
    args = parser.parse_args()

    print(f"Loading {args.traces_path}...")
    sequences = load_sequences(args.traces_path)
    print(f"  {sequences.shape[0]} sequences loaded")

    print("Computing step positions...")
    steps = compute_step_positions(sequences)

    # Per-row mean step position (ignoring clues = NaN)
    print("\nMean solve-step by row:")
    row_means = np.nanmean(steps, axis=(0, 2))
    for r in range(9):
        marker = " <--" if r == 8 else ""
        print(f"  Row {r}: {row_means[r]:.2f}{marker}")

    # 9x9 histogram grid
    fig, axes = plt.subplots(9, 9, figsize=(18, 18), constrained_layout=True)
    max_step = int(np.nanmax(steps))
    bins = np.arange(0, max_step + 2) - 0.5

    for r in range(9):
        for c in range(9):
            ax = axes[r][c]
            vals = steps[:, r, c]
            vals = vals[~np.isnan(vals)]
            if len(vals) > 0:
                ax.hist(vals, bins=bins, color="steelblue", edgecolor="none", density=True)
                ax.set_title(f"({r},{c}) μ={np.mean(vals):.1f}", fontsize=7)
            else:
                ax.set_title(f"({r},{c}) clue", fontsize=7, color="gray")
            ax.tick_params(labelsize=5)
            ax.set_xlim(-0.5, max_step + 0.5)

    fig.suptitle("Distribution of solve-step position per cell", fontsize=14)
    fig.savefig(args.output, dpi=150)
    print(f"\nSaved {args.output}")


if __name__ == "__main__":
    main()
