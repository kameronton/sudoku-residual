"""Analyze how mean activations shift with token position.

Checks whether dimension-wise mean of residual stream activations drifts as a
function of:
  1. Absolute token position in the sequence
  2. Relative position with respect to the END_CLUES (token 729) token

Usage:
    uv run python scripts/analyze_activation_shift.py \
        --cache_path results/3M-backtracking-packing/activations.npz
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

PAD_TOKEN = 730
END_CLUES_TOKEN = 729


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cache_path", default="results/3M-backtracking-packing/activations.npz")
    p.add_argument("--output", default="activation_shift.png")
    p.add_argument("--layers", type=int, nargs="+", default=None,
                   help="Layers to plot (default: all)")
    p.add_argument("--max_abs_pos", type=int, default=None,
                   help="Truncate absolute position plot at this position")
    p.add_argument("--rel_range", type=int, nargs=2, default=[-5, 80],
                   help="Relative position range to plot [lo, hi]")
    p.add_argument("--min_samples", type=int, default=50,
                   help="Minimum number of puzzles with content at a position to include")
    return p.parse_args()


def main():
    args = parse_args()

    # --- Load data ---
    print(f"Loading {args.cache_path} ...")
    meta = np.load(args.cache_path, allow_pickle=False)
    sequences = meta["sequences"]          # (N, T) int16
    n_clues = meta["n_clues"]              # (N,) int16

    acts_path = args.cache_path.replace(".npz", "_acts.npy")
    acts = np.load(acts_path, mmap_mode="r")  # (N, L, T, D)
    N, L, T, D = acts.shape
    print(f"  activations: {acts.shape}  sequences: {sequences.shape}")

    layers = args.layers if args.layers is not None else list(range(L))

    # --- Build mask: which (puzzle, position) has real content (not PAD) ---
    pad_mask = sequences != PAD_TOKEN  # (N, T) bool

    # END_CLUES positions per puzzle
    end_clues_pos = n_clues.astype(int)  # position of END_CLUES token

    # =========================================================
    # 1. ABSOLUTE POSITION: mean_act[pos] = mean over puzzles that
    #    have non-pad content at `pos`
    # =========================================================
    max_pos = args.max_abs_pos if args.max_abs_pos else T
    abs_pos_range = range(max_pos)

    # count of puzzles contributing to each position
    abs_counts = pad_mask[:, :max_pos].sum(axis=0)  # (max_pos,)
    valid_abs = abs_counts >= args.min_samples

    print("Computing absolute-position means...")
    # Shape: (L, max_pos, D) — but we process layer-by-layer to stay memory-friendly
    abs_mean_norm = np.full((len(layers), max_pos), np.nan)  # L2 norm of mean vec
    abs_mean_shift = np.full((len(layers), max_pos), np.nan)  # shift from pos-0 mean

    for li, layer in enumerate(layers):
        print(f"  layer {layer}/{L-1}", end="\r")
        layer_acts = np.array(acts[:, layer, :max_pos, :], dtype=np.float32)  # (N, max_pos, D)
        for pos in abs_pos_range:
            mask = pad_mask[:, pos]
            if mask.sum() < args.min_samples:
                continue
            mean_vec = layer_acts[mask, pos, :].mean(axis=0)  # (D,)
            abs_mean_norm[li, pos] = np.linalg.norm(mean_vec)
        # Shift: cosine similarity between mean at each pos and mean at pos 0
        ref_pos = 0
        while not valid_abs[ref_pos]:
            ref_pos += 1
        ref_vec = None
        for pos in abs_pos_range:
            mask = pad_mask[:, pos]
            if mask.sum() < args.min_samples:
                continue
            mean_vec = layer_acts[mask, pos, :].mean(axis=0)
            if ref_vec is None:
                ref_vec = mean_vec
                abs_mean_shift[li, pos] = 0.0
            else:
                diff = mean_vec - ref_vec
                abs_mean_shift[li, pos] = np.linalg.norm(diff)
    print()

    # =========================================================
    # 2. RELATIVE POSITION w.r.t. END_CLUES
    # =========================================================
    lo, hi = args.rel_range
    rel_positions = range(lo, hi + 1)
    n_rel = len(rel_positions)
    rel_counts = np.zeros(n_rel, dtype=int)
    rel_mean_norm = np.full((len(layers), n_rel), np.nan)
    rel_mean_shift = np.full((len(layers), n_rel), np.nan)

    # Count samples per relative position first
    for i, rel in enumerate(rel_positions):
        abs_positions_for_rel = end_clues_pos + rel
        in_bounds = (abs_positions_for_rel >= 0) & (abs_positions_for_rel < T)
        valid_puzzles = np.where(in_bounds)[0]
        if len(valid_puzzles) == 0:
            continue
        abs_pos_arr = abs_positions_for_rel[valid_puzzles]
        has_content = pad_mask[valid_puzzles, abs_pos_arr]
        rel_counts[i] = has_content.sum()

    # Pre-compute the reference vector (mean at END_CLUES, rel=0) per layer
    rel0_idx = list(rel_positions).index(0)

    def _mean_vec_at_rel(layer_acts, rel):
        abs_positions_for_rel = end_clues_pos + rel
        in_bounds = (abs_positions_for_rel >= 0) & (abs_positions_for_rel < T)
        valid_puzzles = np.where(in_bounds)[0]
        abs_pos_arr = abs_positions_for_rel[valid_puzzles]
        has_content = pad_mask[valid_puzzles, abs_pos_arr]
        sel_puzzles = valid_puzzles[has_content]
        sel_abs = abs_pos_arr[has_content]
        if len(sel_puzzles) < args.min_samples:
            return None
        return layer_acts[sel_puzzles, sel_abs, :].mean(axis=0)

    print("Computing relative-position means...")
    for li, layer in enumerate(layers):
        print(f"  layer {layer}/{L-1}", end="\r")
        layer_acts = np.array(acts[:, layer, :, :], dtype=np.float32)  # (N, T, D)
        ref_vec = _mean_vec_at_rel(layer_acts, 0)  # always anchored at END_CLUES (rel=0)
        for i, rel in enumerate(rel_positions):
            if rel_counts[i] < args.min_samples:
                continue
            mean_vec = _mean_vec_at_rel(layer_acts, rel)
            if mean_vec is None:
                continue
            rel_mean_norm[li, i] = np.linalg.norm(mean_vec)
            if ref_vec is not None:
                rel_mean_shift[li, i] = np.linalg.norm(mean_vec - ref_vec)
    print()

    # =========================================================
    # Plot
    # =========================================================
    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)
    cmap = plt.cm.tab10

    # Top-left: abs pos — L2 norm of mean
    ax1 = fig.add_subplot(gs[0, 0])
    for li, layer in enumerate(layers):
        xs = np.where(~np.isnan(abs_mean_norm[li]))[0]
        ys = abs_mean_norm[li, xs]
        ax1.plot(xs, ys, label=f"L{layer}", color=cmap(li % 10), lw=1.2, alpha=0.85)
    ax1.set_title("||mean activation|| vs absolute position")
    ax1.set_xlabel("Absolute token position")
    ax1.set_ylabel("L2 norm of mean activation vector")
    ax1.legend(fontsize=7, ncol=2)
    ax1.grid(True, alpha=0.3)

    # Top-right: abs pos — L2 drift from pos-0
    ax2 = fig.add_subplot(gs[0, 1])
    for li, layer in enumerate(layers):
        xs = np.where(~np.isnan(abs_mean_shift[li]))[0]
        ys = abs_mean_shift[li, xs]
        ax2.plot(xs, ys, label=f"L{layer}", color=cmap(li % 10), lw=1.2, alpha=0.85)
    ax2.set_title("Drift of mean activation from position 0 (absolute)")
    ax2.set_xlabel("Absolute token position")
    ax2.set_ylabel("||mean(pos) - mean(pos_0)||")
    ax2.legend(fontsize=7, ncol=2)
    ax2.grid(True, alpha=0.3)

    # Bottom-left: rel pos — L2 norm
    ax3 = fig.add_subplot(gs[1, 0])
    rel_xs = np.array(list(rel_positions))
    for li, layer in enumerate(layers):
        valid = ~np.isnan(rel_mean_norm[li])
        ax3.plot(rel_xs[valid], rel_mean_norm[li, valid],
                 label=f"L{layer}", color=cmap(li % 10), lw=1.2, alpha=0.85)
    ax3.axvline(0, color="black", lw=1.5, ls="--", label="END_CLUES")
    ax3.set_title("||mean activation|| vs relative position (re: END_CLUES)")
    ax3.set_xlabel("Token position relative to END_CLUES")
    ax3.set_ylabel("L2 norm of mean activation vector")
    ax3.legend(fontsize=7, ncol=2)
    ax3.grid(True, alpha=0.3)

    # Bottom-right: rel pos — drift from END_CLUES position
    ax4 = fig.add_subplot(gs[1, 1])
    for li, layer in enumerate(layers):
        valid = ~np.isnan(rel_mean_shift[li])
        ax4.plot(rel_xs[valid], rel_mean_shift[li, valid],
                 label=f"L{layer}", color=cmap(li % 10), lw=1.2, alpha=0.85)
    ax4.axvline(0, color="black", lw=1.5, ls="--", label="END_CLUES")
    ax4.set_title("Drift from END_CLUES mean (relative)")
    ax4.set_xlabel("Token position relative to END_CLUES")
    ax4.set_ylabel("||mean(pos) - mean(END_CLUES)||")
    ax4.legend(fontsize=7, ncol=2)
    ax4.grid(True, alpha=0.3)

    fig.suptitle(f"Activation mean shift analysis\n{args.cache_path}", fontsize=11)
    plt.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"Saved plot to {args.output}")

    # Also print a quick numerical summary
    print("\n=== Numerical summary (last layer) ===")
    li = len(layers) - 1
    layer = layers[li]
    print(f"Layer {layer}:")
    print(f"  Abs pos 0:   norm={abs_mean_norm[li, 0]:.3f}")
    mid = max_pos // 2
    print(f"  Abs pos {mid}: norm={abs_mean_norm[li, mid]:.3f}, drift={abs_mean_shift[li, mid]:.3f}")

    print(f"  Rel pos  0 (END_CLUES): norm={rel_mean_norm[li, rel0_idx]:.3f}, drift={rel_mean_shift[li, rel0_idx]:.3f}")
    pos10_idx = list(rel_positions).index(10) if 10 in rel_positions else None
    if pos10_idx is not None:
        print(f"  Rel pos +10:            norm={rel_mean_norm[li, pos10_idx]:.3f}, drift={rel_mean_shift[li, pos10_idx]:.3f}")


if __name__ == "__main__":
    main()
