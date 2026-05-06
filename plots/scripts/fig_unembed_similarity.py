"""Figure: pairwise cosine similarity of unembedding vectors for placement tokens (0–728).

Usage:
    uv run python plots/scripts/fig_unembed_similarity.py
    uv run python plots/scripts/fig_unembed_similarity.py --ckpt results/3M-lr1e3/checkpoint
    uv run python plots/scripts/fig_unembed_similarity.py --data plots/data/fig_unembed_similarity.npy
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

from sudoku.activations import load_checkpoint

sns.set_theme(style="ticks", font="Avenir")

DEFAULT_CKPT = "results/3M-backtracking-packing/checkpoint"
OUTPUT = "plots/figures/fig_unembed_similarity.pdf"
DATA_PATH = "plots/data/fig_unembed_similarity.npy"


def compute_data(ckpt_dir: str) -> np.ndarray:
    params, _ = load_checkpoint(ckpt_dir)
    W = np.array(params["lm_head"]["kernel"])  # (d_model, vocab_size)
    W_place = W[:, :729]                        # placement tokens only
    norms = np.linalg.norm(W_place, axis=0, keepdims=True)
    W_norm = W_place / (norms + 1e-9)
    return W_norm.T @ W_norm                    # (729, 729)


def plot(sim: np.ndarray):
    fig, ax = plt.subplots(figsize=(10, 9))

    im = ax.imshow(sim, vmin=-1, vmax=1, cmap="RdBu_r",
                   interpolation="nearest", aspect="equal", origin="upper")

    for pos in range(0, 730, 81):
        ax.axhline(pos - 0.5, color="black", linewidth=0.8, alpha=0.6)
        ax.axvline(pos - 0.5, color="black", linewidth=0.8, alpha=0.6)
    for pos in range(0, 730, 9):
        ax.axhline(pos - 0.5, color="black", linewidth=0.2, alpha=0.3)
        ax.axvline(pos - 0.5, color="black", linewidth=0.2, alpha=0.3)

    row_ticks = [r * 81 + 40 for r in range(9)]
    ax.set_xticks(row_ticks)
    ax.set_yticks(row_ticks)
    ax.set_xticklabels([f"row {r+1}" for r in range(9)], fontsize=12)
    ax.set_yticklabels([f"row {r+1}" for r in range(9)], fontsize=12)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Cosine similarity", fontsize=10)

    # Inset: zoom into row 6 × row 6 (tokens 405–485)
    axins = ax.inset_axes([0.57, 0.57, 0.41, 0.41])
    axins.imshow(sim, vmin=-1, vmax=1, cmap="RdBu_r",
                 interpolation="nearest", aspect="equal", origin="upper")
    axins.set_xlim(404.5, 485.5)
    axins.set_ylim(485.5, 404.5)
    for pos in range(405, 486, 9):
        axins.axhline(pos - 0.5, color="black", linewidth=0.4, alpha=0.5)
        axins.axvline(pos - 0.5, color="black", linewidth=0.4, alpha=0.5)
    axins.set_xticks([])
    axins.set_yticks([])
    for spine in axins.spines.values():
        spine.set_linewidth(2)

    ax.add_patch(mpatches.Rectangle(
        (404.5, 404.5), 81, 81,
        fill=False, edgecolor="black", alpha=0.6, linewidth=2,
    ))
    for xyA, xyB in [
        ((404.5, 404.5), (0, 1)),
        ((485.5, 485.5), (1, 0)),
    ]:
        fig.add_artist(mpatches.ConnectionPatch(
            xyA=xyA, coordsA=ax.transData,
            xyB=xyB, coordsB=axins.transAxes,
            color="black", alpha=0.6, linewidth=2,
        ))

    sns.despine(fig, left=True, bottom=True)
    fig.tight_layout()
    fig.savefig(OUTPUT, bbox_inches="tight")
    print(f"Saved {OUTPUT}")


def print_group_stats(sim: np.ndarray):
    tokens = np.arange(729)
    rows   = tokens // 81
    cols   = (tokens % 81) // 9
    digits = tokens % 9
    boxes  = (rows // 3) * 3 + (cols // 3)

    same_row   = rows[:, None]   == rows[None, :]
    same_col   = cols[:, None]   == cols[None, :]
    same_box   = boxes[:, None]  == boxes[None, :]
    same_digit = digits[:, None] == digits[None, :]
    off_diag   = ~np.eye(729, dtype=bool)

    properties = [("Row", same_row), ("Col", same_col), ("Box", same_box), ("Digit", same_digit)]

    print()
    print(f"| {'Row'} | {'Col'} | {'Box'} | {'Digit'} | {'mean sim':>20} | {'n pairs':>10} |")
    print(f"|-----|-----|-----|-------|---------------------:|----------:|")
    for bits in range(16):
        mask = off_diag.copy()
        label_parts = []
        for bit, (_, same_mat) in enumerate(properties):
            if (bits >> bit) & 1:
                mask &= same_mat
                label_parts.append("same")
            else:
                mask &= ~same_mat
                label_parts.append("diff")
        n = mask.sum()
        mean_sim = f"{sim[mask].mean():.4f} ± {sim[mask].std():.4f}" if n > 0 else "—"
        print(f"| {label_parts[0]} | {label_parts[1]} | {label_parts[2]} | {label_parts[3]} | {mean_sim:>20} | {n:>10,} |")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default=DEFAULT_CKPT)
    parser.add_argument("--data", default=None, help="load precomputed similarity matrix (.npy)")
    args = parser.parse_args()

    if args.data:
        sim = np.load(args.data)
    else:
        sim = compute_data(args.ckpt)
        np.save(DATA_PATH, sim)
        print(f"Saved data to {DATA_PATH}")

    plot(sim)
    print_group_stats(sim)


if __name__ == "__main__":
    main()
