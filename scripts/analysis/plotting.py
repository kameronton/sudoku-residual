"""Plotting helpers for Sudoku analysis notebooks."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from scripts.analysis.sudoku_state import cell_candidates_from_grid, token_label


def draw_board(ax, grid: str, show_candidates: bool = True) -> None:
    ax.set_xlim(0, 9)
    ax.set_ylim(0, 9)
    ax.set_aspect("equal")
    ax.axis("off")
    for cell in range(81):
        r, c = divmod(cell, 9)
        y = 8 - r
        ch = grid[cell]
        ax.add_patch(plt.Rectangle((c, y), 1, 1, color="#d8d8d8" if ch != "0" else "white", zorder=1))
        if ch != "0":
            ax.text(c + 0.5, y + 0.5, ch, ha="center", va="center", fontsize=18, fontweight="bold")
        elif show_candidates:
            cands = cell_candidates_from_grid(grid, cell)
            for d, is_cand in enumerate(cands):
                if not is_cand:
                    continue
                dc, dr = d % 3, d // 3
                ax.text(c + dc / 3 + 1 / 6, y + (2 - dr) / 3 + 1 / 6, str(d + 1),
                        ha="center", va="center", fontsize=6, color="#222", fontweight="bold")
    for i in range(10):
        lw = 2.0 if i % 3 == 0 else 0.5
        ax.axhline(i, color="black", linewidth=lw, zorder=3)
        ax.axvline(i, color="black", linewidth=lw, zorder=3)


def plot_top_tokens(ax, logits: np.ndarray, k: int = 10, title: str = "Top tokens") -> None:
    logits = np.asarray(logits)
    shifted = logits - logits.max()
    probs = np.exp(shifted) / np.exp(shifted).sum()
    top = np.argsort(probs)[::-1][:k]
    labels = [token_label(tok) for tok in top]
    values = probs[top]
    ax.barh(np.arange(k), values[::-1], color="steelblue")
    ax.set_yticks(np.arange(k))
    ax.set_yticklabels(labels[::-1], fontsize=8)
    ax.set_title(title)
    ax.set_xlabel("probability")
    ax.grid(axis="x", alpha=0.25)


def plot_metric_by_nempty(results: dict, title: str, output: str | None = None) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(title, fontsize=13)
    colors = {"simple": "steelblue", "hard": "tomato"}
    labels = {"simple": "Simple (no backtracking)", "hard": "Hard (backtracking)"}
    for group in ("simple", "hard"):
        xs = results[group]["n_empty"]
        axes[0].plot(xs, results[group]["auc"], marker="o", color=colors[group], label=labels[group])
        axes[1].plot(xs, results[group]["brier"], marker="o", color=colors[group], label=labels[group])
    axes[0].set_title("AUC")
    axes[1].set_title("Brier")
    for ax in axes:
        ax.set_xlabel("n_empty")
        ax.grid(alpha=0.3)
        ax.legend()
    axes[0].set_ylabel("AUC")
    axes[1].set_ylabel("Brier score")
    plt.tight_layout()
    if output:
        plt.savefig(output, dpi=160, bbox_inches="tight")
    plt.show()


def plot_heatmap(matrix: np.ndarray, title: str, xlabel: str, ylabel: str, output: str | None = None) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(matrix, aspect="auto", cmap="viridis")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.colorbar(im, ax=ax)
    plt.tight_layout()
    if output:
        plt.savefig(output, dpi=160, bbox_inches="tight")
    plt.show()

