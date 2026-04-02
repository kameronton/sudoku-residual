"""Visualization for probe results."""

import numpy as np


def plot_all_layers(
    all_accuracies: dict[int, list[float]],
    output_path: str = "probe_accuracies.png", metric_name: str = "Accuracy",
    show: bool = True, vmin: float = 0.0, vmax: float = 1.0, cmap: str = "RdYlGn",
):
    """Plot 9x9 heatmap per layer with shared colorbar."""
    import matplotlib.pyplot as plt

    n_layers = len(all_accuracies)
    if n_layers == 8:
        ncols = 4
        nrows = 2
    else:
        ncols = min(n_layers, 3)
        nrows = (n_layers + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols + 0.5, 3 * nrows))
    if n_layers == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    ims = []
    for idx, (layer, accs) in enumerate(sorted(all_accuracies.items())):
        ax = axes[idx]
        grid = np.array(accs).reshape(9, 9)
        im = ax.imshow(grid, cmap=cmap, vmin=vmin, vmax=vmax)
        ims.append(im)
        avg = np.mean(accs)
        ax.set_title(f"Layer {layer} (mean={avg:.3f})")
        ax.set_xticks(range(9))
        ax.set_yticks(range(9))
        for r in range(9):
            for c in range(9):
                ax.text(c, r, f"{grid[r, c]:.2f}", ha="center", va="center", fontsize=6)
        for i in range(0, 10, 3):
            ax.axhline(i - 0.5, color="black", linewidth=2)
            ax.axvline(i - 0.5, color="black", linewidth=2)

    # Hide unused axes
    for idx in range(n_layers, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle(f"Per-cell probe {metric_name.lower()} by layer", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.colorbar(ims[0], ax=axes[:n_layers].tolist(), shrink=0.6, label=metric_name, pad=0.02)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved {output_path}")
    if show:
        plt.show()
    plt.close(fig)


def plot_all_layers_per_digit(
    all_per_digit: dict[int, np.ndarray],
    output_path: str = "probe_per_digit.png",
    show: bool = True,
):
    """Plot 9x9 grid per layer where each cell contains a 3x3 mini-heatmap of per-digit F1."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    n_layers = len(all_per_digit)
    if n_layers == 8:
        ncols = 4
        nrows = 2
    else:
        ncols = min(n_layers, 3)
        nrows = (n_layers + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows),
                             gridspec_kw={"wspace": 0.1, "hspace": 0.1})
    if n_layers == 1:
        axes = np.array([axes])
    axes = np.atleast_1d(axes).flatten()

    norm = Normalize(vmin=0, vmax=1)
    cmap = plt.get_cmap("RdYlGn")

    for idx, (layer, per_digit_arr) in enumerate(sorted(all_per_digit.items())):
        ax = axes[idx]
        ax.set_xlim(-0.5, 8.5)
        ax.set_ylim(8.5, -0.5)
        ax.set_aspect("equal")
        ax.set_xticks(range(9))
        ax.set_yticks(range(9))
        ax.tick_params(length=0)

        avg = np.nanmean(per_digit_arr)
        ax.set_title(f"Layer {layer} (mean F1={avg:.3f})")

        # Draw sudoku box lines
        for i in range(0, 10, 3):
            ax.axhline(i - 0.5, color="black", linewidth=2)
            ax.axvline(i - 0.5, color="black", linewidth=2)
        # Thin cell lines
        for i in range(10):
            ax.axhline(i - 0.5, color="gray", linewidth=0.5)
            ax.axvline(i - 0.5, color="gray", linewidth=0.5)

        for cell in range(81):
            r, c = divmod(cell, 9)
            digits = per_digit_arr[cell]  # shape (9,)
            mini = digits.reshape(3, 3)
            # Inset axes: map cell (c, r) to figure coords
            inset = ax.inset_axes(
                [c - 0.5, r - 0.5, 1, 1],
                transform=ax.transData,
            )
            inset.imshow(mini, cmap=cmap, norm=norm, aspect="equal")
            inset.set_xticks([])
            inset.set_yticks([])
            inset.patch.set_alpha(0)
            # Cell boundary
            for spine in inset.spines.values():
                spine.set_edgecolor("gray")
                spine.set_linewidth(0.5)
            # Intra-cell grid lines between the 3x3 digits
            for pos in [0.5, 1.5]:
                inset.axhline(pos, color="gray", linewidth=0.3, alpha=0.6)
                inset.axvline(pos, color="gray", linewidth=0.3, alpha=0.6)

    for idx in range(n_layers, len(axes)):
        axes[idx].set_visible(False)

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    fig.colorbar(sm, ax=axes[:n_layers].tolist(), shrink=0.6, label="F1", pad=0.02)
    fig.suptitle("Per-digit candidate F1 by layer", fontsize=14, y=1.02)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved {output_path}")
    if show:
        plt.show()
    plt.close(fig)


def plot_structure(
    all_scores: dict[int, dict[str, list[float]]],
    output_path: str = "probe_structure.png",
    show: bool = True, vmin: float = 0.0, vmax: float = 1.0, cmap: str = "RdYlGn",
):
    """Plot structure probe F1: n_layers rows x 3 cols (row/col/box substructures).

    Row/col subtypes are shown as a 1x9 horizontal heatmap strip.
    Box subtype is shown as a 3x3 heatmap matching the Sudoku grid layout.
    """
    import matplotlib.pyplot as plt

    n_layers = len(all_scores)
    subtypes = ["row", "col", "box"]
    col_titles = ["Rows", "Columns", "Boxes"]
    fig, axes = plt.subplots(n_layers, 3, figsize=(6.5, 1.5 * n_layers), constrained_layout=True)
    if n_layers == 1:
        axes = axes[np.newaxis, :]

    ims = []
    for layer_idx, (layer, scores) in enumerate(sorted(all_scores.items())):
        for col_idx, subtype in enumerate(subtypes):
            ax = axes[layer_idx, col_idx]
            vals = np.array(scores[subtype])
            if subtype == "box":
                data = vals.reshape(3, 3)
            elif subtype == "row":
                data = vals.reshape(9, 1)
            else:  # col
                data = vals.reshape(1, 9)
            im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
            if layer_idx == 0:
                ims.append(im)
            for r in range(data.shape[0]):
                for c in range(data.shape[1]):
                    ax.text(c, r, f"{data[r, c]:.2f}", ha="center", va="center", fontsize=5)
            ax.set_xticks([])
            ax.set_yticks([])
            if layer_idx == 0:
                ax.set_title(col_titles[col_idx], fontsize=9)
            if col_idx == 0:
                ax.set_ylabel(f"L{layer}", fontsize=8, rotation=0, labelpad=20, va="center")

    fig.colorbar(ims[0], ax=axes.ravel().tolist(), shrink=0.5, pad=0.02)
    fig.suptitle("Structure probe (row / col / box)", fontsize=11)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved {output_path}")
    if show:
        plt.show()
    plt.close(fig)


def plot_cell_temporal(
    filtered_results: dict[int, dict[str, list[float]]],
    full_results: dict[int, dict[str, list[float]]],
    cell_idx: int,
    output_path: str,
    show: bool = True,
):
    """Line charts of candidate AUC and Brier vs step for a single cell.

    Two line styles per layer: solid = filtered probe (cell empty at step),
    dashed = full probe (all puzzles, filled cells get zero-candidate targets).
    """
    import matplotlib.pyplot as plt

    n_layers = len(filtered_results)
    steps = list(range(len(next(iter(filtered_results.values()))["auc"])))
    row, col = divmod(cell_idx, 9)

    fig, (ax_auc, ax_brier) = plt.subplots(1, 2, figsize=(12, 4))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for layer in range(n_layers):
        color = colors[layer % len(colors)]
        label = f"L{layer}"

        filt_auc = filtered_results[layer]["auc"]
        filt_brier = filtered_results[layer]["brier"]
        full_auc = full_results[layer]["auc"]
        full_brier = full_results[layer]["brier"]

        ax_auc.plot(steps, filt_auc, color=color, linestyle="-", marker="o", markersize=3, label=f"{label} filtered")
        ax_auc.plot(steps, full_auc, color=color, linestyle="--", marker="s", markersize=3, label=f"{label} full")
        ax_brier.plot(steps, filt_brier, color=color, linestyle="-", marker="o", markersize=3, label=f"{label} filtered")
        ax_brier.plot(steps, full_brier, color=color, linestyle="--", marker="s", markersize=3, label=f"{label} full")

    for ax in (ax_auc, ax_brier):
        ax.set_xlabel("Step")
        ax.legend(fontsize=7, ncol=2)

    ax_auc.set_ylabel("AUC")
    ax_auc.set_ylim(0, 1)
    ax_auc.set_title(f"Candidate AUC — cell ({row},{col})")
    ax_brier.set_ylabel("Brier")
    ax_brier.set_ylim(0, 0.25)
    ax_brier.set_title(f"Candidate Brier — cell ({row},{col})")

    fig.suptitle(
        f"Cell ({row},{col}) candidate probes  |  solid = filtered (empty at step), dashed = full",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved {output_path}")
    if show:
        plt.show()
    plt.close(fig)


def plot_cross_step(
    auc_by_step: dict[int, dict[int, float]],
    brier_by_step: dict[int, dict[int, float]],
    train_step: int,
    output_path: str,
    show: bool = True,
):
    """Line chart of AUC and Brier vs eval step, one line per layer."""
    import matplotlib.pyplot as plt

    steps = sorted(auc_by_step.keys())
    n_layers = len(next(iter(auc_by_step.values())))

    fig, (ax_auc, ax_brier) = plt.subplots(1, 2, figsize=(11, 4))

    for layer in range(n_layers):
        aucs = [auc_by_step[s][layer] for s in steps]
        briers = [brier_by_step[s][layer] for s in steps]
        ax_auc.plot(steps, aucs, marker="o", markersize=3, label=f"L{layer}")
        ax_brier.plot(steps, briers, marker="o", markersize=3, label=f"L{layer}")

    for ax in (ax_auc, ax_brier):
        ax.axvline(train_step, color="black", linestyle="--", linewidth=1, alpha=0.5, label="train step")
        ax.set_xlabel("Eval step")
        ax.legend(fontsize=8)

    ax_auc.set_ylabel("AUC")
    ax_auc.set_ylim(0, 1)
    ax_auc.set_title(f"AUC vs eval step (trained at step {train_step})")
    ax_brier.set_ylabel("Brier")
    ax_brier.set_ylim(0, 0.25)
    ax_brier.set_title(f"Brier vs eval step (trained at step {train_step})")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved {output_path}")
    if show:
        plt.show()
    plt.close(fig)
