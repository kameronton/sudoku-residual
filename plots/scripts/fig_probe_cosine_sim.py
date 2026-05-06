"""Figure: cosine similarity of a cell's candidate probe weight vs. all 81 cells.

For a given digit, cell, and layer, fits candidate probes for all 81 cells,
extracts the weight vector for the specified digit from each, and plots the
pairwise cosine similarity as a 9x9 grid.

Also computes mean/std of cosine similarities across all digit-cell pairs,
grouped by spatial relationship between cells.

Usage:
    uv run python plots/scripts/fig_probe_cosine_sim.py --cell 40 --digit 5 --layer 4
    uv run python plots/scripts/fig_probe_cosine_sim.py --cell 0 --digit 1 --layer 3 --step 5
    uv run python plots/scripts/fig_probe_cosine_sim.py --cell 40 --digit 5 --layer 4 --cache results/3M-backtracking-packing/activations.npz
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sudoku.activations import load_probe_dataset, derive_n_clues
from sudoku.probes import (
    prepare_probe_inputs,
    get_activations_at_positions,
    CandidatesMode,
)

sns.set_theme(style="ticks", context="paper")
plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 10,
    "axes.labelsize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 9,
    "font.family": "serif",
    "pdf.fonttype": 42,
})

DEFAULT_CACHE = "results/3M-backtracking-packing/activations.npz"

GROUP_NAMES = {
    1: "row ∩ box",
    2: "col ∩ box",
    3: "row, ¬box",
    4: "col, ¬box",
    5: "box, ¬row/col",
    6: "stack, ¬col/box",
    7: "band, ¬row/box",
    8: "none",
}


def _build_groups() -> dict[int, list[tuple[int, int]]]:
    """Assign each unordered pair (c1, c2) to one of 8 mutually exclusive spatial groups."""
    groups: dict[int, list[tuple[int, int]]] = {i: [] for i in range(1, 9)}
    for c1 in range(81):
        r1, col1 = divmod(c1, 9)
        b1    = (r1 // 3) * 3 + (col1 // 3)
        band1 = r1 // 3
        stk1  = col1 // 3
        for c2 in range(c1 + 1, 81):
            r2, col2 = divmod(c2, 9)
            b2    = (r2 // 3) * 3 + (col2 // 3)
            band2 = r2 // 3
            stk2  = col2 // 3

            sr = r1 == r2
            sc = col1 == col2
            sb = b1 == b2

            if sr:
                g = 1 if sb else 3
            elif sc:
                g = 2 if sb else 4
            elif sb:
                g = 5
            elif stk1 == stk2:
                g = 6
            elif band1 == band2:
                g = 7
            else:
                g = 8
            groups[g].append((c1, c2))
    return groups


def extract_all_weights(acts: np.ndarray, grids: list[str]) -> np.ndarray:
    """Fit candidate probes for all 81 cells; return weight matrix (9, 81, d_model).

    Uses all available samples (no train/test split) since we only care about
    the direction of each probe vector, not its generalisation score.
    """
    mode = CandidatesMode()
    d_model = acts.shape[1]
    weights = np.full((9, 81, d_model), np.nan)

    for cell in range(81):
        targets, labels = mode.build_targets(grids, cell)
        _, X, y = mode.prepare_samples(acts, targets, labels)
        if len(X) < 4:
            continue
        clf = mode.fit(X, y)
        for d in range(9):
            weights[d, cell] = clf.estimators_[d].coef_[0]  # type: ignore[union-attr]

    return weights


def compute_group_stats(
    all_weights: np.ndarray,
) -> dict[int, tuple[float, float]]:
    """Mean and std of cosine similarities for each spatial group, across all digits.

    all_weights: (9, 81, d_model)
    Returns: {group_id: (mean, std)}
    """
    groups = _build_groups()
    group_sims: dict[int, list[float]] = {i: [] for i in range(1, 9)}

    for d in range(9):
        W = all_weights[d]                                      # (81, d_model)
        norms = np.linalg.norm(W, axis=1)                      # (81,)
        valid = ~np.any(np.isnan(W), axis=1) & (norms > 0)
        W_norm = np.where(valid[:, None], W / np.where(norms[:, None] > 0, norms[:, None], 1.0), np.nan)
        cos = W_norm @ W_norm.T                                 # (81, 81)

        for g, pairs in groups.items():
            for c1, c2 in pairs:
                v = cos[c1, c2]
                if not np.isnan(v):
                    group_sims[g].append(float(v))

    return {
        g: (float(np.mean(v)), float(np.std(v)))
        for g, v in group_sims.items()
        if v
    }


def print_group_stats(stats: dict[int, tuple[float, float]]) -> None:
    print(f"\n{'Group':<22}  {'Mean':>7}  {'Std':>7}  {'N':>7}")
    print("-" * 48)
    groups = _build_groups()
    for g in sorted(stats):
        mean, std = stats[g]
        n = len(groups[g]) * 9
        print(f"{GROUP_NAMES[g]:<22}  {mean:>7.4f}  {std:>7.4f}  {n:>7,}")


# Short column headers for the all-layers LaTeX table
GROUP_HEADERS = {
    1: r"r$\cap$box",
    2: r"c$\cap$box",
    3: r"r,$\lnot$box",
    4: r"c,$\lnot$box",
    5: r"box",
    6: r"stack",
    7: r"band",
    8: r"none",
}


def print_latex_table(all_layers_stats: dict[int, dict[int, tuple[float, float]]]) -> None:
    """Print a LaTeX table: rows = spatial groups, columns = layers (mean±std)."""
    layers = sorted(all_layers_stats)
    groups = sorted(next(iter(all_layers_stats.values())))
    col_spec = "l" + "".join(["X"] * len(layers))
    headers  = " & ".join(str(l + 1) for l in layers)

    print()
    print(r"\begin{tabularx}{\textwidth}{" + col_spec + r"}")
    print(r"\toprule")
    print(r"Group & " + headers + r" \\")
    print(r"\midrule")
    for g in groups:
        cells = []
        for layer in layers:
            mean, std = all_layers_stats[layer][g]
            cells.append(f"${mean:.3f} \\pm {std:.3f}$")
        print(GROUP_HEADERS[g] + " & " + " & ".join(cells) + r" \\")
    print(r"\bottomrule")
    print(r"\end{tabularx}")


def plot_group_stats(stats: dict[int, tuple[float, float]], output: str) -> None:
    ordered = sorted(stats)
    labels  = [GROUP_NAMES[i] for i in ordered]
    means   = np.array([stats[i][0] for i in ordered])
    stds    = np.array([stats[i][1] for i in ordered])

    fig, ax = plt.subplots(figsize=(4.5, 3.0))
    y = np.arange(len(labels))
    ax.barh(y, means, xerr=stds, align="center", height=0.6,
            error_kw={"elinewidth": 1, "capsize": 3})
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Cosine similarity (mean ± std across all digits)")
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    sns.despine(fig)
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    print(f"Saved {output}")


def plot(weights: np.ndarray, ref_cell: int, digit: int, layer: int, output: str) -> None:
    ref_w = weights[ref_cell]
    ref_norm = np.linalg.norm(ref_w)

    sims = np.full(81, np.nan)
    for c in range(81):
        w = weights[c]
        if np.any(np.isnan(w)):
            continue
        n = np.linalg.norm(w)
        if ref_norm == 0 or n == 0:
            continue
        sims[c] = float(np.dot(ref_w, w) / (ref_norm * n))

    grid = sims.reshape(9, 9)
    ref_row, ref_col = divmod(ref_cell, 9)

    fig, ax = plt.subplots(figsize=(4.0, 3.5))
    sns.heatmap(
        grid, ax=ax,
        vmin=-1, vmax=1, center=0,
        cmap="RdBu_r",
        annot=True, fmt=".2f",
        annot_kws={"size": 6},
        xticklabels=False,
        yticklabels=False,
        linewidths=0.5, linecolor="gray",
        square=True,
        cbar_kws={"label": "cosine similarity", "shrink": 0.8},
    )
    # Highlight the reference cell
    ax.add_patch(plt.Rectangle(
        (ref_col, ref_row), 1, 1,
        fill=False, edgecolor="black", lw=2, clip_on=False,
    ))
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")

    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    print(f"Saved {output}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", default=DEFAULT_CACHE)
    parser.add_argument("--cell", type=int, required=True, help="reference cell index 0-80")
    parser.add_argument("--digit", type=int, required=True, help="digit 1-9")
    parser.add_argument("--layer", type=int, required=True, help="layer index (0-based)")
    parser.add_argument("--step", type=int, default=0)
    parser.add_argument("--act-type", default="post_mlp")
    parser.add_argument("--output", default=None, help="output PDF path for heatmap (auto-generated if omitted)")
    args = parser.parse_args()

    if not (0 <= args.cell <= 80):
        parser.error("--cell must be 0-80")
    if not (1 <= args.digit <= 9):
        parser.error("--digit must be 1-9")

    activations, puzzles, sequences, n_clues, _ = load_probe_dataset(
        args.cache, act_type=args.act_type
    )
    if n_clues is None:
        n_clues = derive_n_clues(puzzles)

    activations, grids, positions, keep = prepare_probe_inputs(
        activations, puzzles, sequences, n_clues, args.step
    )

    n_layers = activations.shape[1]

    print(f"Extracting layer {args.layer} activations...")
    acts = get_activations_at_positions(activations, positions, args.layer, keep=keep)

    print("Fitting candidate probes for all digits across all 81 cells...")
    all_weights = extract_all_weights(acts, grids)  # (9, 81, d_model)

    # Heatmap for the requested digit/cell
    output = args.output or (
        f"plots/figures/fig_probe_cosine_sim_d{args.digit}_c{args.cell}_l{args.layer}.pdf"
    )
    plot(all_weights[args.digit - 1], args.cell, args.digit, args.layer, output)

    # Spatial group statistics across all layers
    print("Computing spatial group statistics across all layers...")
    all_layers_stats: dict[int, dict[int, tuple[float, float]]] = {}
    all_layers_stats[args.layer] = compute_group_stats(all_weights)

    for layer in range(n_layers):
        if layer == args.layer:
            continue
        print(f"  Layer {layer}...")
        acts_l = get_activations_at_positions(activations, positions, layer, keep=keep)
        weights_l = extract_all_weights(acts_l, grids)
        all_layers_stats[layer] = compute_group_stats(weights_l)

    print_latex_table(all_layers_stats)

    output_groups = f"plots/figures/fig_probe_cosine_sim_groups_l{args.layer}.pdf"
    plot_group_stats(all_layers_stats[args.layer], output_groups)


if __name__ == "__main__":
    main()
