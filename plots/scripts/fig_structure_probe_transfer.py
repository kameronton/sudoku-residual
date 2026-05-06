"""Figure: substructure candidate probe trained at SEP, evaluated at later n_empty levels.

Probes predict which digits are *not yet placed* (candidates) in each row/col/box.
Two conditions at each n_empty level:
  - full:     all test substructures
  - filtered: exclude substructures that are already fully solved (no candidates left)

Metrics: AUC, Brier (MSE), element-wise accuracy (threshold 0.5), exact match accuracy.

Usage:
    uv run python plots/scripts/fig_structure_probe_transfer.py
    uv run python plots/scripts/fig_structure_probe_transfer.py --cache results/3M-lr1e3/activations.npz
    uv run python plots/scripts/fig_structure_probe_transfer.py --data plots/data/fig_structure_probe_transfer.csv
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sudoku.probes.session import ProbeSession
from sudoku.probes.modes import STRUCTURE

sns.set_theme(style="ticks", context="paper")

plt.rcParams.update({
    'font.size': 10,
    'axes.labelsize': 10,
    'axes.titlesize': 10,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 7,
    'font.family': 'serif',
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
})

DEFAULT_CACHE = "results/3M-backtracking-packing/activations.npz"
OUTPUT_AUC = "plots/figures/fig_structure_probe_transfer_auc.pdf"
OUTPUT_MSE = "plots/figures/fig_structure_probe_transfer_mse.pdf"
OUTPUT_EW  = "plots/figures/fig_structure_probe_transfer_ew.pdf"
OUTPUT_EM  = "plots/figures/fig_structure_probe_transfer_em.pdf"
DATA_PATH  = "plots/data/fig_structure_probe_transfer.csv"

SUBTYPES = [("row", i) for i in range(9)] + [("col", i) for i in range(9)] + [("box", i) for i in range(9)]


def build_candidate_targets(grids: list[str], subtype: str, sidx: int) -> np.ndarray:
    """(N, 9) — targets[i, d] = 1 if digit d+1 is still a candidate (not yet placed)."""
    n = len(grids)
    targets = np.ones((n, 9), dtype=np.float32)
    for i, g in enumerate(grids):
        if subtype == "row":
            cells = g[sidx * 9:(sidx + 1) * 9]
        elif subtype == "col":
            cells = [g[r * 9 + sidx] for r in range(9)]
        else:
            br, bc = (sidx // 3) * 3, (sidx % 3) * 3
            cells = [g[(br + dr) * 9 + (bc + dc)] for dr in range(3) for dc in range(3)]
        for ch in cells:
            if ch in "123456789":
                targets[i, int(ch) - 1] = 0.0
    return targets


def _accs_multilabel(clf, X_test: np.ndarray, y_test: np.ndarray) -> tuple[float, float]:
    """Returns (element-wise accuracy, exact match accuracy) for a multi-label probe."""
    proba_list = clf.predict_proba(X_test)
    probas = np.column_stack([p[:, 1] for p in proba_list])  # (n, 9)
    preds = (probas > 0.5).astype(int)
    elementwise = float(np.mean(preds == y_test))
    exactmatch  = float(np.mean(np.all(preds == y_test, axis=1)))
    return elementwise, exactmatch


def compute_data(args) -> pd.DataFrame:
    session = ProbeSession(args.cache, act_type=args.act_type)

    idx0 = session.index.at_step(0).first_per_puzzle()
    train_mask, test_mask = session.split(idx0)
    train_idx = idx0[train_mask]
    test_idx  = idx0[test_mask]
    test_puzzle_set = set(test_idx.puzzle_idx.tolist())

    n_empty_max = int((81 - session.index.n_filled[session.index.step == 0]).max())
    n_empty_values = list(range(n_empty_max, -1, -1))
    print(f"n_empty sweep: {n_empty_max} → 0  ({len(n_empty_values)} levels)")

    rows = []

    for layer in range(session.n_layers):
        print(f"\nLayer {layer}/{session.n_layers - 1} — training probes...")

        grids_train = session.grids(train_idx)
        X_train = session.acts(train_idx, layer=layer)

        clfs = {}
        for subtype, sidx in SUBTYPES:
            y_tr = build_candidate_targets(grids_train, subtype, sidx)
            clfs[(subtype, sidx)] = STRUCTURE.fit(X_train, y_tr)

        print(f"Layer {layer} — evaluating across {len(n_empty_values)} n_empty levels...")

        for n_empty in n_empty_values:
            idx_slot = (session.index
                        .where_filled(81 - n_empty)
                        .first_per_puzzle())
            in_test = np.isin(idx_slot.puzzle_idx, list(test_puzzle_set))
            idx_slot = idx_slot[in_test]

            if len(idx_slot) < 4:
                continue

            X_slot = session.acts(idx_slot, layer=layer)
            grids_slot = session.grids(idx_slot)

            for subtype, sidx in SUBTYPES:
                targets = build_candidate_targets(grids_slot, subtype, sidx)
                clf = clfs[(subtype, sidx)]

                auc_full,  brier_full  = STRUCTURE.evaluate(clf, X_slot, targets)
                ew_full,   em_full     = _accs_multilabel(clf, X_slot, targets)

                has_cands = targets.sum(axis=1) > 0
                if has_cands.sum() >= 4:
                    X_filt, t_filt = X_slot[has_cands], targets[has_cands]
                    auc_filt,  brier_filt  = STRUCTURE.evaluate(clf, X_filt, t_filt)
                    ew_filt,   em_filt     = _accs_multilabel(clf, X_filt, t_filt)
                else:
                    auc_filt = brier_filt = ew_filt = em_filt = float("nan")

                rows.append({
                    "n_empty": n_empty,
                    "layer": layer + 1,
                    "subtype": subtype,
                    "sidx": sidx,
                    "full_auc": auc_full,
                    "filtered_auc": auc_filt,
                    "full_brier": brier_full,
                    "filtered_brier": brier_filt,
                    "full_ew": ew_full,
                    "filtered_ew": ew_filt,
                    "full_em": em_full,
                    "filtered_em": em_filt,
                    "n_puzzles": int(has_cands.sum()),
                })

    return pd.DataFrame(rows)


def _plot_metric(
    df: pd.DataFrame,
    full_col: str,
    filtered_col: str,
    ylabel: str,
    output: str,
    ylim: tuple | None = None,
):
    value_cols = [c for c in [full_col, filtered_col] if c in df.columns]
    label_map = {full_col: "full", filtered_col: "filtered"}

    agg = (df.groupby(["n_empty", "layer"])[value_cols]
             .mean()
             .reset_index())
    agg["Layer"] = agg["layer"].map(lambda x: f"Layer {x}")

    long = agg.melt(id_vars=["n_empty", "layer", "Layer"], value_vars=value_cols,
                    var_name="condition", value_name=ylabel)
    long["condition"] = long["condition"].map(label_map)

    palette = sns.color_palette("tab10", n_colors=agg["layer"].nunique())
    multi_condition = len(value_cols) > 1

    fig, ax = plt.subplots(figsize=(4, 2.2))
    if multi_condition:
        sns.lineplot(data=long, x="n_empty", y=ylabel, hue="Layer", style="condition",
                     palette=palette, dashes={"full": (2, 2), "filtered": (1, 0)}, ax=ax)
    else:
        sns.lineplot(data=long, x="n_empty", y=ylabel, hue="Layer", palette=palette, ax=ax)
    ax.set_xlabel("Empty cells remaining")
    ax.set_ylabel(ylabel)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.invert_xaxis()
    ax.legend(frameon=False)
    sns.despine(fig)
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    print(f"Saved {output}")


def plot(df: pd.DataFrame):
    _plot_metric(df, "full_auc",   "filtered_auc",   "AUC",             OUTPUT_AUC, ylim=(0.85, 1.02))
    _plot_metric(df, "full_brier", "filtered_brier",  "MSE",             OUTPUT_MSE, ylim=(-0.005, None))
    _plot_metric(df, "full_ew",    "filtered_ew",     "Element-wise Acc", OUTPUT_EW)
    _plot_metric(df, "full_em",    "filtered_em",     "Mean exact match acc",  OUTPUT_EM)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", default=DEFAULT_CACHE)
    parser.add_argument("--act-type", default="post_mlp")
    parser.add_argument("--data", default=None, help="load precomputed CSV instead of running probes")
    parser.add_argument("--layers", nargs="+", type=int, default=None, help="layers to plot, e.g. --layers 3 4 5")
    parser.add_argument("--no-filtered", action="store_true", help="omit filtered condition from plot")
    args = parser.parse_args()

    if args.data:
        df = pd.read_csv(args.data)
    else:
        df = compute_data(args)
        df.to_csv(DATA_PATH, index=False)
        print(f"Saved data to {DATA_PATH}")

    if args.layers:
        df = df[df["layer"].isin(args.layers)]
    if args.no_filtered:
        df = df.drop(columns=["filtered_auc", "filtered_brier", "filtered_ew", "filtered_em"], errors="ignore")

    plot(df)


if __name__ == "__main__":
    main()
