"""Figure: accuracy of state_filled, candidates, and substructure probes by layer.

Metrics:
  element-wise — average per-digit binary accuracy (threshold 0.5).
                 For state_filled: fraction of cells where predicted digit == true digit.
  exact match  — fraction of cells/substructures where all 9 binary predictions are correct.
                 For state_filled: same as element-wise (single-label, one prediction per sample).

Usage:
    uv run python plots/scripts/fig_probe_acc_by_layer.py
    uv run python plots/scripts/fig_probe_acc_by_layer.py --cache results/3M-lr1e3/activations.npz
    uv run python plots/scripts/fig_probe_acc_by_layer.py --data plots/data/fig_probe_acc_by_layer.csv
    uv run python plots/scripts/fig_probe_acc_by_layer.py --step 5 --act-type post_attn
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

from sudoku.activations import load_probe_dataset, derive_n_clues
from sudoku.probes import prepare_probe_inputs, MODES, STRUCTURE
from sudoku.probes.activations import get_activations_at_positions

sns.set_theme(style="ticks", context="paper")

plt.rcParams.update({
    'font.size': 10,
    'axes.titlesize': 10,
    'axes.labelsize': 10,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 9,
    'font.family': 'serif',
    'pdf.fonttype': 42,
})

DEFAULT_CACHE = "results/3M-backtracking-packing/activations.npz"
OUTPUT_EW = "plots/figures/fig_probe_acc_by_layer_elementwise.pdf"
OUTPUT_EM = "plots/figures/fig_probe_acc_by_layer_exactmatch.pdf"
DATA_PATH = "plots/data/fig_probe_acc_by_layer.csv"


def _accs_multiclass(clf, X_test, y_test) -> tuple[float, float]:
    """Element-wise and exact-match accuracy for single-label multiclass (identical)."""
    acc = float(np.mean(clf.predict(X_test) == y_test))
    return acc, acc


def _accs_multilabel(clf, X_test, y_test) -> tuple[float, float]:
    """Element-wise and exact-match accuracy for multi-label binary predictions."""
    proba_list = clf.predict_proba(X_test)
    probas = np.column_stack([p[:, 1] for p in proba_list])  # (n, 9)
    preds = (probas > 0.5).astype(int)
    elementwise = float(np.mean(preds == y_test))
    exactmatch = float(np.mean(np.all(preds == y_test, axis=1)))
    return elementwise, exactmatch


def _probe_cell_accs(
    acts: np.ndarray, grids: list[str], cell: int, mode_name: str
) -> tuple[float, float]:
    mode = MODES[mode_name]
    targets, labels = mode.build_targets(grids, cell)
    rel_idx, X, y = mode.prepare_samples(acts, targets, labels)
    if len(X) < 4:
        return float("nan"), float("nan")
    idx_train, idx_test = train_test_split(np.arange(len(X)), test_size=0.2, random_state=42)
    clf = mode.fit(X[idx_train], y[idx_train])
    if mode_name == "state_filled":
        return _accs_multiclass(clf, X[idx_test], y[idx_test])
    return _accs_multilabel(clf, X[idx_test], y[idx_test])


def _probe_structure_accs(
    acts: np.ndarray, grids: list[str], subtype: str, idx: int
) -> tuple[float, float]:
    targets = STRUCTURE.build_targets(grids, subtype, idx)
    idx_train, idx_test = train_test_split(np.arange(len(grids)), test_size=0.2, random_state=42)
    clf = STRUCTURE.fit(acts[idx_train], targets[idx_train])
    return _accs_multilabel(clf, acts[idx_test], targets[idx_test])


def compute_data(args) -> pd.DataFrame:
    activations, puzzles, sequences, n_clues, _ = load_probe_dataset(
        args.cache, act_type=args.act_type
    )
    if n_clues is None:
        n_clues = derive_n_clues(puzzles)

    activations, probe_grids, probe_positions, keep = prepare_probe_inputs(
        activations, puzzles, sequences, n_clues, args.step
    )

    n_layers = activations.shape[1]
    rows = []

    for layer in range(n_layers):
        print(f"Layer {layer}...")
        acts = get_activations_at_positions(activations, probe_positions, layer, keep=keep)

        state_ew, state_em = zip(*[_probe_cell_accs(acts, probe_grids, c, "state_filled") for c in range(81)])
        cand_ew,  cand_em  = zip(*[_probe_cell_accs(acts, probe_grids, c, "candidates")   for c in range(81)])
        struct_pairs = [
            _probe_structure_accs(acts, probe_grids, subtype, idx)
            for subtype in ("row", "col", "box")
            for idx in range(9)
        ]
        struct_ew, struct_em = zip(*struct_pairs)

        print(f"  element-wise  state={np.nanmean(state_ew):.3f}  cand={np.nanmean(cand_ew):.3f}"
              f"  struct={np.nanmean(struct_ew):.3f}")
        print(f"  exact match   state={np.nanmean(state_em):.3f}  cand={np.nanmean(cand_em):.3f}"
              f"  struct={np.nanmean(struct_em):.3f}")

        rows += [
            {"Layer": "L" + str(layer + 1), "Metric": "element-wise", "Accuracy": np.nanmean(state_ew),  "Probe": "cell state"},
            {"Layer": "L" + str(layer + 1), "Metric": "element-wise", "Accuracy": np.nanmean(cand_ew),   "Probe": "cell candidates"},
            {"Layer": "L" + str(layer + 1), "Metric": "element-wise", "Accuracy": np.nanmean(struct_ew), "Probe": "substructure state"},
            {"Layer": "L" + str(layer + 1), "Metric": "exact match",  "Accuracy": np.nanmean(state_em),  "Probe": "cell state"},
            {"Layer": "L" + str(layer + 1), "Metric": "exact match",  "Accuracy": np.nanmean(cand_em),   "Probe": "cell candidates"},
            {"Layer": "L" + str(layer + 1), "Metric": "exact match",  "Accuracy": np.nanmean(struct_em), "Probe": "substructure state"},
        ]

    return pd.DataFrame(rows)


def _plot_metric(df: pd.DataFrame, metric: str, output: str):
    sub = df[df["Metric"] == metric]
    layer_labels = sorted(sub["Layer"].unique())
    fig, ax = plt.subplots(figsize=(4, 2.2))
    sns.lineplot(data=sub, x="Layer", y="Accuracy", hue="Probe", style="Probe",
                 markers=["o", "s", "^"], dashes=False, ax=ax)
    ax.set_xticks(layer_labels)
    # ax.set_title(metric)
    ax.set_ylabel("Mean exact match acc")
    ax.legend(frameon=False)
    sns.despine(fig)
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    print(f"Saved {output}")


def plot(df: pd.DataFrame):
    _plot_metric(df, "element-wise", OUTPUT_EW)
    _plot_metric(df, "exact match",  OUTPUT_EM)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", default=DEFAULT_CACHE)
    parser.add_argument("--step", type=int, default=0)
    parser.add_argument("--act-type", default="post_mlp")
    parser.add_argument("--data", default=None, help="load precomputed CSV instead of running probes")
    args = parser.parse_args()

    if args.data:
        df = pd.read_csv(args.data)
    else:
        df = compute_data(args)
        df.to_csv(DATA_PATH, index=False)
        print(f"Saved data to {DATA_PATH}")

    plot(df)


if __name__ == "__main__":
    main()
