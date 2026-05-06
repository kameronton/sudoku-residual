"""Figure: substructure candidate probe transfer across layers.

Train row/col/box candidate probes on activations from layer L, then evaluate
the same probes on activations from layer L'. This checks whether the linear
readout basis is shared across residual-stream layers.

Usage:
    uv run python plots/scripts/fig_structure_probe_layer_transfer.py
    uv run python plots/scripts/fig_structure_probe_layer_transfer.py --cache results/3M-lr1e3/activations.npz
    uv run python plots/scripts/fig_structure_probe_layer_transfer.py --step 5 --act-type post_attn
    uv run python plots/scripts/fig_structure_probe_layer_transfer.py --data plots/data/fig_structure_probe_layer_transfer.csv
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sudoku.probes.modes import STRUCTURE
from sudoku.probes.session import ProbeSession

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
DATA_PATH = "plots/data/fig_structure_probe_layer_transfer.csv"
OUTPUT_AUC = "plots/figures/fig_structure_probe_layer_transfer_auc.pdf"
OUTPUT_MSE = "plots/figures/fig_structure_probe_layer_transfer_mse.pdf"
OUTPUT_EW = "plots/figures/fig_structure_probe_layer_transfer_ew.pdf"
OUTPUT_EM = "plots/figures/fig_structure_probe_layer_transfer_em.pdf"

SUBTYPES = [("row", i) for i in range(9)] + [("col", i) for i in range(9)] + [("box", i) for i in range(9)]


def build_candidate_targets(grids: list[str], subtype: str, sidx: int) -> np.ndarray:
    """(N, 9): targets[i, d] = 1 if digit d+1 is not yet placed."""
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
    """Returns (element-wise accuracy, exact match accuracy)."""
    proba_list = clf.predict_proba(X_test)
    probas = np.column_stack([p[:, 1] for p in proba_list])
    preds = (probas > 0.5).astype(int)
    elementwise = float(np.mean(preds == y_test))
    exactmatch = float(np.mean(np.all(preds == y_test, axis=1)))
    return elementwise, exactmatch


def _select_layers(all_layers: int, requested: list[int] | None) -> list[int]:
    """Return 0-indexed layers; CLI layer numbers are 1-indexed for plot consistency."""
    if requested is None:
        return list(range(all_layers))
    layers = [l - 1 for l in requested]
    bad = [l + 1 for l in layers if l < 0 or l >= all_layers]
    if bad:
        raise ValueError(f"Layer(s) out of range: {bad}; expected 1..{all_layers}")
    return layers


def compute_data(args) -> pd.DataFrame:
    session = ProbeSession(args.cache, act_type=args.act_type)
    train_layers = _select_layers(session.n_layers, args.train_layers)
    eval_layers = _select_layers(session.n_layers, args.eval_layers)

    idx = session.index.at_step(args.step).first_per_puzzle()
    if len(idx) == 0:
        raise ValueError(f"No samples found at step {args.step}")

    train_mask, test_mask = session.split(idx)
    train_idx = idx[train_mask]
    test_idx = idx[test_mask]
    grids_train = session.grids(train_idx)
    grids_test = session.grids(test_idx)

    targets_train = {
        key: build_candidate_targets(grids_train, *key)
        for key in SUBTYPES
    }
    targets_test = {
        key: build_candidate_targets(grids_test, *key)
        for key in SUBTYPES
    }
    X_test_by_layer = {
        layer: session.acts(test_idx, layer=layer)
        for layer in eval_layers
    }

    rows = []
    for train_layer in train_layers:
        print(f"\nTraining probes on layer {train_layer + 1}/{session.n_layers}...")
        X_train = session.acts(train_idx, layer=train_layer)
        clfs = {
            key: STRUCTURE.fit(X_train, targets_train[key])
            for key in SUBTYPES
        }

        print(f"Evaluating layer {train_layer + 1} probes on {len(eval_layers)} eval layers...")
        for eval_layer in eval_layers:
            X_test = X_test_by_layer[eval_layer]
            for subtype, sidx in SUBTYPES:
                key = (subtype, sidx)
                targets = targets_test[key]
                auc, brier = STRUCTURE.evaluate(clfs[key], X_test, targets)
                ew, em = _accs_multilabel(clfs[key], X_test, targets)
                rows.append({
                    "step": args.step,
                    "train_layer": train_layer + 1,
                    "eval_layer": eval_layer + 1,
                    "subtype": subtype,
                    "sidx": sidx,
                    "auc": auc,
                    "brier": brier,
                    "elementwise_acc": ew,
                    "exactmatch_acc": em,
                    "n_train": len(train_idx),
                    "n_test": len(test_idx),
                })

    return pd.DataFrame(rows)


def _plot_heatmap(df: pd.DataFrame, value_col: str, label: str, output: str, cmap: str, vmin=None, vmax=None):
    mat = df.pivot_table(
        index="train_layer",
        columns="eval_layer",
        values=value_col,
        aggfunc="mean",
    )
    fig, ax = plt.subplots(figsize=(3.0, 2.6))
    sns.heatmap(
        mat,
        ax=ax,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        square=True,
        cbar_kws={"label": label},
    )
    ax.set_xlabel("Eval layer")
    ax.set_ylabel("Train layer")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    print(f"Saved {output}")


def plot(df: pd.DataFrame):
    _plot_heatmap(df, "auc", "AUC", OUTPUT_AUC, cmap="viridis", vmin=0.5, vmax=1.0)
    _plot_heatmap(df, "brier", "MSE", OUTPUT_MSE, cmap="mako_r", vmin=0.0, vmax=None)
    _plot_heatmap(df, "elementwise_acc", "Element-wise acc", OUTPUT_EW, cmap="viridis", vmin=0.5, vmax=1.0)
    _plot_heatmap(df, "exactmatch_acc", "Exact match acc", OUTPUT_EM, cmap="viridis", vmin=0.0, vmax=1.0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", default=DEFAULT_CACHE)
    parser.add_argument("--step", type=int, default=0)
    parser.add_argument("--act-type", default="post_mlp")
    parser.add_argument("--data", default=None, help="load precomputed CSV instead of running probes")
    parser.add_argument("--train-layers", nargs="+", type=int, default=None, help="1-indexed layers to train on")
    parser.add_argument("--eval-layers", nargs="+", type=int, default=None, help="1-indexed layers to evaluate on")
    args = parser.parse_args()

    if args.data:
        df = pd.read_csv(args.data)
    else:
        df = compute_data(args)
        df.to_csv(DATA_PATH, index=False)
        print(f"Saved data to {DATA_PATH}")

    if args.train_layers:
        df = df[df["train_layer"].isin(args.train_layers)]
    if args.eval_layers:
        df = df[df["eval_layer"].isin(args.eval_layers)]

    plot(df)


if __name__ == "__main__":
    main()
