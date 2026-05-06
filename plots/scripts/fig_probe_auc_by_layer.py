"""Figure: mean AUC of candidates (cell-level) vs. structure (substructure) probes by layer.

Usage:
    uv run python plots/scripts/fig_probe_auc_by_layer.py                         # run probes + save data + plot
    uv run python plots/scripts/fig_probe_auc_by_layer.py --data plots/data/fig_probe_auc_by_layer.csv  # plot only
    uv run python plots/scripts/fig_probe_auc_by_layer.py --cache results/3M-lr1e3/activations.npz
    uv run python plots/scripts/fig_probe_auc_by_layer.py --step 5 --act-type post_attn
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sudoku.activations import load_probe_dataset, derive_n_clues
from sudoku.probes import prepare_probe_inputs, run_probe_loop, run_structure_probe_loop

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
OUTPUT_AUC = "plots/figures/fig_probe_auc_by_layer_auc.pdf"
OUTPUT_MSE = "plots/figures/fig_probe_auc_by_layer_mse.pdf"
DATA_PATH = "plots/data/fig_probe_auc_by_layer.csv"


def compute_data(args) -> pd.DataFrame:
    activations, puzzles, sequences, n_clues, _ = load_probe_dataset(
        args.cache, act_type=args.act_type
    )
    if n_clues is None:
        n_clues = derive_n_clues(puzzles)

    activations, probe_grids, probe_positions, keep = prepare_probe_inputs(
        activations, puzzles, sequences, n_clues, args.step
    )

    print("Running cell state probes...")
    state_auc, _, state_brier = run_probe_loop(
        activations, probe_grids, probe_positions, mode="state_filled", keep=keep
    )

    print("Running cell candidates probes...")
    cand_auc, _, cand_brier = run_probe_loop(
        activations, probe_grids, probe_positions, mode="candidates", keep=keep
    )

    print("Running structure probes...")
    struct_scores, struct_brier = run_structure_probe_loop(
        activations, probe_grids, probe_positions, keep=keep
    )

    layers = sorted(state_auc)
    rows = []
    for l in layers:
        struct_auc_vals = [v for sub in struct_scores[l].values() for v in sub]
        struct_mse_vals = [v for sub in struct_brier[l].values() for v in sub]
        rows += [
            {"Layer": l + 1, "AUC": np.nanmean(state_auc[l]),   "MSE": np.nanmean(state_brier[l]),   "Probe": "cell state"},
            {"Layer": l + 1, "AUC": np.nanmean(cand_auc[l]),    "MSE": np.nanmean(cand_brier[l]),    "Probe": "cell candidates"},
            {"Layer": l + 1, "AUC": np.nanmean(struct_auc_vals), "MSE": np.nanmean(struct_mse_vals), "Probe": "substructure candidates"},
        ]
    return pd.DataFrame(rows)


def plot(df: pd.DataFrame):
    layer_labels = sorted(df["Layer"].unique())

    fig_auc, ax_auc = plt.subplots(figsize=(4, 2.2))
    sns.lineplot(data=df, x="Layer", y="AUC", hue="Probe", style="Probe",
                 markers=["o", "s", "^"], dashes=False, ax=ax_auc)
    ax_auc.set_xticks(layer_labels)
    ax_auc.set_ylim(0.8, 1.02)
    ax_auc.legend(frameon=False)
    sns.despine(fig_auc)
    fig_auc.tight_layout()
    fig_auc.savefig(OUTPUT_AUC, bbox_inches="tight")
    print(f"Saved {OUTPUT_AUC}")

    fig_mse, ax_mse = plt.subplots(figsize=(4, 2.2))
    sns.lineplot(data=df, x="Layer", y="MSE", hue="Probe", style="Probe",
                 markers=["o", "s", "^"], dashes=False, ax=ax_mse)
    ax_mse.set_xticks(layer_labels)
    ax_mse.set_ylim(bottom=-0.005)
    ax_mse.legend(frameon=False)
    sns.despine(fig_mse)
    fig_mse.tight_layout()
    fig_mse.savefig(OUTPUT_MSE, bbox_inches="tight")
    print(f"Saved {OUTPUT_MSE}")


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
