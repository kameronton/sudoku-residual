import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np

from sudoku.experiment_config import experiment_dir


def load_log(name, results_dir):
    path = os.path.join(results_dir, name, "train_log.json")
    with open(path) as f:
        return json.load(f)


def plot_loss_curves(names, results_dir, output):
    fig, ax = plt.subplots(figsize=(10, 6))
    prop_cycle = plt.rcParams["axes.prop_cycle"]
    colors = [p["color"] for p in prop_cycle]

    for i, name in enumerate(names):
        color = colors[i % len(colors)]
        data = load_log(name, results_dir)
        entries = data["entries"]
        tokens_per_step = data["tokens_per_step"]
        steps = np.array([e["step"] for e in entries])
        tokens = steps * tokens_per_step
        train_losses = np.array([e["train_loss"] for e in entries])
        val_losses = np.array([e["val_loss"] for e in entries])
        ax.plot(tokens, train_losses, alpha=0.4, color=color, label=f"{name} train")
        ax.plot(tokens, val_losses, alpha=0.9, linewidth=2, color=color, label=f"{name} val")

    ax.set_xlabel("Tokens processed")
    ax.set_ylabel("Loss")
    ax.set_title("Loss curves")
    ax.legend()
    ax.grid()

    if output:
        out_path = output
    elif len(names) == 1:
        out_path = os.path.join(experiment_dir(names[0]), "train_log.png")
    else:
        out_path = "loss_curves.png"

    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved to {out_path}")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot loss curves for one or more experiments")
    parser.add_argument("names", nargs="+", help="Experiment name(s)")
    parser.add_argument("--results-dir", default="results", help="Results directory (default: results)")
    parser.add_argument("--output", default=None, help="Output PNG path (overrides default)")
    args = parser.parse_args()
    plot_loss_curves(args.names, args.results_dir, args.output)
