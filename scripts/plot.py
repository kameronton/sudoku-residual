import argparse
import json

import matplotlib.pyplot as plt
import numpy as np


def plot_loss_curve(log_file):
    with open(log_file) as f:
        data = json.load(f)

    entries = data["entries"]
    tokens_per_step = data["tokens_per_step"]

    steps = np.array([e["step"] for e in entries])
    train_losses = np.array([e["train_loss"] for e in entries])
    val_losses = np.array([e["val_loss"] for e in entries])
    tokens = steps * tokens_per_step

    plt.figure(figsize=(10, 6))
    plt.plot(tokens, train_losses, alpha=0.5, label="Train loss")
    plt.plot(tokens, val_losses, alpha=0.5, linewidth=2, label="Val loss")
    plt.xlabel("Tokens processed")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid()
    out_path = log_file.rsplit(".", 1)[0] + ".png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved to {out_path}")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot loss curves from train_log.json")
    parser.add_argument("log_file", type=str, help="Path to train_log.json")
    args = parser.parse_args()
    plot_loss_curve(args.log_file)
