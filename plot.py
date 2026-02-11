import argparse
import json

import matplotlib.pyplot as plt
import numpy as np


def plot_loss_curve(log_file, tokens_per_step):
    with open(log_file) as f:
        data = json.load(f)

    # Train losses — one entry per log interval
    train_losses = data["step_losses"]
    train_tokens = np.arange(1, len(train_losses) + 1) * tokens_per_step

    # Val losses — stored as [step, loss] pairs
    val_entries = data["val_losses"]
    val_steps = np.array([e[0] for e in val_entries])
    val_losses = np.array([e[1] for e in val_entries])
    val_tokens = val_steps * tokens_per_step

    plt.figure(figsize=(10, 6))
    plt.plot(train_tokens, train_losses, alpha=0.5, label="Train loss")
    plt.plot(val_tokens, val_losses, linewidth=2, label="Val loss")
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
    parser.add_argument("--tokens_per_step", type=int, required=True,
                        help="Tokens per training step (batch_size * (seq_len - 1))")
    args = parser.parse_args()
    plot_loss_curve(args.log_file, args.tokens_per_step)
