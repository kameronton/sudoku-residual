"""Linear probes on residual stream activations at the SEP token."""

import argparse
import csv
import os

import jax
import jax.numpy as jnp
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

from data import SEP_TOKEN, PAD_TOKEN, encode_fill, MAX_SEQ_LEN
from model import GPT2Model, TransformerConfig
from training import encode_clues
from evaluate import load_checkpoint


def make_forward_fn(model: GPT2Model):
    @jax.jit
    def forward(params, tokens):
        return model.apply({"params": params}, tokens, return_intermediates=True)
    return forward


def load_puzzles(data_path: str, n: int) -> list[str]:
    puzzles = []
    with open(data_path) as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i >= n:
                break
            puzzles.append(row["puzzle"])
    return puzzles


def collect_activations(forward_fn, params, puzzles: list[str], batch_size: int):
    """Get residual stream activations at all layers and tokens.

    Returns array of shape (n_puzzles, n_layers, seq_len, d_model).
    """
    tokenized = [encode_clues(p) for p in puzzles]
    max_length = max(len(t) for t in tokenized)
    padded = jnp.array(
        [t + [PAD_TOKEN] * (max_length - len(t)) for t in tokenized],
        dtype=jnp.int32,
    )

    all_acts = []
    for start in range(0, len(puzzles), batch_size):
        batch = padded[start : start + batch_size]
        _, intermediates = forward_fn(params, batch)
        # intermediates: (n_layers, batch, seq_len, d_model) -> (batch, n_layers, seq_len, d_model)
        all_acts.append(np.array(intermediates.transpose(1, 0, 2, 3)))

    return np.concatenate(all_acts, axis=0).astype(np.float32)


def save_probe_dataset(path: str, activations: np.ndarray, puzzles: list[str]):
    """Save activations and puzzles together."""
    # Encode puzzles as fixed-length byte strings
    puzzle_arr = np.array(puzzles, dtype=f"U{len(puzzles[0])}")
    np.savez_compressed(path, activations=activations, puzzles=puzzle_arr)
    print(f"Saved probe dataset to {path} ({activations.shape})")


def load_probe_dataset(path: str):
    """Load cached activations and puzzles."""
    data = np.load(path)
    activations = data["activations"]
    puzzles = list(data["puzzles"])
    print(f"Loaded probe dataset from {path} ({activations.shape})")
    return activations, puzzles


def get_activations_at(activations: np.ndarray, puzzles: list[str], layer: int):
    """Extract activations at the SEP token for a given layer.

    activations: (n_puzzles, n_layers, seq_len, d_model)
    Returns: (n_puzzles, d_model)
    """
    tokenized = [encode_clues(p) for p in puzzles]
    sep_positions = [len(t) - 1 for t in tokenized]
    layer_acts = activations[:, layer, :, :]  # (n_puzzles, seq_len, d_model)
    return np.array([layer_acts[i, pos] for i, pos in enumerate(sep_positions)])


def probe_cell(activations: np.ndarray, puzzles: list[str], cell_idx: int, verbose: bool = False):
    """Train a ridge regression to predict the digit at a given cell from activations.

    Targets are 9-dim one-hot (digits 1-9), empty cells get zero vector. Predicted class = argmax + 1.
    """
    labels = np.array([int(p[cell_idx]) if p[cell_idx] in "123456789" else 0 for p in puzzles])
    # 9-class one-hot: digits 1-9 map to cols 0-8, empty (0) maps to zero vector
    targets = np.zeros((len(labels), 9))
    filled_mask = labels > 0
    targets[filled_mask] = np.eye(9)[labels[filled_mask] - 1]

    X_train, X_test, y_train_oh, y_test = train_test_split(
        activations, targets, test_size=0.2, random_state=42,
    )
    # Split labels in parallel for statistics (using same random_state)
    _, _, labels_train, labels_test = train_test_split(
        activations, labels, test_size=0.2, random_state=42,
    )
    y_test_labels = y_test.argmax(axis=1) + 1  # back to 1-9

    if verbose:
        n_filled = filled_mask.sum()
        print(f"\n  Dataset: {n_filled} filled, {len(labels) - n_filled} empty out of {len(labels)}")
        print(f"  {'Class':>5}  {'Train':>6}  {'Test':>6}  {'Train%':>7}  {'Test%':>7}")
        for d in range(0, 10):
            n_tr = (labels_train == d).sum()
            n_te = (labels_test == d).sum()
            pct_tr = n_tr / len(labels_train) * 100
            pct_te = n_te / len(labels_test) * 100
            label = "empty" if d == 0 else str(d)
            print(f"  {label:>5}  {n_tr:>6}  {n_te:>6}  {pct_tr:>6.1f}%  {pct_te:>6.1f}%")

    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train, y_train_oh)
    preds = ridge.predict(X_test).argmax(axis=1) + 1  # back to 1-9

    # Evaluate only on filled cells (empty cells have zero-vector targets, not meaningful)
    filled = labels_test != 0
    acc = accuracy_score(y_test_labels[filled], preds[filled]) if filled.any() else float("nan")

    return acc, y_test_labels[filled], preds[filled]


def plot_all_layers(all_accuracies: dict[int, list[float]], empty_pcts: list[float]):
    """Plot 9x9 heatmap per layer with shared colorbar."""
    import matplotlib.pyplot as plt

    n_layers = len(all_accuracies)
    ncols = min(n_layers, 3)
    nrows = (n_layers + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols + 2, 6 * nrows))
    if n_layers == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    empty_grid = np.array(empty_pcts).reshape(9, 9)
    ims = []
    for idx, (layer, accs) in enumerate(sorted(all_accuracies.items())):
        ax = axes[idx]
        grid = np.array(accs).reshape(9, 9)
        im = ax.imshow(grid, cmap="RdYlGn", vmin=0, vmax=1)
        ims.append(im)
        avg = np.mean(accs)
        ax.set_title(f"Layer {layer} (mean={avg:.3f})")
        ax.set_xticks(range(9))
        ax.set_yticks(range(9))
        for r in range(9):
            for c in range(9):
                ax.text(c, r - 0.12, f"{grid[r, c]:.2f}", ha="center", va="center", fontsize=7)
                ax.text(c, r + 0.18, f"{empty_grid[r, c]:.0f}%", ha="center", va="center", fontsize=5, color="gray")
        for i in range(0, 10, 3):
            ax.axhline(i - 0.5, color="black", linewidth=2)
            ax.axvline(i - 0.5, color="black", linewidth=2)

    # Hide unused axes
    for idx in range(n_layers, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle("Per-cell probe accuracy by layer", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.colorbar(ims[0], ax=axes[:n_layers].tolist(), shrink=0.6, label="Accuracy", pad=0.02)
    fig.savefig("probe_accuracies.png", dpi=150, bbox_inches="tight")
    print("Saved probe_accuracies.png")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Linear probes on residual stream")
    parser.add_argument("--ckpt_dir", default="checkpoints")
    parser.add_argument("--data_path", default="sudoku-3m.csv")
    parser.add_argument("--cache_path", default="probe_acts.npz", help="Path to cache activations + puzzles")
    parser.add_argument("--n_puzzles", type=int, default=6400)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--n_layers", type=int, default=6)
    parser.add_argument("--n_heads", type=int, default=6)
    parser.add_argument("--d_model", type=int, default=384)
    parser.add_argument("--d_ff", type=int, default=1536)
    parser.add_argument("--dtype", default="bfloat16")
    args = parser.parse_args()

    if os.path.exists(args.cache_path):
        activations, puzzles = load_probe_dataset(args.cache_path)
    else:
        model_cfg = TransformerConfig(
            n_layers=args.n_layers, n_heads=args.n_heads,
            d_model=args.d_model, d_ff=args.d_ff, dtype=args.dtype,
        )
        params, model = load_checkpoint(args.ckpt_dir, model_cfg)
        forward_fn = make_forward_fn(model)
        print("Model loaded")

        puzzles = load_puzzles(args.data_path, args.n_puzzles)
        print(f"Loaded {len(puzzles)} puzzles")

        activations = collect_activations(forward_fn, params, puzzles, args.batch_size)
        save_probe_dataset(args.cache_path, activations, puzzles)

    n_layers = activations.shape[1]
    empty_pcts = []
    for cell in range(81):
        n_empty = sum(1 for p in puzzles if p[cell] not in "123456789")
        empty_pcts.append(n_empty / len(puzzles) * 100)

    all_accuracies = {}
    for layer in range(n_layers):
        sep_acts = get_activations_at(activations, puzzles, layer)
        print(f"\nLayer {layer}, activations shape: {sep_acts.shape}")
        accuracies = []
        for cell in range(81):
            print(f"  Layer {layer} | Cell {cell:2d}/81", end="\r")
            acc, _, _ = probe_cell(sep_acts, puzzles, cell)
            accuracies.append(acc)
        avg = sum(accuracies) / len(accuracies)
        print(f"  Layer {layer} | Mean accuracy (filled): {avg:.3f}")
        all_accuracies[layer] = accuracies

    plot_all_layers(all_accuracies, empty_pcts)


if __name__ == "__main__":
    main()
