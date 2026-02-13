"""Linear probes on residual stream activations."""

import argparse
import csv
import os

import jax
import jax.numpy as jnp
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

from data import SEP_TOKEN, PAD_TOKEN, MAX_SEQ_LEN
from model import GPT2Model
from training import encode_clues
from evaluate import load_checkpoint, generate_traces_batched, traces_to_sequences, make_forward_fn as make_gen_forward_fn


def make_intermediates_fn(model: GPT2Model):
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




def collect_activations(intermediates_fn, params, sequences: list[list[int]], batch_size: int):
    """Forward pass on complete sequences to get activations at all layers/tokens.

    Returns array of shape (n_puzzles, n_layers, max_seq_len, d_model).
    """
    max_length = max(len(s) for s in sequences)
    padded = jnp.array(
        [s + [PAD_TOKEN] * (max_length - len(s)) for s in sequences],
        dtype=jnp.int32,
    )

    all_acts = []
    for start in range(0, len(sequences), batch_size):
        print(f"  Forward pass {start}/{len(sequences)}", end="\r")
        batch = padded[start : start + batch_size]
        _, intermediates = intermediates_fn(params, batch)
        # intermediates: (n_layers, batch, seq_len, d_model) -> (batch, n_layers, seq_len, d_model)
        all_acts.append(np.array(intermediates.transpose(1, 0, 2, 3)))
    print()

    return np.concatenate(all_acts, axis=0).astype(np.float32)


def save_probe_dataset(path: str, activations: np.ndarray, puzzles: list[str], sequences: list[list[int]]):
    """Save activations, puzzles, and token sequences together."""
    puzzle_arr = np.array(puzzles, dtype=f"U{len(puzzles[0])}")
    # Pad sequences to same length for storage
    max_len = max(len(s) for s in sequences)
    seq_arr = np.full((len(sequences), max_len), PAD_TOKEN, dtype=np.int16)
    for i, s in enumerate(sequences):
        seq_arr[i, :len(s)] = s
    np.savez_compressed(path, activations=activations, puzzles=puzzle_arr, sequences=seq_arr)
    size_mb = os.path.getsize(path) / 1e6 if os.path.exists(path) else 0
    print(f"Saved probe dataset to {path} ({activations.shape}, {size_mb:.0f} MB)")


def load_probe_dataset(path: str):
    """Load cached activations, puzzles, and sequences."""
    data = np.load(path)
    activations = data["activations"]
    puzzles = list(data["puzzles"])
    seq_arr = data["sequences"]
    # Convert back to list of lists, stripping padding
    sequences = []
    for row in seq_arr:
        seq = row[row != PAD_TOKEN].tolist()
        sequences.append(seq)
    print(f"Loaded probe dataset from {path} ({activations.shape})")
    return activations, puzzles, sequences


def get_activations_at_token(activations: np.ndarray, sequences: list[list[int]], layer: int, token_type: str = "sep"):
    """Extract activations at a specific token position for a given layer.

    activations: (n_puzzles, n_layers, seq_len, d_model)
    token_type: "sep" for SEP token position
    Returns: (n_puzzles, d_model)
    """
    layer_acts = activations[:, layer, :, :]
    
    if ":" in token_type: # concatenating two tokens, checking if the information is distributed
        token1, token2 = token_type.split(":")

        if token1 == "sep":
            positions1 = []
            for seq in sequences:
                pos = seq.index(SEP_TOKEN)
                positions1.append(pos)
        else:
            raise ValueError(f"Unsupported token_type: {token1}")
        
        if token2.startswith("sep") or token1.isdigit():
            positions2 = []
            if token2.startswith("sep"):
                shift = int(token2[3:])
                for seq in sequences:
                    pos = seq.index(SEP_TOKEN) + shift
                    positions2.append(pos)
            elif token2.isdigit():
                positions2 = [int(token2) for seq in sequences]
        else:
            raise ValueError(f"Unknown token_type: {token1}")
        
        return np.array([ np.concat([layer_acts[i, pos1], layer_acts[i, pos2]]) for i, (pos1, pos2) in enumerate(zip(positions1, positions2))])
    
    else:
        if token_type == "sep":
            positions = []
            for seq in sequences:
                pos = seq.index(SEP_TOKEN)
                positions.append(pos)
        elif token_type.startswith("sep"):
            shift = int(token_type[3:])
            positions = []
            for seq in sequences:
                pos = seq.index(SEP_TOKEN) + shift
                positions.append(pos)
        elif token_type.isdigit():
            positions = [int(token_type) for seq in sequences]
        else:
            raise ValueError(f"Unknown token_type: {token_type}")

        return np.array([layer_acts[i, pos] for i, pos in enumerate(positions)])


def _cell_candidates(puzzle: str, cell_idx: int) -> list[int]:
    """Compute candidate digits for a cell from the puzzle string (empty cell only)."""
    r, c = divmod(cell_idx, 9)
    used = set()
    for j in range(9):
        ch = puzzle[r * 9 + j]
        if ch in "123456789":
            used.add(int(ch))
        ch = puzzle[j * 9 + c]
        if ch in "123456789":
            used.add(int(ch))
    br, bc = (r // 3) * 3, (c // 3) * 3
    for dr in range(3):
        for dc in range(3):
            ch = puzzle[(br + dr) * 9 + (bc + dc)]
            if ch in "123456789":
                used.add(int(ch))
    return [1 if d not in used else 0 for d in range(1, 10)]


def probe_cell(activations: np.ndarray, puzzles: list[str], cell_idx: int, verbose: bool = False, mode: str = "state_filled"):
    """Train a ridge probe to predict a per-cell target from activations.

    Modes:
    - "filled": binary classification (filled vs empty), evaluated on all cells.
    - "state_filled": 9-class digit classification, evaluated only on filled cells.
    - "candidates": 9-dim binary candidate sets, evaluated only on empty cells.
    """
    n = len(puzzles)
    labels = np.array([int(p[cell_idx]) if p[cell_idx] in "123456789" else 0 for p in puzzles])
    filled_mask = labels > 0

    if mode == "filled":
        # Binary: 2-class one-hot [empty, filled]
        targets = np.eye(2)[(labels > 0).astype(int)]
    elif mode == "state_filled":
        # 9-class one-hot for filled cells; zero vector for empty
        targets = np.zeros((n, 9))
        targets[filled_mask] = np.eye(9)[labels[filled_mask] - 1]
    elif mode == "candidates":
        # 9-dim binary candidate vectors for empty cells; zero vector for filled
        targets = np.zeros((n, 9))
        empty_mask = ~filled_mask
        for i in np.where(empty_mask)[0]:
            targets[i] = _cell_candidates(puzzles[i], cell_idx)
    else:
        raise ValueError(f"Unsupported target mode: {mode}")

    # Single split using indices to keep labels/targets aligned
    indices = np.arange(n)
    idx_train, idx_test = train_test_split(indices, test_size=0.2, random_state=42)

    X_train, X_test = activations[idx_train], activations[idx_test]
    y_train, y_test = targets[idx_train], targets[idx_test]
    labels_train, labels_test = labels[idx_train], labels[idx_test]

    if verbose:
        n_filled = filled_mask.sum()
        print(f"\n  Dataset: {n_filled} filled, {n - n_filled} empty out of {n}")
        print(f"  {'Class':>5}  {'Train':>6}  {'Test':>6}  {'Train%':>7}  {'Test%':>7}")
        for d in range(0, 10):
            n_tr = (labels_train == d).sum()
            n_te = (labels_test == d).sum()
            pct_tr = n_tr / len(labels_train) * 100
            pct_te = n_te / len(labels_test) * 100
            label = "empty" if d == 0 else str(d)
            print(f"  {label:>5}  {n_tr:>6}  {n_te:>6}  {pct_tr:>6.1f}%  {pct_te:>6.1f}%")

    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train, y_train)
    raw_preds = ridge.predict(X_test)

    if mode == "filled":
        preds = raw_preds.argmax(axis=1)  # 0=empty, 1=filled
        y_true = (labels_test > 0).astype(int)
        acc = accuracy_score(y_true, preds)
        return acc, y_true, preds
    elif mode == "state_filled":
        # Evaluate only on filled cells
        filled_test = labels_test > 0
        if not filled_test.any():
            return float("nan"), np.array([]), np.array([])
        preds = raw_preds[filled_test].argmax(axis=1) + 1  # back to 1-9
        y_true = labels_test[filled_test]
        acc = accuracy_score(y_true, preds)
        return acc, y_true, preds
    elif mode == "candidates":
        # Evaluate only on empty cells; threshold at 0.5 for binary predictions
        empty_test = labels_test == 0
        if not empty_test.any():
            return float("nan"), np.array([]), np.array([])
        binary_preds = (raw_preds[empty_test] > 0.5).astype(int)
        y_true = y_test[empty_test]
        acc = accuracy_score(y_true.ravel(), binary_preds.ravel())  # per-digit accuracy
        return acc, y_true, binary_preds


def plot_all_layers(all_accuracies: dict[int, list[float]], empty_pcts: list[float], output_path: str = "probe_accuracies.png"):
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
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved {output_path}")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Linear probes on residual stream")
    parser.add_argument("--ckpt_dir", default="checkpoints")
    parser.add_argument("--data_path", default="sudoku-3m.csv")
    parser.add_argument("--cache_path", default="probe_acts.npz", help="Path to cache activations + puzzles")
    parser.add_argument("--token_type", default="sep")
    parser.add_argument("--output", default="probe_accuracies.png")
    parser.add_argument("--n_puzzles", type=int, default=6400)
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    if os.path.exists(args.cache_path):
        activations, puzzles, sequences = load_probe_dataset(args.cache_path)
    else:
        params, model = load_checkpoint(args.ckpt_dir)
        print("Model loaded")

        puzzles = load_puzzles(args.data_path, args.n_puzzles)
        print(f"Loaded {len(puzzles)} puzzles")

        # Step 1: batched autoregressive trace generation
        gen_fn = make_gen_forward_fn(model)
        print("Generating traces...")
        traces = generate_traces_batched(gen_fn, params, puzzles, args.batch_size)
        sequences = traces_to_sequences(puzzles, traces)
        avg_len = np.mean([len(s) for s in sequences])
        print(f"Average sequence length: {avg_len:.1f}")

        # Step 2: single batched forward pass for activations
        print("Collecting activations...")
        intermediates_fn = make_intermediates_fn(model)
        activations = collect_activations(intermediates_fn, params, sequences, args.batch_size)
        save_probe_dataset(args.cache_path, activations, puzzles, sequences)

    n_layers = activations.shape[1]
    empty_pcts = []
    for cell in range(81):
        n_empty = sum(1 for p in puzzles if p[cell] not in "123456789")
        empty_pcts.append(n_empty / len(puzzles) * 100)

    all_accuracies = {}
    for layer in range(n_layers):
        sep_acts = get_activations_at_token(activations, sequences, layer, args.token_type)
        print(f"\nLayer {layer}, activations shape: {sep_acts.shape}")
        accuracies = []
        for cell in range(81):
            print(f"  Layer {layer} | Cell {cell:2d}/81", end="\r")
            acc, _, _ = probe_cell(sep_acts, puzzles, cell)
            accuracies.append(acc)
        avg = sum(accuracies) / len(accuracies)
        print(f"  Layer {layer} | Mean accuracy (filled): {avg:.3f}")
        all_accuracies[layer] = accuracies

    plot_all_layers(all_accuracies, empty_pcts, args.output)


if __name__ == "__main__":
    main()
