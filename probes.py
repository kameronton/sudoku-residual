"""Linear probes on residual stream activations."""

import argparse
import csv
import os
import sys

import jax
import jax.numpy as jnp
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, f1_score

from data import SEP_TOKEN, PAD_TOKEN, MAX_SEQ_LEN, solve
from model import GPT2Model
from evaluate import load_checkpoint, generate_traces_batched, generate_traces_batched_cached, sequences_to_traces, make_forward_fn as make_gen_forward_fn, encode_clues, evaluate_puzzle


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


def save_probe_dataset(path: str, activations: np.ndarray, puzzles: list[str], sequences: list[list[int]], compress: bool = True):
    """Save activations, puzzles, and token sequences together."""
    puzzle_arr = np.array(puzzles, dtype=f"U{len(puzzles[0])}")
    # Pad sequences to same length for storage
    max_len = max(len(s) for s in sequences)
    seq_arr = np.full((len(sequences), max_len), PAD_TOKEN, dtype=np.int16)
    for i, s in enumerate(sequences):
        seq_arr[i, :len(s)] = s
    save_fn = np.savez_compressed if compress else np.savez
    save_fn(path, activations=activations, puzzles=puzzle_arr, sequences=seq_arr)
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


def build_probe_targets(puzzles: list[str], cell_idx: int, mode: str):
    """Build target vectors and labels for a cell probe.

    Returns (targets, labels) where:
    - labels: (n,) int array, digit 1-9 for filled cells, 0 for empty
    - targets: (n, k) float array, format depends on mode
    """
    n = len(puzzles)
    labels = np.array([int(p[cell_idx]) if p[cell_idx] in "123456789" else 0 for p in puzzles])
    filled_mask = labels > 0

    if mode == "filled":
        targets = np.eye(2)[(labels > 0).astype(int)]
    elif mode == "state_filled":
        targets = np.zeros((n, 9))
        targets[filled_mask] = np.eye(9)[labels[filled_mask] - 1]
    elif mode == "candidates":
        targets = np.zeros((n, 9))
        empty_mask = ~filled_mask
        for i in np.where(empty_mask)[0]:
            targets[i] = _cell_candidates(puzzles[i], cell_idx)
    else:
        raise ValueError(f"Unsupported target mode: {mode}")

    return targets, labels


def fit_probe(X_train: np.ndarray, y_train: np.ndarray, alpha: float = 1.0) -> Ridge:
    """Fit a Ridge probe on training data."""
    ridge = Ridge(alpha=alpha)
    ridge.fit(X_train, y_train)
    return ridge


def eval_probe(ridge: Ridge, X_test: np.ndarray, y_test: np.ndarray, labels_test: np.ndarray, mode: str):
    """Evaluate a fitted probe on test data.

    Returns (metric, y_true, preds, per_digit).
    """
    raw_preds = ridge.predict(X_test)

    if mode == "filled":
        preds = raw_preds.argmax(axis=1)
        y_true = (labels_test > 0).astype(int)
        acc = accuracy_score(y_true, preds)
        return acc, y_true, preds, None
    elif mode == "state_filled":
        filled_test = labels_test > 0
        if not filled_test.any():
            return float("nan"), np.array([]), np.array([]), None
        preds = raw_preds[filled_test].argmax(axis=1) + 1
        y_true = labels_test[filled_test]
        acc = accuracy_score(y_true, preds)
        return acc, y_true, preds, None
    elif mode == "candidates":
        empty_test = labels_test == 0
        if not empty_test.any():
            return float("nan"), np.array([]), np.array([]), np.full(9, float("nan"))
        binary_preds = (raw_preds[empty_test] > 0.5).astype(int)
        y_true = y_test[empty_test]
        per_digit = np.array([
            f1_score(y_true[:, d], binary_preds[:, d], zero_division=0.0)
            for d in range(9)
        ])
        f1 = per_digit.mean()
        return f1, y_true, binary_preds, per_digit
    else:
        raise ValueError(f"Unsupported mode: {mode}")


def probe_cell(activations: np.ndarray, puzzles: list[str], cell_idx: int, verbose: bool = False, mode: str = "candidates"):
    """Train and evaluate a ridge probe for a single cell. Convenience wrapper.

    Modes:
    - "filled": binary classification (filled vs empty), evaluated on all cells.
    - "state_filled": 9-class digit classification, evaluated only on filled cells.
    - "candidates": 9-dim binary candidate sets, evaluated only on empty cells.
    """
    targets, labels = build_probe_targets(puzzles, cell_idx, mode)

    indices = np.arange(len(puzzles))
    idx_train, idx_test = train_test_split(indices, test_size=0.2, random_state=42)

    if verbose:
        n_filled = (labels > 0).sum()
        n = len(puzzles)
        labels_train, labels_test = labels[idx_train], labels[idx_test]
        print(f"\n  Dataset: {n_filled} filled, {n - n_filled} empty out of {n}")
        print(f"  {'Class':>5}  {'Train':>6}  {'Test':>6}  {'Train%':>7}  {'Test%':>7}")
        for d in range(0, 10):
            n_tr = (labels_train == d).sum()
            n_te = (labels_test == d).sum()
            pct_tr = n_tr / len(labels_train) * 100
            pct_te = n_te / len(labels_test) * 100
            label = "empty" if d == 0 else str(d)
            print(f"  {label:>5}  {n_tr:>6}  {n_te:>6}  {pct_tr:>6.1f}%  {pct_te:>6.1f}%")

    ridge = fit_probe(activations[idx_train], targets[idx_train])
    return eval_probe(ridge, activations[idx_test], targets[idx_test], labels[idx_test], mode)


def plot_all_layers(
    all_accuracies: dict[int, list[float]],
    output_path: str = "probe_accuracies.png", metric_name: str = "Accuracy",
):
    """Plot 9x9 heatmap per layer with shared colorbar."""
    import matplotlib.pyplot as plt

    n_layers = len(all_accuracies)
    if n_layers == 8:
        ncols = 4
        nrows = 2
    else:
        ncols = min(n_layers, 3)
        nrows = (n_layers + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols + 0.5, 3 * nrows))
    if n_layers == 1:
        axes = np.array([axes])
    axes = axes.flatten()

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
                ax.text(c, r, f"{grid[r, c]:.2f}", ha="center", va="center", fontsize=6)
        for i in range(0, 10, 3):
            ax.axhline(i - 0.5, color="black", linewidth=2)
            ax.axvline(i - 0.5, color="black", linewidth=2)

    # Hide unused axes
    for idx in range(n_layers, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle(f"Per-cell probe {metric_name.lower()} by layer", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.colorbar(ims[0], ax=axes[:n_layers].tolist(), shrink=0.6, label=metric_name, pad=0.02)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved {output_path}")
    plt.show()


def plot_all_layers_per_digit(
    all_per_digit: dict[int, np.ndarray],
    output_path: str = "probe_per_digit.png",
):
    """Plot 9×9 grid per layer where each cell contains a 3×3 mini-heatmap of per-digit F1."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    n_layers = len(all_per_digit)
    ncols = min(n_layers, 3)
    nrows = (n_layers + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows),
                             gridspec_kw={"wspace": 0.1, "hspace": 0.1})
    if n_layers == 1:
        axes = np.array([axes])
    axes = np.atleast_1d(axes).flatten()

    norm = Normalize(vmin=0, vmax=1)
    cmap = plt.get_cmap("RdYlGn")

    for idx, (layer, per_digit_arr) in enumerate(sorted(all_per_digit.items())):
        ax = axes[idx]
        ax.set_xlim(-0.5, 8.5)
        ax.set_ylim(8.5, -0.5)
        ax.set_aspect("equal")
        ax.set_xticks(range(9))
        ax.set_yticks(range(9))
        ax.tick_params(length=0)

        avg = np.nanmean(per_digit_arr)
        ax.set_title(f"Layer {layer} (mean F1={avg:.3f})")

        # Draw sudoku box lines
        for i in range(0, 10, 3):
            ax.axhline(i - 0.5, color="black", linewidth=2)
            ax.axvline(i - 0.5, color="black", linewidth=2)
        # Thin cell lines
        for i in range(10):
            ax.axhline(i - 0.5, color="gray", linewidth=0.5)
            ax.axvline(i - 0.5, color="gray", linewidth=0.5)

        for cell in range(81):
            r, c = divmod(cell, 9)
            digits = per_digit_arr[cell]  # shape (9,)
            mini = digits.reshape(3, 3)
            # Inset axes: map cell (c, r) to figure coords
            inset = ax.inset_axes(
                [c - 0.5, r - 0.5, 1, 1],
                transform=ax.transData,
            )
            inset.imshow(mini, cmap=cmap, norm=norm, aspect="equal")
            inset.set_xticks([])
            inset.set_yticks([])
            inset.patch.set_alpha(0)
            # Cell boundary
            for spine in inset.spines.values():
                spine.set_edgecolor("gray")
                spine.set_linewidth(0.5)
            # Intra-cell grid lines between the 3×3 digits
            for pos in [0.5, 1.5]:
                inset.axhline(pos, color="gray", linewidth=0.3, alpha=0.6)
                inset.axvline(pos, color="gray", linewidth=0.3, alpha=0.6)

    for idx in range(n_layers, len(axes)):
        axes[idx].set_visible(False)

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    fig.colorbar(sm, ax=axes[:n_layers].tolist(), shrink=0.6, label="F1", pad=0.02)
    fig.suptitle("Per-digit candidate F1 by layer", fontsize=14, y=1.02)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved {output_path}")
    plt.show()


def build_grid_at_step(puzzle: str, trace: list[tuple[int, int, int]], step: int) -> str:
    """Replay the first step+1 fill actions on the puzzle grid. Returns 81-char grid string."""
    grid = list(puzzle)
    for i in range(min(step + 1, len(trace))):
        r, c, d = trace[i]
        grid[r * 9 + c] = str(d)
    return "".join(grid)


def get_activations_at_positions(activations: np.ndarray, positions: list[int], layer: int) -> np.ndarray:
    """Extract activations at specific per-puzzle positions for a given layer.

    activations: (n_puzzles, n_layers, seq_len, d_model)
    positions: list of length n_puzzles, position index per puzzle
    Returns: (n_puzzles, d_model)
    """
    layer_acts = activations[:, layer, :, :]
    idx = np.arange(len(positions))
    return layer_acts[idx, positions]


def get_activations_multi_token(activations: np.ndarray, positions: list[list[int]], layer: int) -> np.ndarray:
    """Extract and concatenate activations at multiple positions per puzzle.

    activations: (n_puzzles, n_layers, seq_len, d_model)
    positions: list of length n_puzzles, each a list of position indices
    Returns: (n_puzzles, n_positions * d_model)
    """
    layer_acts = activations[:, layer, :, :]
    parts = []
    for tok_idx in range(len(positions[0])):
        pos = [p[tok_idx] for p in positions]
        idx = np.arange(len(pos))
        parts.append(layer_acts[idx, pos])
    return np.concatenate(parts, axis=1)


def _compute_solve_mask(puzzles, traces):
    """Return boolean array: True for puzzles the model solved correctly."""
    mask = np.zeros(len(puzzles), dtype=bool)
    for i, (puzzle, trace) in enumerate(zip(puzzles, traces)):
        result = solve(puzzle)
        if result is None:
            continue
        stats = evaluate_puzzle(trace, puzzle, result[0], verbose=False)
        mask[i] = stats["puzzle_solved"]
    return mask


def _subset_by_indices(keep, activations, puzzles, sequences, traces=None):
    """Return subsetted copies of all arrays/lists by index list."""
    activations = activations[keep]
    puzzles = [puzzles[i] for i in keep]
    sequences = [sequences[i] for i in keep]
    if traces is not None:
        traces = [traces[i] for i in keep]
        return activations, puzzles, sequences, traces
    return activations, puzzles, sequences


def main():
    parser = argparse.ArgumentParser(description="Linear probes on residual stream")
    parser.add_argument("--ckpt_dir", default="checkpoints")
    parser.add_argument("--ckpt_step", type=int)
    parser.add_argument("--data_path", default="sudoku-3m.csv")
    parser.add_argument("--cache_path", default="probe_acts.npz", help="Path to cache activations + puzzles")
    parser.add_argument("--output", default="probe_accuracies.png")
    parser.add_argument("--n_puzzles", type=int, default=6400)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--no-compress", action="store_true", help="Skip compression when saving probe cache")
    parser.add_argument("--mode", default="state_filled", choices=["filled", "state_filled", "candidates"])
    parser.add_argument("--per-digit", action="store_true", help="Per-digit F1 heatmap (candidates mode only)")
    parser.add_argument("--step", type=int, default=0, help="Trace step to probe at (0 = SEP/initial board, 1 = after first fill, ...)")
    parser.add_argument("--filter", choices=["all", "solved", "unsolved"], default="all",
                        help="Filter puzzles for both train and eval")
    parser.add_argument("--eval-filter", choices=["all", "solved", "unsolved"], default="all",
                        help="Train on all, evaluate only on this subset of held-out data")
    parser.add_argument("--positions", type=str, default="0",
                        help="Comma-separated offsets relative to SEP+step (e.g. '-2,-1,0'). "
                             "Ground truth = board state after max offset fills.")
    # Preprocess sys.argv: find --positions and merge its value if split by argparse
    argv = sys.argv[1:]
    for i, arg in enumerate(argv):
        if arg == "--positions" and i + 1 < len(argv):
            # Force argparse to see it as one token: --positions=VALUE
            argv[i] = f"--positions={argv[i + 1]}"
            del argv[i + 1]
            break
    args = parser.parse_args(argv)

    offsets = [int(x) for x in args.positions.split(",")]

    # --- Load data ---
    if os.path.exists(args.cache_path):
        activations, puzzles, sequences = load_probe_dataset(args.cache_path)
    else:
        params, model = load_checkpoint(args.ckpt_dir, ckpt_step=args.ckpt_step)
        print("Model loaded")

        puzzles = load_puzzles(args.data_path, args.n_puzzles)
        print(f"Loaded {len(puzzles)} puzzles")

        print("Generating traces...")
        traces, sequences = generate_traces_batched_cached(model, params, puzzles, args.batch_size)
        avg_len = np.mean([len(s) for s in sequences])
        print(f"Average sequence length: {avg_len:.1f}")

        print("Collecting activations...")
        intermediates_fn = make_intermediates_fn(model)
        activations = collect_activations(intermediates_fn, params, sequences, args.batch_size)
        save_probe_dataset(args.cache_path, activations, puzzles, sequences, compress=not args.no_compress)

    traces = sequences_to_traces(sequences)

    # --- Filter puzzles (affects both train and eval) ---
    if args.filter != "all":
        print(f"Filtering puzzles by solve status: {args.filter}")
        solve_mask = _compute_solve_mask(puzzles, traces)
        if args.filter == "solved":
            keep = np.where(solve_mask)[0]
        else:
            keep = np.where(~solve_mask)[0]
        activations, puzzles, sequences, traces = _subset_by_indices(keep, activations, puzzles, sequences, traces)
        print(f"  Kept {len(puzzles)} {args.filter} puzzles")

    # --- Build probe grids and positions ---
    step = args.step
    # The ground truth board state is determined by step (the max offset anchor).
    # Each offset in `offsets` is relative to SEP+step.
    min_offset = min(offsets)

    # Filter puzzles that have enough trace steps and enough prefix tokens
    required_trace_steps = step  # need at least `step` fills after SEP
    keep = []
    for i, (seq, t) in enumerate(zip(sequences, traces)):
        sep_pos = seq.index(SEP_TOKEN)
        abs_min = sep_pos + step + min_offset
        if len(t) >= required_trace_steps and abs_min >= 0:
            keep.append(i)
    if len(keep) < len(puzzles):
        print(f"Filtered to {len(keep)}/{len(puzzles)} puzzles (need >= {required_trace_steps} trace steps, min position >= 0)")
        activations, puzzles, sequences, traces = _subset_by_indices(keep, activations, puzzles, sequences, traces)

    if not puzzles:
        print("No puzzles remaining after filtering.")
        return

    # Ground truth = board after `step` fills (step=0 means initial board)
    if step == 0:
        probe_grids = puzzles
    else:
        probe_grids = [build_grid_at_step(p, t, step - 1) for p, t in zip(puzzles, traces)]

    # Build per-puzzle position lists for activation extraction
    use_multi = len(offsets) > 1
    sep_positions = [seq.index(SEP_TOKEN) for seq in sequences]
    if use_multi:
        probe_positions_multi = [[sp + step + off for off in offsets] for sp in sep_positions]
    probe_positions = [sp + step for sp in sep_positions]

    # --- Compute eval mask (train on all, eval on subset) ---
    eval_mask = None
    if args.eval_filter != "all":
        print(f"Eval filter: {args.eval_filter} (train on all, evaluate on {args.eval_filter})")
        solve_mask = _compute_solve_mask(puzzles, traces)
        if args.eval_filter == "solved":
            eval_mask = solve_mask
        else:
            eval_mask = ~solve_mask
        n_eval = eval_mask.sum()
        print(f"  {n_eval}/{len(puzzles)} puzzles in eval set")
        if n_eval == 0:
            print("No puzzles in eval set.")
            return

    # --- Probing loop ---
    n_layers = activations.shape[1]
    all_accuracies = {}
    all_per_digit = {}
    for layer in range(n_layers):
        if use_multi:
            acts = get_activations_multi_token(activations, probe_positions_multi, layer)
        else:
            acts = get_activations_at_positions(activations, probe_positions, layer)
        print(f"\nLayer {layer}, activations shape: {acts.shape}")

        accuracies = []
        per_digit_layer = []
        for cell in range(81):
            print(f"  Layer {layer} | Cell {cell:2d}/81", end="\r")

            if eval_mask is not None:
                # Train on 80% of all, eval on solved/unsolved subset of held-out 20%
                targets, labels = build_probe_targets(probe_grids, cell, args.mode)
                indices = np.arange(len(probe_grids))
                idx_train, idx_test = train_test_split(indices, test_size=0.2, random_state=42)
                ridge = fit_probe(acts[idx_train], targets[idx_train])
                # Filter test set by eval_mask
                test_eval = np.intersect1d(idx_test, np.where(eval_mask)[0])
                if len(test_eval) == 0:
                    metric_val, per_digit = float("nan"), None
                else:
                    metric_val, _, _, per_digit = eval_probe(
                        ridge, acts[test_eval], targets[test_eval], labels[test_eval], args.mode,
                    )
            else:
                metric_val, _, _, per_digit = probe_cell(acts, probe_grids, cell, mode=args.mode)

            accuracies.append(metric_val)
            if per_digit is not None:
                per_digit_layer.append(per_digit)

        metric = "F1" if args.mode == "candidates" else "Accuracy"
        avg = np.nanmean(accuracies)
        print(f"  Layer {layer} | Mean {metric.lower()} ({args.mode}): {avg:.3f}")
        all_accuracies[layer] = accuracies
        if per_digit_layer:
            all_per_digit[layer] = np.array(per_digit_layer)

    # --- Output ---
    output = args.output
    if output == "probe_accuracies.png" and step > 0:
        output = f"probe_step{step}.png"
    if output == "probe_accuracies.png" and use_multi:
        pos_tag = "_".join(str(o) for o in offsets)
        output = f"probe_positions_{pos_tag}.png"

    if args.per_digit and all_per_digit:
        plot_all_layers_per_digit(all_per_digit, output.replace(".png", "_per_digit.png"))
    else:
        plot_all_layers(all_accuracies, output, metric_name=metric)


if __name__ == "__main__":
    main()
