"""Linear probes on residual stream activations."""

import argparse
import os

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

from data import SEP_TOKEN, decode_fill
from solver import solve
from evaluate import evaluate_puzzle
from activations import (
    load_probe_dataset, derive_n_clues, anchor_positions,
    generate_probe_dataset, sequences_to_traces,
)


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


def probe_cell(activations: np.ndarray, puzzles: list[str], cell_idx: int, mode: str = "candidates"):
    """Train and evaluate a ridge probe for a single cell.

    Returns (metric, y_true, preds, per_digit).
    """
    targets, labels = build_probe_targets(puzzles, cell_idx, mode)

    indices = np.arange(len(puzzles))
    idx_train, idx_test = train_test_split(indices, test_size=0.2, random_state=42)

    ridge = fit_probe(activations[idx_train], targets[idx_train])
    return eval_probe(ridge, activations[idx_test], targets[idx_test], labels[idx_test], mode)


def plot_all_layers(
    all_accuracies: dict[int, list[float]],
    output_path: str = "probe_accuracies.png", metric_name: str = "Accuracy",
    show: bool = True,
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
    if show:
        plt.show()
    plt.close(fig)


def plot_all_layers_per_digit(
    all_per_digit: dict[int, np.ndarray],
    output_path: str = "probe_per_digit.png",
    show: bool = True,
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
    if show:
        plt.show()
    plt.close(fig)


def metric_name_for_mode(mode: str) -> str:
    return "F1" if mode == "candidates" else "Accuracy"


def prepare_probe_inputs(
    activations: np.ndarray,
    puzzles: list[str],
    sequences: list[list[int]],
    n_clues: np.ndarray,
    step: int,
) -> tuple[np.ndarray, list[str], list[int]]:
    """Detect anchor, filter by step length, compute probe grids and positions.

    Returns (activations, probe_grids, probe_positions) after filtering.
    """
    has_sep = any(SEP_TOKEN in seq for seq in sequences[:10])
    anchor = "sep" if has_sep else "last_clue"

    traces = sequences_to_traces(sequences, n_clues)
    anchor_pos = anchor_positions(n_clues, anchor)

    # Filter puzzles with enough trace steps
    keep = [i for i, (t, ap) in enumerate(zip(traces, anchor_pos))
            if len(t) >= step and ap + step >= 0]
    if len(keep) < len(puzzles):
        print(f"Filtered to {len(keep)}/{len(puzzles)} puzzles (need >= {step} trace steps)")
        activations = activations[keep]
        puzzles = [puzzles[i] for i in keep]
        sequences = [sequences[i] for i in keep]
        anchor_pos = [anchor_pos[i] for i in keep]

    probe_positions = [ap + step for ap in anchor_pos]

    if step == 0 and anchor == "sep":
        probe_grids = puzzles
    else:
        probe_grids = [build_grid_at_step(seq, pos) for seq, pos in zip(sequences, probe_positions)]

    return activations, probe_grids, probe_positions


def build_grid_at_step(sequence: list[int], position: int) -> str:
    """Build grid state by replaying all fill tokens in sequence[0:position+1].

    Starts from an empty board and applies every fill token up to (and including)
    the given position. Works correctly for both clue tokens and trace tokens,
    and for negative step offsets (partial clue visibility).
    """
    grid = ["0"] * 81
    for tok in sequence[: position + 1]:
        if 0 <= tok <= 728:
            r, c, d = decode_fill(tok)
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


def run_probe_loop(
    activations: np.ndarray,
    probe_grids: list[str],
    probe_positions: list[int],
    mode: str = "state_filled",
) -> tuple[dict[int, list[float]], dict[int, np.ndarray]]:
    """Run probing loop across all layers and cells.

    Returns (all_accuracies, all_per_digit) dicts keyed by layer index.
    """
    n_layers = activations.shape[1]
    all_accuracies = {}
    all_per_digit = {}

    for layer in range(n_layers):
        acts = get_activations_at_positions(activations, probe_positions, layer)
        print(f"\nLayer {layer}, activations shape: {acts.shape}")

        accuracies = []
        per_digit_layer = []
        for cell in range(81):
            print(f"  Layer {layer} | Cell {cell:2d}/81", end="\r")
            metric_val, _, _, per_digit = probe_cell(acts, probe_grids, cell, mode=mode)
            accuracies.append(metric_val)
            if per_digit is not None:
                per_digit_layer.append(per_digit)

        avg = np.nanmean(accuracies)
        print(f"  Layer {layer} | Mean {metric_name_for_mode(mode).lower()} ({mode}): {avg:.3f}")
        all_accuracies[layer] = accuracies
        if per_digit_layer:
            all_per_digit[layer] = np.array(per_digit_layer)

    return all_accuracies, all_per_digit


def main():
    parser = argparse.ArgumentParser(description="Linear probes on residual stream")
    parser.add_argument("--ckpt_dir", default="checkpoints")
    parser.add_argument("--ckpt_step", type=int)
    parser.add_argument("--traces_path", default=None, help="NPZ file with test split puzzles")
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
    args = parser.parse_args()

    # --- Load data ---
    if os.path.exists(args.cache_path):
        activations, puzzles, sequences, n_clues = load_probe_dataset(args.cache_path)
    else:
        if not args.traces_path:
            raise ValueError("--traces_path required when no cache exists")
        activations, puzzles, sequences, n_clues = generate_probe_dataset(
            ckpt_dir=args.ckpt_dir, ckpt_step=args.ckpt_step,
            traces_path=args.traces_path,
            n_puzzles=args.n_puzzles, batch_size=args.batch_size,
            cache_path=args.cache_path, compress=not args.no_compress,
        )

    if n_clues is None:
        n_clues = derive_n_clues(puzzles)

    # --- Filter puzzles by solve status ---
    if args.filter != "all":
        print(f"Filtering puzzles by solve status: {args.filter}")
        traces = sequences_to_traces(sequences, n_clues)
        solve_mask = _compute_solve_mask(puzzles, traces)
        keep = np.where(solve_mask if args.filter == "solved" else ~solve_mask)[0]
        activations, puzzles, sequences = _subset_by_indices(keep, activations, puzzles, sequences)
        n_clues = n_clues[keep]
        print(f"  Kept {len(puzzles)} {args.filter} puzzles")

    # --- Prepare probe inputs ---
    activations, probe_grids, probe_positions = prepare_probe_inputs(
        activations, puzzles, sequences, n_clues, args.step,
    )

    if not probe_grids:
        print("No puzzles remaining after filtering.")
        return

    # --- Probing loop + plot ---
    all_accuracies, all_per_digit = run_probe_loop(
        activations, probe_grids, probe_positions, args.mode,
    )

    output = args.output
    if output == "probe_accuracies.png" and args.step > 0:
        output = f"probe_step{args.step}.png"

    metric = metric_name_for_mode(args.mode)
    if args.per_digit and all_per_digit:
        plot_all_layers_per_digit(all_per_digit, output.replace(".png", "_per_digit.png"))
    else:
        plot_all_layers(all_accuracies, output, metric_name=metric)


if __name__ == "__main__":
    main()
