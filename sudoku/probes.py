"""Linear probes on residual stream activations."""

import argparse
import os

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from sudoku.data import SEP_TOKEN, decode_fill
from sudoku.solver import solve
from sudoku.evaluate import evaluate_puzzle
from sudoku.activations import (
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


def fit_probe(X_train: np.ndarray, y_train: np.ndarray, mode: str, C: float = 1.0):
    """Fit a logistic probe.

    y_train format:
    - filled: (n,) binary {0, 1}
    - state_filled: (n,) int {1..9}
    - candidates / structure: (n, 9) binary multi-label
    """
    if mode in ("candidates", "structure"):
        clf = MultiOutputClassifier(LogisticRegression(C=C, max_iter=1000))
        clf.fit(X_train, y_train.astype(int))
    else:
        clf = LogisticRegression(C=C, max_iter=1000)
        clf.fit(X_train, y_train)
    return clf


def eval_probe(clf, X_test: np.ndarray, y_test: np.ndarray, mode: str):
    """Evaluate a fitted probe using AUC and Brier score.

    Returns (auc, brier, y_true, per_digit_auc, per_digit_brier).
    per_digit_* are (9,) arrays for candidates/structure, None otherwise.
    """
    if mode == "filled":
        probas = clf.predict_proba(X_test)[:, 1]
        if len(np.unique(y_test)) < 2:
            return float("nan"), float("nan"), y_test, None, None
        brier = float(np.mean((probas - y_test) ** 2))
        return roc_auc_score(y_test, probas), brier, y_test, None, None

    elif mode == "state_filled":
        probas = clf.predict_proba(X_test)
        full_probas = np.zeros((len(X_test), 9))
        for j, cls in enumerate(clf.classes_):
            full_probas[:, int(cls) - 1] = probas[:, j]
        present = np.unique(y_test)
        if len(present) < 2:
            return float("nan"), float("nan"), y_test, None, None
        col_idx = [int(c) - 1 for c in present]
        auc = roc_auc_score(y_test, full_probas[:, col_idx], multi_class="ovr", average="macro")
        y_onehot = np.zeros((len(y_test), 9))
        y_onehot[np.arange(len(y_test)), y_test - 1] = 1.0
        brier = float(np.mean((full_probas - y_onehot) ** 2))
        return auc, brier, y_test, None, None

    elif mode in ("candidates", "structure"):
        proba_list = clf.predict_proba(X_test)
        probas = np.column_stack([p[:, 1] for p in proba_list])
        per_digit_auc = np.array([
            roc_auc_score(y_test[:, d], probas[:, d])
            if len(np.unique(y_test[:, d])) > 1 else float("nan")
            for d in range(9)
        ])
        per_digit_brier = np.mean((probas - y_test) ** 2, axis=0)
        return np.nanmean(per_digit_auc), float(np.mean(per_digit_brier)), y_test, per_digit_auc, per_digit_brier

    else:
        raise ValueError(f"Unsupported mode: {mode}")


def probe_cell(activations: np.ndarray, puzzles: list[str], cell_idx: int, mode: str = "candidates"):
    """Train and evaluate a logistic probe for a single cell.

    Returns (auc, y_true, per_digit_or_None).
    """
    targets, labels = build_probe_targets(puzzles, cell_idx, mode)

    # Filter to cells relevant for this mode before fitting
    if mode == "candidates":
        rel = labels == 0
    elif mode == "state_filled":
        rel = labels > 0
    else:  # filled: all cells are relevant
        rel = np.ones(len(labels), dtype=bool)

    rel_idx = np.where(rel)[0]
    X = activations[rel_idx]
    if mode == "state_filled":
        y = labels[rel_idx]
    elif mode == "filled":
        y = (labels[rel_idx] > 0).astype(int)
    else:
        y = targets[rel_idx]

    if len(X) < 4:
        return float("nan"), y, (np.full(9, float("nan")) if mode == "candidates" else None)

    idx = np.arange(len(rel_idx))
    idx_train, idx_test = train_test_split(idx, test_size=0.2, random_state=42)
    clf = fit_probe(X[idx_train], y[idx_train], mode)
    return eval_probe(clf, X[idx_test], y[idx_test], mode)


def plot_all_layers(
    all_accuracies: dict[int, list[float]],
    output_path: str = "probe_accuracies.png", metric_name: str = "Accuracy",
    show: bool = True, vmin: float = 0.0, vmax: float = 1.0, cmap: str = "RdYlGn",
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
        im = ax.imshow(grid, cmap=cmap, vmin=vmin, vmax=vmax)
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
    if n_layers == 8:
        ncols = 4
        nrows = 2
    else:
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
    return "AUC"


def build_structure_targets(puzzles: list[str], subtype: str, idx: int) -> np.ndarray:
    """Build (n, 9) binary targets: target[i, d] = 1 if digit d+1 is present in the substructure."""
    n = len(puzzles)
    targets = np.zeros((n, 9))
    for i, puzzle in enumerate(puzzles):
        if subtype == "row":
            cells = puzzle[idx * 9:(idx + 1) * 9]
        elif subtype == "col":
            cells = [puzzle[r * 9 + idx] for r in range(9)]
        else:  # box
            br, bc = (idx // 3) * 3, (idx % 3) * 3
            cells = [puzzle[(br + dr) * 9 + (bc + dc)] for dr in range(3) for dc in range(3)]
        for ch in cells:
            if ch in "123456789":
                targets[i, int(ch) - 1] = 1.0
    return targets


def probe_structure(acts: np.ndarray, puzzles: list[str], subtype: str, idx: int) -> tuple[float, float]:
    """Fit and evaluate a structure probe for one row/col/box. Returns (mean AUC, mean Brier)."""
    targets = build_structure_targets(puzzles, subtype, idx)
    indices = np.arange(len(puzzles))
    idx_train, idx_test = train_test_split(indices, test_size=0.2, random_state=42)
    clf = fit_probe(acts[idx_train], targets[idx_train], mode="structure")
    proba_list = clf.predict_proba(acts[idx_test])
    probas = np.column_stack([p[:, 1] for p in proba_list])
    per_digit_auc = [
        roc_auc_score(targets[idx_test, d], probas[:, d])
        if len(np.unique(targets[idx_test, d])) > 1 else float("nan")
        for d in range(9)
    ]
    per_digit_brier = np.mean((probas - targets[idx_test]) ** 2, axis=0)
    return float(np.nanmean(per_digit_auc)), float(np.mean(per_digit_brier))


def run_structure_probe_loop(
    activations: np.ndarray,
    probe_grids: list[str],
    probe_positions: list[int],
) -> tuple[dict[int, dict[str, list[float]]], dict[int, dict[str, list[float]]]]:
    """Run structure probing: 27 probes per layer (9 rows, 9 cols, 9 boxes).

    Returns (all_auc, all_brier), each dict[layer, dict[subtype, list[float]]].
    """
    n_layers = activations.shape[1]
    all_scores = {}
    all_brier = {}
    for layer in range(n_layers):
        acts = get_activations_at_positions(activations, probe_positions, layer)
        print(f"\nLayer {layer}, activations shape: {acts.shape}")
        layer_scores: dict[str, list[float]] = {"row": [], "col": [], "box": []}
        layer_brier: dict[str, list[float]] = {"row": [], "col": [], "box": []}
        for subtype in ("row", "col", "box"):
            for idx in range(9):
                print(f"  Layer {layer} | {subtype} {idx}/9", end="\r")
                auc, brier = probe_structure(acts, probe_grids, subtype, idx)
                layer_scores[subtype].append(auc)
                layer_brier[subtype].append(brier)
        for subtype in ("row", "col", "box"):
            avg_auc = np.nanmean(layer_scores[subtype])
            avg_brier = np.nanmean(layer_brier[subtype])
            print(f"  Layer {layer} | Mean AUC ({subtype}): {avg_auc:.3f}  Brier: {avg_brier:.4f}")
        all_scores[layer] = layer_scores
        all_brier[layer] = layer_brier
    return all_scores, all_brier


def plot_structure(
    all_scores: dict[int, dict[str, list[float]]],
    output_path: str = "probe_structure.png",
    show: bool = True, vmin: float = 0.0, vmax: float = 1.0, cmap: str = "RdYlGn",
):
    """Plot structure probe F1: n_layers rows x 3 cols (row/col/box substructures).

    Row/col subtypes are shown as a 1x9 horizontal heatmap strip.
    Box subtype is shown as a 3x3 heatmap matching the Sudoku grid layout.
    """
    import matplotlib.pyplot as plt

    n_layers = len(all_scores)
    subtypes = ["row", "col", "box"]
    col_titles = ["Rows", "Columns", "Boxes"]
    fig, axes = plt.subplots(n_layers, 3, figsize=(6.5, 1.5 * n_layers), constrained_layout=True)
    if n_layers == 1:
        axes = axes[np.newaxis, :]

    ims = []
    for layer_idx, (layer, scores) in enumerate(sorted(all_scores.items())):
        for col_idx, subtype in enumerate(subtypes):
            ax = axes[layer_idx, col_idx]
            vals = np.array(scores[subtype])
            if subtype == "box":
                data = vals.reshape(3, 3)
            elif subtype == "row":
                data = vals.reshape(9, 1)
            else:  # col
                data = vals.reshape(1, 9)
            im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
            if layer_idx == 0:
                ims.append(im)
            for r in range(data.shape[0]):
                for c in range(data.shape[1]):
                    ax.text(c, r, f"{data[r, c]:.2f}", ha="center", va="center", fontsize=5)
            ax.set_xticks([])
            ax.set_yticks([])
            if layer_idx == 0:
                ax.set_title(col_titles[col_idx], fontsize=9)
            if col_idx == 0:
                ax.set_ylabel(f"L{layer}", fontsize=8, rotation=0, labelpad=20, va="center")

    fig.colorbar(ims[0], ax=axes.ravel().tolist(), shrink=0.5, pad=0.02)
    fig.suptitle("Structure probe (row / col / box)", fontsize=11)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved {output_path}")
    if show:
        plt.show()
    plt.close(fig)


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


def compute_deltas(activations: np.ndarray) -> np.ndarray:
    """Replace cumulative activations with per-layer deltas.

    activations: (n_puzzles, n_layers, seq_len, d_model)
    Returns same shape: delta[0] = acts[0], delta[i] = acts[i] - acts[i-1] for i >= 1.
    """
    return np.concatenate([activations[:, :1], np.diff(activations, axis=1)], axis=1)


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
) -> tuple[dict[int, list[float]], dict[int, np.ndarray], dict[int, list[float]]]:
    """Run probing loop across all layers and cells.

    Returns (all_auc, all_per_digit_auc, all_brier) dicts keyed by layer index.
    """
    n_layers = activations.shape[1]
    all_accuracies = {}
    all_per_digit = {}
    all_brier = {}

    for layer in range(n_layers):
        acts = get_activations_at_positions(activations, probe_positions, layer)
        print(f"\nLayer {layer}, activations shape: {acts.shape}")

        accuracies = []
        briers = []
        per_digit_layer = []
        for cell in range(81):
            print(f"  Layer {layer} | Cell {cell:2d}/81", end="\r")
            metric_val, brier_val, _, per_digit_auc, _ = probe_cell(acts, probe_grids, cell, mode=mode)
            accuracies.append(metric_val)
            briers.append(brier_val)
            if per_digit_auc is not None:
                per_digit_layer.append(per_digit_auc)

        avg_auc = np.nanmean(accuracies)
        avg_brier = np.nanmean(briers)
        print(f"  Layer {layer} | Mean {metric_name_for_mode(mode).lower()} ({mode}): {avg_auc:.3f}  Brier: {avg_brier:.4f}")
        all_accuracies[layer] = accuracies
        all_brier[layer] = briers
        if per_digit_layer:
            all_per_digit[layer] = np.array(per_digit_layer)

    return all_accuracies, all_per_digit, all_brier


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
    parser.add_argument("--mode", default="state_filled", choices=["filled", "state_filled", "candidates", "structure"])
    parser.add_argument("--per-digit", action="store_true", help="Per-digit F1 heatmap (candidates mode only)")
    parser.add_argument("--use-deltas", action="store_true", help="Probe layer-wise deltas (acts[i] - acts[i-1]) instead of cumulative activations")
    parser.add_argument("--step", type=int, default=0, help="Trace step to probe at (0 = SEP/initial board, 1 = after first fill, ...)")
    parser.add_argument("--filter", choices=["all", "solved", "unsolved"], default="all",
                        help="Filter puzzles for both train and eval")
    args = parser.parse_args()

    # --- Load data ---
    if os.path.exists(args.cache_path):
        activations, puzzles, sequences, n_clues, _ = load_probe_dataset(args.cache_path)
    else:
        if not args.traces_path:
            raise ValueError("--traces_path required when no cache exists")
        activations, puzzles, sequences, n_clues, _ = generate_probe_dataset(
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

    if args.use_deltas:
        activations = compute_deltas(activations)

    output = args.output
    if output == "probe_accuracies.png" and args.step > 0:
        output = f"probe_step{args.step}.png"
    if args.use_deltas:
        output = output.replace(".png", "_deltas.png")

    # --- Probing loop + plot ---
    if args.mode == "structure":
        all_scores, all_brier_struct = run_structure_probe_loop(activations, probe_grids, probe_positions)
        plot_structure(all_scores, output)
        plot_structure(all_brier_struct, output.replace(".png", "_brier.png"),
                       vmin=0.0, vmax=0.25, cmap="RdYlGn_r")
    else:
        all_accuracies, all_per_digit, all_brier = run_probe_loop(
            activations, probe_grids, probe_positions, args.mode,
        )
        metric = metric_name_for_mode(args.mode)
        if args.per_digit and all_per_digit:
            plot_all_layers_per_digit(all_per_digit, output.replace(".png", "_per_digit.png"))
        else:
            plot_all_layers(all_accuracies, output, metric_name=metric)
        plot_all_layers(all_brier, output.replace(".png", "_brier.png"),
                        metric_name="Brier", vmin=0.0, vmax=0.25, cmap="RdYlGn_r")


if __name__ == "__main__":
    main()
