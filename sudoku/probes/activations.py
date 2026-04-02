"""Activation extraction and grid state reconstruction."""

import numpy as np

from sudoku.data import decode_fill
from sudoku.data_bt import PUSH_TOKEN, POP_TOKEN


def build_grid_at_step(sequence: list[int], position: int) -> str:
    """Build grid state by replaying tokens up to position, respecting PUSH/POP backtracking.

    Maintains a stack so that fills placed inside a failed branch are correctly
    undone when a POP is encountered. Non-BT sequences have no PUSH/POP tokens
    so the behaviour is identical to the previous implementation for those.
    """
    grid = ["0"] * 81
    stack: list[list[str]] = []
    for tok in sequence[: position + 1]:
        if tok == PUSH_TOKEN:
            stack.append(grid[:])
        elif tok == POP_TOKEN:
            if stack:
                grid = stack.pop()
        elif 0 <= tok <= 728:
            r, c, d = decode_fill(tok)
            grid[r * 9 + c] = str(d)
    return "".join(grid)


def compute_deltas(activations: np.ndarray) -> np.ndarray:
    """Replace cumulative activations with per-layer deltas.

    activations: (n_puzzles, n_layers, seq_len, d_model)
    Returns same shape: delta[0] = acts[0], delta[i] = acts[i] - acts[i-1] for i >= 1.
    """
    return np.concatenate([activations[:, :1], np.diff(activations, axis=1)], axis=1)


def get_activations_at_positions(
    activations: np.ndarray,
    positions: list[int],
    layer: int,
    keep: np.ndarray | None = None,
    use_deltas: bool = False,
) -> np.ndarray:
    """Extract activations at specific per-puzzle positions for a given layer.

    activations: (n_total, n_layers, seq_len, d_model) — may be a memmap
    positions: list of length n_keep, per-puzzle probe token position
    keep: int array of puzzle indices into activations; None means 0..n_keep-1
    use_deltas: if True, return acts[layer] - acts[layer-1] (identity at layer 0)
    Returns: (n_keep, d_model)

    Uses a direct 3-D fancy index so only the needed elements are read from disk.
    """
    idx = keep if keep is not None else np.arange(len(positions))
    pos = np.array(positions)
    acts = activations[idx, layer, pos]
    if use_deltas and layer > 0:
        acts = acts - activations[idx, layer - 1, pos]
    return acts
