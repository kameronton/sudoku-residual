"""Probe target building — pure sudoku logic, no ML dependencies."""

import numpy as np


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


def filter_by_mode(
    X: np.ndarray,
    targets: np.ndarray,
    labels: np.ndarray,
    mode: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Select relevant samples and form y based on probe mode.

    Returns (rel_idx, X_filtered, y_filtered) where:
    - rel_idx: indices of relevant samples
    - X_filtered: activations for relevant samples
    - y_filtered: target values for relevant samples
    """
    if mode == "candidates":
        rel = labels == 0
    elif mode == "state_filled":
        rel = labels > 0
    else:  # filled: all cells are relevant
        rel = np.ones(len(labels), dtype=bool)

    rel_idx = np.where(rel)[0]
    X_out = X[rel_idx]

    if mode == "state_filled":
        y_out = labels[rel_idx]
    elif mode == "filled":
        y_out = (labels[rel_idx] > 0).astype(int)
    else:
        y_out = targets[rel_idx]

    return rel_idx, X_out, y_out
