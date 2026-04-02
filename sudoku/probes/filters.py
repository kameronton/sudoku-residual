"""Input preparation and puzzle filtering."""

import numpy as np

from sudoku.data import SEP_TOKEN
from sudoku.solver import solve
from sudoku.evaluate import evaluate_puzzle
from sudoku.activations import anchor_positions, sequences_to_traces

from .activations import build_grid_at_step


def prepare_probe_inputs(
    activations: np.ndarray,
    puzzles: list[str],
    sequences: list[list[int]],
    n_clues: np.ndarray,
    step: int,
) -> tuple[np.ndarray, list[str], list[int], np.ndarray | None]:
    """Detect anchor, filter by step length, compute probe grids and positions.

    Returns (activations, probe_grids, probe_positions, keep) where keep is an
    int array of puzzle indices into activations (None if no filtering occurred).
    activations is returned unchanged — callers must pass keep to probe loops.
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
        keep_arr: np.ndarray | None = np.array(keep)
        puzzles = [puzzles[i] for i in keep]
        sequences = [sequences[i] for i in keep]
        anchor_pos = [anchor_pos[i] for i in keep]
    else:
        keep_arr = None

    probe_positions = [ap + step for ap in anchor_pos]

    if step == 0 and anchor == "sep":
        probe_grids = puzzles
    else:
        probe_grids = [build_grid_at_step(seq, pos) for seq, pos in zip(sequences, probe_positions)]

    return activations, probe_grids, probe_positions, keep_arr


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
