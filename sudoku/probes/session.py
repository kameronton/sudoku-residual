"""ProbeSession and ActivationIndex — query activations by board state.

Typical usage::

    session = ProbeSession("results/baseline/activations.npz")

    # Classic: one sample per puzzle at step 0 (SEP token)
    idx = session.index.at_step(0)
    acts = session.acts(idx, layer=4)
    grids = session.grids(idx)

    # One sample per puzzle where exactly 35 cells are filled
    idx = session.index.where_filled(35).first_per_puzzle()
    acts = session.acts(idx, layer=4)
    grids = session.grids(idx)

    # Arbitrary predicate: cell 40 is still empty
    acts, grids, puzzle_idx = session.query(
        layer=4, predicate=lambda g: g[40] == "0"
    )

    # Puzzle-level train/test split
    idx = session.index.where_filled(20).first_per_puzzle()
    train_mask, test_mask = session.split(idx)
    acts = session.acts(idx, layer=4)
    clf.fit(acts[train_mask], ...)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from sudoku.data import SEP_TOKEN
from sudoku.activations import load_probe_dataset, derive_n_clues
from sudoku.data_bt import PUSH_TOKEN, POP_TOKEN

from .activations import build_grid_at_step, get_activations_at_positions


@dataclass
class ActivationIndex:
    """Flat index of (puzzle, position) samples with precomputed board metadata.

    All arrays are parallel — element k describes one sample:
        puzzle_idx[k], seq_pos[k], step[k], n_filled[k]

    Filter with the methods below, then pass to ProbeSession.acts() / .grids()
    to materialise data.
    """

    puzzle_idx: np.ndarray  # (N,) int32 — row in activations/sequences arrays
    seq_pos:    np.ndarray  # (N,) int32 — token position in the sequence
    step:       np.ndarray  # (N,) int32 — offset from anchor (0 = SEP / last_clue)
    n_filled:   np.ndarray  # (N,) int16 — cells filled at this position (incl. clues)

    def __len__(self) -> int:
        return len(self.puzzle_idx)

    def __getitem__(self, idx) -> ActivationIndex:
        return ActivationIndex(
            puzzle_idx=self.puzzle_idx[idx],
            seq_pos=   self.seq_pos[idx],
            step=      self.step[idx],
            n_filled=  self.n_filled[idx],
        )

    def filter(self, mask: np.ndarray) -> ActivationIndex:
        """Return a sub-index using a boolean or integer mask."""
        return self[mask]

    def at_step(self, step: int) -> ActivationIndex:
        """Keep only samples at exactly this step offset from the anchor."""
        return self.filter(self.step == step)

    def where_filled(self, n: int) -> ActivationIndex:
        """Keep only samples where exactly n cells are filled (including clues)."""
        return self.filter(self.n_filled == n)

    def first_per_puzzle(self) -> ActivationIndex:
        """Keep the first (lowest seq_pos) sample for each puzzle.

        Useful after where_filled() on BT traces, where a given cell count
        can appear multiple times per puzzle due to backtracking.
        """
        _, first = np.unique(self.puzzle_idx, return_index=True)
        return self[first]

    def last_per_puzzle(self) -> ActivationIndex:
        """Keep the last (highest seq_pos) sample for each puzzle."""
        rev = self[::-1]
        _, first_of_rev = np.unique(rev.puzzle_idx, return_index=True)
        return rev[first_of_rev]


class ProbeSession:
    """Load activations once; query by board state, step, or arbitrary predicate.

    The index covers all trace positions (anchor onward) for every puzzle.
    Filter it with ActivationIndex methods, then call acts() / grids() / split().
    """

    def __init__(self, cache_path: str, max_step: int | None = None):
        """
        cache_path  Path to the .npz metadata file; the companion _acts.npy is
                    loaded automatically as an OS-level mmap.
        max_step    If set, include only trace positions with step <= max_step.
                    Saves memory for long BT traces.
        """
        self.activations, self.puzzles, self.sequences, self.n_clues, self.solutions = \
            load_probe_dataset(cache_path)
        if self.n_clues is None:
            self.n_clues = derive_n_clues(self.puzzles)

        self.has_sep = any(SEP_TOKEN in seq for seq in self.sequences[:10])
        self.n_puzzles = len(self.puzzles)
        self.n_layers = self.activations.shape[1] if self.activations is not None else 0

        print("Building activation index...", end=" ", flush=True)
        self.index = self._build_index(max_step)
        print(f"{len(self.index):,} samples ({self.n_puzzles:,} puzzles)")

    def _build_index(self, max_step: int | None) -> ActivationIndex:
        puzzle_idxs:  list[int] = []
        seq_positions: list[int] = []
        steps:         list[int] = []
        n_filleds:     list[int] = []

        for i, (seq, nc) in enumerate(zip(self.sequences, self.n_clues)):
            # Anchor: SEP position (= n_clues) or last clue position (= n_clues - 1)
            anchor = int(nc) if self.has_sep else int(nc) - 1

            # Compute n_filled incrementally, respecting BT push/pop semantics.
            # Clue tokens (0-728) are counted too, so n_filled at the anchor
            # equals n_clues[i] for standard traces.
            n_filled = 0
            stack: list[int] = []

            for pos, tok in enumerate(seq):
                if 0 <= tok <= 728:
                    n_filled += 1
                elif tok == PUSH_TOKEN:
                    stack.append(n_filled)
                elif tok == POP_TOKEN:
                    if stack:
                        n_filled = stack.pop()

                if pos < anchor:
                    continue
                k = pos - anchor
                if max_step is not None and k > max_step:
                    break

                puzzle_idxs.append(i)
                seq_positions.append(pos)
                steps.append(k)
                n_filleds.append(n_filled)

        return ActivationIndex(
            puzzle_idx=np.array(puzzle_idxs,   dtype=np.int32),
            seq_pos=   np.array(seq_positions, dtype=np.int32),
            step=      np.array(steps,         dtype=np.int32),
            n_filled=  np.array(n_filleds,     dtype=np.int16),
        )

    # ------------------------------------------------------------------
    # Data access
    # ------------------------------------------------------------------

    def acts(
        self,
        index: ActivationIndex,
        layer: int,
        use_deltas: bool = False,
    ) -> np.ndarray:
        """Extract activations (N, d_model) for samples in index at the given layer."""
        if self.activations is None:
            raise RuntimeError("No activations in this dataset (traces-only).")
        return get_activations_at_positions(
            self.activations,
            index.seq_pos.tolist(),
            layer,
            keep=index.puzzle_idx,
            use_deltas=use_deltas,
        )

    def grids(self, index: ActivationIndex) -> list[str]:
        """Reconstruct board-state strings for each sample in index.

        Each string is 81 chars, '0' for empty, '1'-'9' for filled.
        Computed from scratch on every call — store the result if needed again.
        """
        return [
            build_grid_at_step(self.sequences[int(pi)], int(sp))
            for pi, sp in zip(index.puzzle_idx, index.seq_pos)
        ]

    # ------------------------------------------------------------------
    # Train/test split
    # ------------------------------------------------------------------

    def split(
        self,
        index: ActivationIndex,
        test_size: float = 0.2,
        seed: int = 42,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Puzzle-level train/test split.

        Splits on unique puzzle_idx values so that all samples from the same
        puzzle end up in the same partition — preventing data leakage when
        multiple positions from one puzzle appear in the index (e.g. after
        where_filled() on a BT trace).

        Returns (train_mask, test_mask), both boolean arrays of length len(index).
        """
        unique_puzzles = np.unique(index.puzzle_idx)
        rng = np.random.default_rng(seed)
        rng.shuffle(unique_puzzles)
        n_test = max(1, int(len(unique_puzzles) * test_size))
        test_set = set(unique_puzzles[:n_test].tolist())
        test_mask = np.fromiter(
            (int(p) in test_set for p in index.puzzle_idx),
            dtype=bool,
            count=len(index),
        )
        return ~test_mask, test_mask

    # ------------------------------------------------------------------
    # Convenience query
    # ------------------------------------------------------------------

    def query(
        self,
        layer: int,
        *,
        step: int | None = None,
        n_filled: int | None = None,
        predicate: Callable[[str], bool] | None = None,
        per_puzzle: str = "first",
        use_deltas: bool = False,
    ) -> tuple[np.ndarray, list[str], np.ndarray]:
        """Filter, extract activations and grids in one call.

        Filters are applied in order: step → n_filled → predicate.

        per_puzzle  "first" — keep earliest matching position per puzzle (default)
                    "last"  — keep latest matching position per puzzle
                    "all"   — keep every matching position; use puzzle-level split()

        Returns (acts, grids, puzzle_idx):
            acts        (N, d_model) float32
            grids       list of N 81-char board-state strings
            puzzle_idx  (N,) int32, pass to split() for grouped splitting
        """
        if per_puzzle not in ("first", "last", "all"):
            raise ValueError(f"per_puzzle must be 'first', 'last', or 'all', got {per_puzzle!r}")

        idx = self.index
        if step is not None:
            idx = idx.at_step(step)
        if n_filled is not None:
            idx = idx.where_filled(n_filled)
        if predicate is not None:
            candidate_grids = self.grids(idx)
            mask = np.fromiter(
                (predicate(g) for g in candidate_grids),
                dtype=bool,
                count=len(idx),
            )
            idx = idx.filter(mask)

        if per_puzzle == "first":
            idx = idx.first_per_puzzle()
        elif per_puzzle == "last":
            idx = idx.last_per_puzzle()

        return self.acts(idx, layer, use_deltas), self.grids(idx), idx.puzzle_idx
