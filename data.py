"""Sudoku data pipeline: solver, trace generation, tokenization, dataset."""

import argparse
import csv
import os
import random
import struct
import subprocess
from collections.abc import Sequence

import numpy as np

# ---------------------------------------------------------------------------
# Tokenization
# ---------------------------------------------------------------------------
# Tokens 0–728: fill tokens encoding (row, col, digit) as row*81 + col*9 + (digit-1)
# Token 729: <sep>
# Token 730: <pad>
SEP_TOKEN = 729
PAD_TOKEN = 730
VOCAB_SIZE = 731
MAX_SEQ_LEN = 82


def encode_fill(row: int, col: int, digit: int) -> int:
    return row * 81 + col * 9 + (digit - 1)


def decode_fill(token: int) -> tuple[int, int, int]:
    row, rem = divmod(token, 81)
    col, d_minus1 = divmod(rem, 9)
    return row, col, d_minus1 + 1


# ---------------------------------------------------------------------------
# Norvig-style solver with constraint propagation + backtracking
# ---------------------------------------------------------------------------

_ALL_BITS = (1 << 9) - 1  # 0b111111111 — all digits 1–9

# Precompute peer and unit tables as tuples of tuples (immutable, faster iteration)
_UNITS_INT: tuple[tuple[tuple[int, ...], ...], ...]
_PEERS_INT: tuple[tuple[int, ...], ...]

def _init_tables():
    global _UNITS_INT, _PEERS_INT
    unitlist = (
        [tuple(r * 9 + c for r in range(9)) for c in range(9)]
        + [tuple(r * 9 + c for c in range(9)) for r in range(9)]
        + [tuple((br*3+dr)*9 + bc*3+dc for dr in range(3) for dc in range(3))
           for br in range(3) for bc in range(3)]
    )
    _UNITS_INT = tuple(
        tuple(u for u in unitlist if i in u) for i in range(81)
    )
    _PEERS_INT = tuple(
        tuple(set().union(*_UNITS_INT[i]) - {i}) for i in range(81)
    )

_init_tables()

def _bit(d: int) -> int:
    return 1 << (d - 1)


class _SolverState:
    __slots__ = ("values", "trace", "clue_set", "peers", "units", "elim_order")
    def __init__(self, values, trace, clue_set, peers, units, elim_order):
        self.values = values
        self.trace = trace
        self.clue_set = clue_set
        self.peers = peers
        self.units = units
        self.elim_order = elim_order


def _shuffled_tables():
    """Return per-solve shuffled copies of peer and unit tables."""
    peers = tuple(tuple(random.sample(p, len(p))) for p in _PEERS_INT)
    units = tuple(
        tuple(tuple(random.sample(u, len(u))) for u in random.sample(cell_units, len(cell_units)))
        for cell_units in _UNITS_INT
    )
    elim_order = list(range(9))
    random.shuffle(elim_order)
    return peers, units, elim_order


def _eliminate(s: _SolverState, cell: int, d_bit: int) -> bool:
    old = s.values[cell]
    if not (old & d_bit):
        return True  # already eliminated
    new = old & ~d_bit
    if new == 0:
        return False
    s.values[cell] = new
    # Naked single: propagate to peers
    if new & (new - 1) == 0:  # popcount == 1
        if cell not in s.clue_set:
            r, c = divmod(cell, 9)
            s.trace.append((r, c, new.bit_length()))
        for peer in s.peers[cell]:
            if not _eliminate(s, peer, new):
                return False
    # Hidden single: for each unit of cell, check if d_bit has only one place
    for unit in s.units[cell]:
        count = 0
        place = -1
        for sq in unit:
            if s.values[sq] & d_bit:
                count += 1
                if count > 1:
                    break
                place = sq
        if count == 0:
            return False
        if count == 1:
            if not _assign(s, place, d_bit):
                return False
    return True


def _assign(s: _SolverState, cell: int, d_bit: int) -> bool:
    other = s.values[cell] & ~d_bit
    # Eliminate all other digits in shuffled order
    for i in s.elim_order:
        bit = 1 << i
        if other & bit:
            if not _eliminate(s, cell, bit):
                return False
    return True


def _search(s: _SolverState) -> bool:
    # Check if solved — MRV heuristic with random tie-breaking
    min_count = 10
    best_cells: list[int] = []
    for i in range(81):
        v = s.values[i]
        if v == 0:
            return False
        cnt = v.bit_count()
        if cnt > 1:
            if cnt < min_count:
                min_count = cnt
                best_cells = [i]
            elif cnt == min_count:
                best_cells.append(i)
    if not best_cells:
        return True  # all cells solved

    cell = random.choice(best_cells)
    bits = s.values[cell]
    # Try digits in shuffled order
    for i in s.elim_order:
        bit = 1 << i
        if bits & bit:
            trace_snap = len(s.trace)
            saved_values = s.values[:]
            if _assign(s, cell, bit):
                if _search(s):
                    return True
            s.values[:] = saved_values
            del s.trace[trace_snap:]
    return False


def solve(puzzle: str) -> tuple[str, list[tuple[int, int, int]]] | None:
    """Solve an 81-char puzzle string. Returns (solution_str, constraint_guided_trace) or None."""
    peers, units, elim_order = _shuffled_tables()
    s = _SolverState(
        values=[_ALL_BITS] * 81,
        trace=[],
        clue_set=set(),
        peers=peers,
        units=units,
        elim_order=elim_order,
    )
    # Parse grid: assign clues in random order
    clue_indices = [(i, ch) for i, ch in enumerate(puzzle) if '1' <= ch <= '9']
    random.shuffle(clue_indices)
    for i, ch in clue_indices:
        s.clue_set.add(i)
        if not _assign(s, i, _bit(int(ch))):
            return None

    # Check if solved
    if all(v.bit_count() == 1 for v in s.values):
        solution = "".join(str(v.bit_length()) for v in s.values)
        return solution, s.trace

    # Need search
    if _search(s):
        solution = "".join(str(v.bit_length()) for v in s.values)
        return solution, s.trace
    return None


# ---------------------------------------------------------------------------
# Trace generation modes
# ---------------------------------------------------------------------------

def random_trace(puzzle: str, solution: str) -> list[tuple[int, int, int]]:
    empties = [(i // 9, i % 9, int(solution[i])) for i in range(81) if puzzle[i] not in "123456789"]
    random.shuffle(empties)
    return empties

# ---------------------------------------------------------------------------
# Tokenize a trace into a sequence
# ---------------------------------------------------------------------------

def tokenize_trace(
    puzzle: str, solution: str, trace: list[tuple[int, int, int]], no_sep_token: bool = False, randomize_clues: bool = True,
) -> np.ndarray:
    """Convert clues + trace into a token sequence, padded to MAX_SEQ_LEN."""
    tokens = []
    clues = []
    # Clue tokens
    for i in range(81):
        if puzzle[i] in "123456789":
            r, c = divmod(i, 9)
            clues.append(encode_fill(r, c, int(puzzle[i])))
    if randomize_clues:
        random.shuffle(clues)
    tokens.extend(clues)
    if not no_sep_token:
        tokens.append(SEP_TOKEN)
    # Trace tokens
    for r, c, d in trace:
        tokens.append(encode_fill(r, c, d))

    # Truncate or pad
    if len(tokens) > MAX_SEQ_LEN:
        tokens = tokens[:MAX_SEQ_LEN]
    else:
        tokens += [PAD_TOKEN] * (MAX_SEQ_LEN - len(tokens))
    return np.array(tokens, dtype=np.int16)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class SudokuDataset:
    def __init__(self, path: str, split: str = "train"):
        data = np.load(path)
        key = f"sequences_{split}"
        if key in data:
            self.sequences = data[key]
        else:
            # Backward compat: old NPZ files without splits
            self.sequences = data["sequences"]

        # Load n_clues (number of clue tokens per puzzle)
        clue_key = f"n_clues_{split}"
        if clue_key in data:
            self.n_clues = data[clue_key]
        elif "n_clues" in data:
            self.n_clues = data["n_clues"]
        else:
            # Fallback: derive from sequences by finding SEP position
            self.n_clues = np.array([
                list(seq).index(SEP_TOKEN) if SEP_TOKEN in seq else len(seq)
                for seq in self.sequences
            ], dtype=np.int16)

    def __len__(self) -> int:
        return self.sequences.shape[0]

    def __getitem__(self, idx: int) -> np.ndarray:
        return np.array(self.sequences[idx], dtype=np.int32)


def collate_batch(dataset: SudokuDataset, indices: Sequence[int]) -> np.ndarray:
    """Gather a batch from the dataset. Returns (batch_size, MAX_SEQ_LEN) int32 array."""
    return np.asarray(dataset.sequences[indices], dtype=np.int32)


# ---------------------------------------------------------------------------
# Preprocessing / prepare
# ---------------------------------------------------------------------------

_SOLVER_BIN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "solver")


def _prepare_constraint_c(
    data_path: str, output: str, max_puzzles: int | None = None,
    randomize_clues: bool = False,
    train_frac: float = 0.90, val_frac: float = 0.05, test_frac: float = 0.05,
    seed: int = 42,
):
    """Use the C solver binary for fast constraint-trace generation."""
    # Extract puzzles from CSV
    puzzles = []
    solutions = []
    with open(data_path) as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if max_puzzles and i >= max_puzzles:
                break
            puzzles.append(row["puzzle"])
            solutions.append(row["solution"])

    # Pipe puzzles to C solver
    input_bytes = "\n".join(puzzles).encode()
    proc = subprocess.run(
        [_SOLVER_BIN], input=input_bytes, stdout=subprocess.PIPE, check=True,
    )
    buf = proc.stdout

    # Parse binary output
    sequences = []
    puzzle_strs = []
    failed = 0
    pos = 0
    for i, (puzzle, expected) in enumerate(zip(puzzles, solutions)):
        status = buf[pos]; pos += 1
        sol = buf[pos:pos+81].decode(); pos += 81
        trace_len = buf[pos]; pos += 1
        trace = []
        for _ in range(trace_len):
            r, c, d = buf[pos], buf[pos+1], buf[pos+2]
            trace.append((r, c, d))
            pos += 3
        if not status:
            failed += 1
            continue
        assert sol == expected, f"C solver mismatch at row {i}: got {sol}"
        seq = tokenize_trace(puzzle, sol, trace, True, randomize_clues)
        sequences.append(seq)
        puzzle_strs.append(puzzle)
        if (i + 1) % 100_000 == 0:
            print(f"  processed {i + 1} puzzles...")

    arr = np.stack(sequences)
    puzzle_arr = np.array(puzzle_strs, dtype="U81")
    _save_splits(arr, puzzle_arr, output, train_frac, val_frac, test_frac, seed)
    print(f"Saved {len(sequences)} sequences to {output} (failed: {failed})")


def _count_clues(puzzle: str) -> int:
    """Count non-zero clue characters in an 81-char puzzle string."""
    return sum(1 for ch in puzzle if ch in "123456789")


def _save_splits(
    sequences: np.ndarray, puzzles: np.ndarray, output: str,
    train_frac: float, val_frac: float, test_frac: float, seed: int,
):
    """Shuffle and split sequences/puzzles into train/val/test, then save to NPZ."""
    n = len(sequences)
    rng = np.random.RandomState(seed)
    indices = rng.permutation(n)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    # test gets the remainder
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    # Compute n_clues per puzzle (index where clues end / trace begins)
    n_clues = np.array([_count_clues(str(p)) for p in puzzles], dtype=np.int16)

    np.savez_compressed(
        output,
        sequences=sequences,  # backward compat
        sequences_train=sequences[train_idx],
        sequences_val=sequences[val_idx],
        sequences_test=sequences[test_idx],
        puzzles_train=puzzles[train_idx],
        puzzles_val=puzzles[val_idx],
        puzzles_test=puzzles[test_idx],
        n_clues=n_clues,  # backward compat
        n_clues_train=n_clues[train_idx],
        n_clues_val=n_clues[val_idx],
        n_clues_test=n_clues[test_idx],
    )
    print(f"Splits: {len(train_idx)} train, {len(val_idx)} val, {len(test_idx)} test")


def prepare_data(
    data_path: str, trace_mode: str, output: str, max_puzzles: int | None = None,
    no_sep_token: bool = False, randomize_clues: bool = False,
    train_frac: float = 0.90, val_frac: float = 0.05, test_frac: float = 0.05,
    seed: int = 42,
):
    # Fast path: use C solver for constraint traces
    if trace_mode == "constraint" and os.path.isfile(_SOLVER_BIN):
        print(f"Using C solver: {_SOLVER_BIN}")
        _prepare_constraint_c(data_path, output, max_puzzles, randomize_clues, train_frac, val_frac, test_frac, seed)
        return

    sequences = []
    puzzle_strs = []
    failed = 0

    with open(data_path) as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if max_puzzles and i >= max_puzzles:
                break
            puzzle = row["puzzle"]
            solution_csv = row["solution"]
            result = solve(puzzle)
            if result is None:
                failed += 1
                continue
            solution, cg_trace = result
            assert solution == solution_csv, f"Solver mismatch at row {i}"
            if trace_mode == "random":
                trace = random_trace(puzzle, solution)
            elif trace_mode == "constraint":
                trace = cg_trace
            seq = tokenize_trace(puzzle, solution, trace, no_sep_token, randomize_clues)
            assert seq.shape == (MAX_SEQ_LEN,), f"Tokenization error at row {i}"
            sequences.append(seq)
            puzzle_strs.append(puzzle)
            if (i + 1) % 10_000 == 0:
                print(f"  processed {i + 1} puzzles...")

    arr = np.stack(sequences)
    puzzle_arr = np.array(puzzle_strs, dtype="U81")
    _save_splits(arr, puzzle_arr, output, train_frac, val_frac, test_frac, seed)
    print(f"Saved {len(sequences)} sequences to {output} (failed: {failed})")


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------

def sanity_check(n: int = 100):
    """Solve n puzzles from CSV, generate random traces, verify correctness."""
    with open("sudoku-3m.csv") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i >= n:
                break
            puzzle = row["puzzle"]
            expected = row["solution"]
            result = solve(puzzle)
            assert result is not None, f"Failed to solve puzzle {i}"
            solution, cg_trace = result
            assert solution == expected, f"Solution mismatch at puzzle {i}"
            # Verify random trace
            rt = random_trace(puzzle, solution)
            grid = list(puzzle)
            for r, c, d in rt:
                grid[r * 9 + c] = str(d)
            assert "".join(grid) == expected, f"Random trace mismatch at puzzle {i}"
            # Verify tokenization round-trip
            seq = tokenize_trace(puzzle, solution, rt)
            assert seq[seq != PAD_TOKEN].shape[0] == 81 + 1, f"Token count off at puzzle {i}"  # 81 cells + sep
    print(f"All {n} puzzles passed sanity check.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--data_path", default="sudoku-3m.csv")
    parser.add_argument("--trace_mode", default="random", choices=["random", "constraint"])
    parser.add_argument("--output", default="traces_random.npz")
    parser.add_argument("--max_puzzles", type=int, default=None)
    parser.add_argument("--randomize_clues", action="store_true")
    parser.add_argument("--no_sep_token", action="store_true")
    parser.add_argument("--train_frac", type=float, default=0.90)
    parser.add_argument("--val_frac", type=float, default=0.05)
    parser.add_argument("--test_frac", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.prepare:
        prepare_data(
            args.data_path, args.trace_mode, args.output, args.max_puzzles,
            no_sep_token=args.no_sep_token, randomize_clues=args.randomize_clues,
            train_frac=args.train_frac, val_frac=args.val_frac,
            test_frac=args.test_frac, seed=args.seed,
        )
