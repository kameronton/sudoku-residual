"""Sudoku data pipeline: solver, trace generation, tokenization, dataset."""

import argparse
import csv
import random
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

def _cross(rows: str, cols: str) -> list[str]:
    return [r + c for r in rows for c in cols]


_ROWS = "ABCDEFGHI"
_COLS = "123456789"
_SQUARES = _cross(_ROWS, _COLS)
_UNITLIST = (
    [_cross(_ROWS, c) for c in _COLS]
    + [_cross(r, _COLS) for r in _ROWS]
    + [_cross(rs, cs) for rs in ("ABC", "DEF", "GHI") for cs in ("123", "456", "789")]
)
_UNITS = {s: [u for u in _UNITLIST if s in u] for s in _SQUARES}
_PEERS = {s: set(sum(_UNITS[s], [])) - {s} for s in _SQUARES}
_SQ_INDEX = {s: i for i, s in enumerate(_SQUARES)}


def _parse_grid(
    puzzle: str, trace: list | None = None, clue_set: set | None = None,
) -> dict[str, str] | None:
    values = {s: _COLS for s in _SQUARES}
    for s, d in zip(_SQUARES, puzzle):
        if d in _COLS:
            if not _assign(values, s, d, trace, clue_set):
                return None
    return values


def _assign(
    values: dict, s: str, d: str,
    trace: list | None = None, clue_set: set | None = None,
) -> dict | None:
    other = values[s].replace(d, "")
    for d2 in other:
        if not _eliminate(values, s, d2, trace, clue_set):
            return None
    return values


def _eliminate(
    values: dict, s: str, d: str,
    trace: list | None = None, clue_set: set | None = None,
) -> dict | None:
    if d not in values[s]:
        return values
    values[s] = values[s].replace(d, "")
    if len(values[s]) == 0:
        return None
    if len(values[s]) == 1:
        d2 = values[s]
        # Record resolution in causal order
        if trace is not None and clue_set is not None and s not in clue_set:
            idx = _SQ_INDEX[s]
            r, c = divmod(idx, 9)
            trace.append((r, c, int(d2)))
        for s2 in _PEERS[s]:
            if not _eliminate(values, s2, d2, trace, clue_set):
                return None
    for u in _UNITS[s]:
        dplaces = [sq for sq in u if d in values[sq]]
        if len(dplaces) == 0:
            return None
        if len(dplaces) == 1:
            if not _assign(values, dplaces[0], d, trace, clue_set):
                return None
    return values


def _search(values: dict, trace: list, clue_set: set) -> dict | None:
    if values is None:
        return None
    if all(len(values[s]) == 1 for s in _SQUARES):
        return values
    # Choose square with fewest candidates (MRV)
    _, s = min((len(values[s]), s) for s in _SQUARES if len(values[s]) > 1)
    for d in values[s]:
        snapshot = len(trace)
        copy = {k: v for k, v in values.items()}
        result = _assign(copy, s, d, trace, clue_set)
        if result is not None:
            out = _search(result, trace, clue_set)
            if out is not None:
                return out
        # Backtrack: discard trace entries from this failed branch
        del trace[snapshot:]
    return None


def solve(puzzle: str) -> tuple[str, list[tuple[int, int, int]]] | None:
    """Solve an 81-char puzzle string. Returns (solution_str, constraint_guided_trace) or None."""
    trace: list[tuple[int, int, int]] = []
    clue_set = {_SQUARES[i] for i, ch in enumerate(puzzle) if ch in _COLS}
    values = _parse_grid(puzzle, trace, clue_set)
    if values is None:
        return None

    if all(len(values[s]) == 1 for s in _SQUARES):
        solution = "".join(values[s] for s in _SQUARES)
        return solution, trace

    # Need search
    result = _search(values, trace, clue_set)
    if result is None:
        return None
    solution = "".join(result[s] for s in _SQUARES)
    return solution, trace


# ---------------------------------------------------------------------------
# Trace generation modes
# ---------------------------------------------------------------------------

def random_trace(puzzle: str, solution: str) -> list[tuple[int, int, int]]:
    empties = []
    for i in range(81):
        if puzzle[i] not in "123456789":
            r, c = divmod(i, 9)
            empties.append((r, c, int(solution[i])))
    random.shuffle(empties)
    return empties

# ---------------------------------------------------------------------------
# Tokenize a trace into a sequence
# ---------------------------------------------------------------------------

def tokenize_trace(
    puzzle: str, solution: str, trace: list[tuple[int, int, int]], sep_token: bool = True,
) -> np.ndarray:
    """Convert clues + trace into a token sequence, padded to MAX_SEQ_LEN."""
    tokens = []
    # Clue tokens
    for i in range(81):
        if puzzle[i] in "123456789":
            r, c = divmod(i, 9)
            tokens.append(encode_fill(r, c, int(puzzle[i])))
    if sep_token:
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
    def __init__(self, path: str):
        data = np.load(path)
        self.sequences = data["sequences"]  # (N, MAX_SEQ_LEN)

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

TRACE_GENERATORS = {
    "random": lambda puzzle, solution, _trace: random_trace(puzzle, solution),
    "constraint": lambda _puzzle, _solution, trace: trace,
}


def prepare_data(data_path: str, trace_mode: str, output: str, max_puzzles: int | None = None):
    gen_fn = TRACE_GENERATORS[trace_mode]
    sequences = []
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
            trace = gen_fn(puzzle, solution, cg_trace)
            seq = tokenize_trace(puzzle, solution, trace)
            assert seq.shape == (MAX_SEQ_LEN,), f"Tokenization error at row {i}"
            sequences.append(seq)
            if (i + 1) % 10_000 == 0:
                print(f"  processed {i + 1} puzzles...")

    arr = np.stack(sequences)
    np.savez_compressed(output, sequences=arr)
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
    parser.add_argument("--trace_mode", default="random", choices=["random", "constraint", "human"])
    parser.add_argument("--output", default="traces_random.npz")
    parser.add_argument("--max_puzzles", type=int, default=None)
    args = parser.parse_args()
    if args.prepare:
        prepare_data(args.data_path, args.trace_mode, args.output, args.max_puzzles)
