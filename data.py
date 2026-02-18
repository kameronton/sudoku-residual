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
    unitlist: list[tuple[int, ...]] = []
    for c in range(9):
        unitlist.append(tuple(r * 9 + c for r in range(9)))
    for r in range(9):
        unitlist.append(tuple(r * 9 + c for c in range(9)))
    for br in range(3):
        for bc in range(3):
            unitlist.append(tuple(
                (br * 3 + dr) * 9 + (bc * 3 + dc)
                for dr in range(3) for dc in range(3)
            ))
    units_list = []
    peers_list = []
    for i in range(81):
        my_units = tuple(tuple(u) for u in unitlist if i in u)
        units_list.append(my_units)
        peer_set: set[int] = set()
        for u in my_units:
            peer_set.update(u)
        peer_set.discard(i)
        peers_list.append(tuple(peer_set))
    _UNITS_INT = tuple(units_list)
    _PEERS_INT = tuple(peers_list)

_init_tables()

def _bit(d: int) -> int:
    return 1 << (d - 1)


def _eliminate(
    values: list[int], cell: int, d_bit: int,
    trace: list | None, clue_set: set[int] | None,
) -> bool:
    old = values[cell]
    if not (old & d_bit):
        return True  # already eliminated
    new = old & ~d_bit
    if new == 0:
        return False
    values[cell] = new
    # Naked single: propagate to peers
    if new & (new - 1) == 0:  # popcount == 1
        # Record trace
        if trace is not None and clue_set is not None and cell not in clue_set:
            r, c = divmod(cell, 9)
            trace.append((r, c, new.bit_length()))
        for peer in _PEERS_INT[cell]:
            if not _eliminate(values, peer, new, trace, clue_set):
                return False
    # Hidden single: for each unit of cell, check if d_bit has only one place
    for unit in _UNITS_INT[cell]:
        count = 0
        place = -1
        for sq in unit:
            if values[sq] & d_bit:
                count += 1
                if count > 1:
                    break
                place = sq
        if count == 0:
            return False
        if count == 1:
            if not _assign(values, place, d_bit, trace, clue_set):
                return False
    return True


def _assign(
    values: list[int], cell: int, d_bit: int,
    trace: list | None, clue_set: set[int] | None,
) -> bool:
    other = values[cell] & ~d_bit
    # Eliminate all other digits
    bit = 1
    while other:
        if other & 1:
            if not _eliminate(values, cell, bit, trace, clue_set):
                return False
        other >>= 1
        bit <<= 1
    return True


def _search(
    values: list[int], trace: list, clue_set: set[int], random_mrv: bool,
) -> bool:
    # Check if solved — MRV heuristic
    min_count = 10
    best_cell = -1
    best_cells: list[int] | None = [] if random_mrv else None
    for i in range(81):
        v = values[i]
        if v == 0:
            return False
        cnt = v.bit_count()
        if cnt > 1:
            if cnt < min_count:
                min_count = cnt
                best_cell = i
                if random_mrv:
                    best_cells = [i]
            elif random_mrv and cnt == min_count:
                best_cells.append(i)
    if best_cell == -1:
        return True  # all cells solved

    cell = random.choice(best_cells) if random_mrv else best_cell
    bits = values[cell]
    bit = 1
    while bit <= bits:
        if bits & bit:
            trace_snap = len(trace)
            copy = values[:]
            if _assign(copy, cell, bit, trace, clue_set):
                if _search(copy, trace, clue_set, random_mrv):
                    values[:] = copy
                    return True
            del trace[trace_snap:]
        bit <<= 1
    return False


def solve(puzzle: str, *, random_mrv: bool = True) -> tuple[str, list[tuple[int, int, int]]] | None:
    """Solve an 81-char puzzle string. Returns (solution_str, constraint_guided_trace) or None.

    Args:
        random_mrv: When True (default), break MRV ties randomly during search.
            Set to False for faster solving when trace variety is not needed.
    """
    trace: list[tuple[int, int, int]] = []
    clue_set: set[int] = set()
    values = [_ALL_BITS] * 81
    # Parse grid: assign clues
    for i, ch in enumerate(puzzle):
        if '1' <= ch <= '9':
            clue_set.add(i)
            if not _assign(values, i, _bit(int(ch)), trace, clue_set):
                return None

    # Check if solved
    if all(v.bit_count() == 1 for v in values):
        solution = "".join(str(v.bit_length()) for v in values)
        return solution, trace

    # Need search
    if _search(values, trace, clue_set, random_mrv):
        solution = "".join(str(v.bit_length()) for v in values)
        return solution, trace
    return None


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
    puzzle: str, solution: str, trace: list[tuple[int, int, int]], sep_token: bool = True, randomize_clues: bool = False,
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


_SOLVER_BIN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "solver")


def _prepare_constraint_c(
    data_path: str, output: str, max_puzzles: int | None = None,
    randomize_clues: bool = False,
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
        if (i + 1) % 100_000 == 0:
            print(f"  processed {i + 1} puzzles...")

    arr = np.stack(sequences)
    np.savez_compressed(output, sequences=arr)
    print(f"Saved {len(sequences)} sequences to {output} (failed: {failed})")


def prepare_data(data_path: str, trace_mode: str, output: str, max_puzzles: int | None = None, sep_token: bool = True, randomize_clues: bool = False):
    # Fast path: use C solver for constraint traces
    if trace_mode == "constraint" and os.path.isfile(_SOLVER_BIN):
        print(f"Using C solver: {_SOLVER_BIN}")
        _prepare_constraint_c(data_path, output, max_puzzles, randomize_clues)
        return

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
            seq = tokenize_trace(puzzle, solution, trace, sep_token, randomize_clues)
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
    parser.add_argument("--randomize_clues", action="store_true")
    args = parser.parse_args()
    if args.prepare:
        if args.randomize_clues:
            print("The clues order is randomized")
        else:
            print("The clues order is NOT randomized")
        prepare_data(args.data_path, args.trace_mode, args.output, args.max_puzzles, sep_token=True, randomize_clues=args.randomize_clues)
