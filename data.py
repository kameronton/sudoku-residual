"""Sudoku data pipeline: trace generation, tokenization, dataset."""

import argparse
import csv
import os
import random
import subprocess
from collections.abc import Sequence

import numpy as np

from solver import solve

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
# Trace generation
# ---------------------------------------------------------------------------

def random_trace(puzzle: str, solution: str) -> list[tuple[int, int, int]]:
    empties = [(i // 9, i % 9, int(solution[i])) for i in range(81) if puzzle[i] not in "123456789"]
    random.shuffle(empties)
    return empties


def tokenize_trace(
    puzzle: str, trace: list[tuple[int, int, int]], no_sep_token: bool = False, randomize_clues: bool = True,
) -> np.ndarray:
    """Convert clues + trace into a token sequence, padded to MAX_SEQ_LEN."""
    clues = []
    for i in range(81):
        if puzzle[i] in "123456789":
            r, c = divmod(i, 9)
            clues.append(encode_fill(r, c, int(puzzle[i])))
    if randomize_clues:
        random.shuffle(clues)

    tokens = clues
    if not no_sep_token:
        tokens.append(SEP_TOKEN)
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


def _read_csv(data_path: str, max_puzzles: int | None = None):
    """Read puzzle/solution pairs from CSV."""
    puzzles, solutions = [], []
    with open(data_path) as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if max_puzzles and i >= max_puzzles:
                break
            puzzles.append(row["puzzle"])
            solutions.append(row["solution"])
    return puzzles, solutions


def _traces_from_c_solver(puzzles, solutions):
    """Run the C solver binary, return list of traces (None for failures)."""
    input_bytes = "\n".join(puzzles).encode()
    proc = subprocess.run(
        [_SOLVER_BIN], input=input_bytes, stdout=subprocess.PIPE, check=True,
    )
    buf = proc.stdout
    traces = []
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
            traces.append(None)
            continue
        assert sol == expected, f"C solver mismatch at row {i}: got {sol}"
        traces.append(trace)
    return traces


def _traces_from_python_solver(puzzles, solutions):
    """Run the Python solver, return list of traces (None for failures)."""
    traces = []
    for i, (puzzle, expected) in enumerate(zip(puzzles, solutions)):
        result = solve(puzzle)
        if result is None:
            traces.append(None)
            continue
        solution, trace = result
        assert solution == expected, f"Solver mismatch at row {i}"
        traces.append(trace)
        if (i + 1) % 10_000 == 0:
            print(f"  processed {i + 1} puzzles...")
    return traces


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
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    n_clues = np.array([
        sum(1 for ch in str(p) if ch in "123456789") for p in puzzles
    ], dtype=np.int16)

    arrays = {}
    for name, idx in [("train", train_idx), ("val", val_idx), ("test", test_idx)]:
        arrays[f"sequences_{name}"] = sequences[idx]
        arrays[f"puzzles_{name}"] = puzzles[idx]
        arrays[f"n_clues_{name}"] = n_clues[idx]

    np.savez_compressed(output, **arrays)
    print(f"Splits: {len(train_idx)} train, {len(val_idx)} val, {len(test_idx)} test")


def prepare_data(
    data_path: str, trace_mode: str, output: str, max_puzzles: int | None = None,
    no_sep_token: bool = True, randomize_clues: bool = False,
    train_frac: float = 0.90, val_frac: float = 0.05, test_frac: float = 0.05,
    seed: int = 42,
):
    puzzles, solutions = _read_csv(data_path, max_puzzles)

    # Generate traces
    if trace_mode == "constraint" and os.path.isfile(_SOLVER_BIN):
        print(f"Using C solver: {_SOLVER_BIN}")
        traces = _traces_from_c_solver(puzzles, solutions)
    elif trace_mode == "constraint":
        traces = _traces_from_python_solver(puzzles, solutions)
    else:
        traces = [random_trace(p, s) for p, s in zip(puzzles, solutions)]

    # Tokenize
    sequences = []
    puzzle_strs = []
    failed = 0
    for i, (puzzle, trace) in enumerate(zip(puzzles, traces)):
        if trace is None:
            failed += 1
            continue
        sequences.append(tokenize_trace(puzzle, trace, no_sep_token, randomize_clues))
        puzzle_strs.append(puzzle)
        if (i + 1) % 100_000 == 0:
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
            seq = tokenize_trace(puzzle, rt)
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
    parser.add_argument("--sep_token", action="store_true", help="Include SEP token between clues and trace (default: no SEP)")
    parser.add_argument("--train_frac", type=float, default=0.90)
    parser.add_argument("--val_frac", type=float, default=0.05)
    parser.add_argument("--test_frac", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.prepare:
        prepare_data(
            args.data_path, args.trace_mode, args.output, args.max_puzzles,
            no_sep_token=not args.sep_token, randomize_clues=args.randomize_clues,
            train_frac=args.train_frac, val_frac=args.val_frac,
            test_frac=args.test_frac, seed=args.seed,
        )
