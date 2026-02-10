"""Visualize Sudoku solving traces step-by-step."""

import argparse
import csv
import random
import sys

from data import solve, random_trace, human_like_trace


def print_grid(grid: list[str], highlight: tuple[int, int] | None = None):
    """Print a 9x9 grid with box separators. Highlight a cell if given."""
    for r in range(9):
        if r % 3 == 0 and r > 0:
            print("------+-------+------")
        row_parts = []
        for c in range(9):
            ch = grid[r * 9 + c]
            if ch == "0" or ch == ".":
                ch = "."
            if highlight and (r, c) == highlight:
                ch = f"\033[1;32m{ch}\033[0m"  # green bold
            row_parts.append(ch)
            if c % 3 == 2 and c < 8:
                row_parts.append("|")
        print(" ".join(row_parts))


def visualize(puzzle: str, trace_mode: str = "random", step_through: bool = False):
    result = solve(puzzle)
    if result is None:
        print("Failed to solve puzzle!")
        return
    solution, cg_trace = result

    if trace_mode == "random":
        trace = random_trace(puzzle, solution)
    elif trace_mode == "constraint":
        trace = cg_trace
    elif trace_mode == "human":
        trace = human_like_trace(puzzle, solution)
    else:
        raise ValueError(f"Unknown trace mode: {trace_mode}")

    grid = list(puzzle.replace(".", "0"))
    n_clues = sum(1 for ch in puzzle if ch in "123456789")
    n_empty = 81 - n_clues

    print(f"Puzzle ({n_clues} clues, {n_empty} to fill, mode={trace_mode}):")
    print_grid(grid)
    print()

    for i, (r, c, d) in enumerate(trace):
        grid[r * 9 + c] = str(d)
        if step_through:
            print(f"Step {i + 1}/{len(trace)}: fill ({r},{c}) = {d}")
            print_grid(grid, highlight=(r, c))
            print()
            if i < len(trace) - 1:
                input("Press Enter for next step...")
        else:
            print(f"  Step {i + 1:>2d}: ({r},{c}) = {d}")

    if not step_through:
        print(f"\nFinal grid:")
        print_grid(grid)

    filled = "".join(grid)
    if filled == solution:
        print("\nTrace is correct!")
    else:
        print("\nERROR: trace does not produce correct solution!")
        print(f"  Expected: {solution}")
        print(f"  Got:      {filled}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize Sudoku solving traces")
    parser.add_argument("--puzzle", type=str, default=None, help="81-char puzzle string")
    parser.add_argument("--index", type=int, default=None, help="Row index in CSV")
    parser.add_argument("--data_path", default="sudoku-3m.csv")
    parser.add_argument("--mode", default="random", choices=["random", "constraint", "human"])
    parser.add_argument("--step", action="store_true", help="Step through one fill at a time")
    args = parser.parse_args()

    if args.puzzle:
        puzzle = args.puzzle
    else:
        idx = args.index if args.index is not None else random.randint(0, 1000)
        with open(args.data_path) as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                if i == idx:
                    puzzle = row["puzzle"]
                    print(f"Puzzle index: {idx}")
                    break
            else:
                print(f"Index {idx} not found")
                sys.exit(1)

    visualize(puzzle, trace_mode=args.mode, step_through=args.step)
