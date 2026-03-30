"""Evaluate model traces against ground-truth Sudoku solutions."""

import argparse

import numpy as np

from sudoku.data import SEP_TOKEN, decode_fill
from sudoku.data_bt import PUSH_TOKEN, POP_TOKEN, SUCCESS_TOKEN
from sudoku.solver import solve
from sudoku.visualize import print_grid


def _is_consistent(grid: list[str], r: int, c: int, d: int) -> bool:
    """Check if placing digit d at (r,c) is consistent with current grid state."""
    ds = str(d)
    for j in range(9):
        if j != c and grid[r * 9 + j] == ds:
            return False
    for i in range(9):
        if i != r and grid[i * 9 + c] == ds:
            return False
    br, bc = (r // 3) * 3, (c // 3) * 3
    for i in range(br, br + 3):
        for j in range(bc, bc + 3):
            if (i, j) != (r, c) and grid[i * 9 + j] == ds:
                return False
    return True


def evaluate_puzzle(trace: list[tuple[int, int, int]], puzzle: str, solution: str, verbose: bool = True) -> dict:
    """Evaluate a generated trace against a puzzle. Returns stats dict."""
    grid = list(puzzle)
    n_empties = sum(1 for ch in puzzle if ch not in "123456789")

    correct = 0
    overwrites_clue = 0
    overwrites_fill = 0
    inconsistent = 0
    wrong_consistent = 0
    filled_positions = set()

    details = []
    for r, c, d in trace:
        pos = r * 9 + c
        if puzzle[pos] in "123456789":
            overwrites_clue += 1
            details.append((r, c, d, "OVERWRITES_CLUE"))
            continue
        if pos in filled_positions:
            overwrites_fill += 1
            details.append((r, c, d, "OVERWRITES_FILL"))
            continue
        filled_positions.add(pos)
        consistent = _is_consistent(grid, r, c, d)
        grid[pos] = str(d)
        if str(d) == solution[pos]:
            correct += 1
            details.append((r, c, d, "CORRECT"))
        elif not consistent:
            inconsistent += 1
            details.append((r, c, d, "INCONSISTENT"))
        else:
            wrong_consistent += 1
            details.append((r, c, d, "WRONG_CONSISTENT"))

    missing = n_empties - len(filled_positions)

    if verbose:
        print(f"  Empties: {n_empties} | Correct: {correct} | "
              f"Wrong(consistent): {wrong_consistent} | Inconsistent: {inconsistent} | "
              f"Clue overwrite: {overwrites_clue} | Fill overwrite: {overwrites_fill} | "
              f"Missing: {missing}")
        print("  Trace:")
        for i, (r, c, d, kind) in enumerate(details):
            expected = solution[r * 9 + c]
            if kind == "CORRECT":
                print(f"    {i+1:3d}. ({r},{c})={d}  CORRECT")
            else:
                print(f"    {i+1:3d}. ({r},{c})={d}  {kind} (expected={expected})")
        print("  Model output:")
        print_grid(grid)

    return {
        "n_empties": n_empties,
        "correct": correct,
        "wrong_consistent": wrong_consistent,
        "inconsistent": inconsistent,
        "overwrites_clue": overwrites_clue,
        "overwrites_fill": overwrites_fill,
        "missing": missing,
        "cell_accuracy": correct / n_empties if n_empties > 0 else 1.0,
        "puzzle_solved": correct == n_empties,
    }


def evaluate_traces(
    puzzles: list[str],
    traces: list[list[tuple[int, int, int]]],
    quiet: bool = True,
) -> list[dict]:
    """Solve each puzzle for ground truth and evaluate traces. Returns per-puzzle stats."""
    all_stats = []
    for idx, (puzzle, trace) in enumerate(zip(puzzles, traces)):
        result = solve(puzzle)
        if result is None:
            raise ValueError(f"Solver failed: {puzzle[:20]}...")
        solution = result[0]
        if not quiet:
            print(f"\nPuzzle {idx + 1}/{len(puzzles)}:")
            print_grid(list(puzzle))
        stats = evaluate_puzzle(trace, puzzle, solution, verbose=not quiet)
        all_stats.append(stats)
    return all_stats


def evaluate_sequence(
    tokens_after_boundary: list[int],
    puzzle: str,
    solution: str,
) -> dict:
    """Replay token sequence with push/pop stack, check correctness after each fill.

    Works for both standard (no PUSH/POP) and BT (with PUSH/POP) sequences.
    For standard sequences the stack simulation is a no-op.

    Returns dict with: solved, cell_accuracy, n_correct, n_empties, unmatched_pops.
    """
    grid = list(puzzle)
    stack: list[list[str]] = []
    unmatched_pops = 0
    n_empties = sum(1 for ch in puzzle if ch not in "123456789")

    for tok in tokens_after_boundary:
        if tok == PUSH_TOKEN:
            stack.append(grid[:])
        elif tok == POP_TOKEN:
            if stack:
                grid = stack.pop()
            else:
                unmatched_pops += 1
        elif tok == SUCCESS_TOKEN:
            break
        elif 0 <= tok <= 728:
            r, c, d = decode_fill(tok)
            grid[r * 9 + c] = str(d)
            if "".join(grid) == solution:
                return {
                    "solved": True,
                    "cell_accuracy": 1.0,
                    "n_empties": n_empties,
                    "n_correct": n_empties,
                    "unmatched_pops": unmatched_pops,
                }

    n_correct = sum(
        1 for i in range(81)
        if puzzle[i] not in "123456789" and grid[i] == solution[i]
    )
    return {
        "solved": False,
        "cell_accuracy": n_correct / n_empties if n_empties > 0 else 1.0,
        "n_empties": n_empties,
        "n_correct": n_correct,
        "unmatched_pops": unmatched_pops,
    }


def evaluate_sequences(
    puzzles: list[str],
    sequences: list[list[int]],
    n_clues: np.ndarray,
    quiet: bool = True,
) -> list[dict]:
    """Evaluate generated sequences with stack simulation.

    Works for both standard and BT sequences. Solves each puzzle for ground
    truth internally.
    """
    all_stats = []
    for i, (puzzle, seq) in enumerate(zip(puzzles, sequences)):
        result = solve(puzzle)
        if result is None:
            raise ValueError(f"Solver failed: {puzzle[:20]}...")
        solution = result[0]

        nc = int(n_clues[i])
        start = nc + 1 if nc < len(seq) and seq[nc] == SEP_TOKEN else nc
        tokens_after = seq[start:]

        stats = evaluate_sequence(tokens_after, puzzle, solution)
        all_stats.append(stats)

        if not quiet:
            print(
                f"Puzzle {i+1}: solved={stats['solved']}  "
                f"acc={stats['cell_accuracy']:.1%}  "
                f"pops={stats['unmatched_pops']}"
            )
    return all_stats


def first_inconsistent_cell(
    trace: list[tuple[int, int, int]], puzzle: str,
) -> tuple[int, int, int] | None:
    """Replay trace on puzzle grid, return (row, col, step_idx) of first inconsistency, or None."""
    grid = list(puzzle)
    for step_idx, (r, c, d) in enumerate(trace):
        pos = r * 9 + c
        if grid[pos] not in ".0":
            # Overwriting a clue or previous fill — treat as inconsistency
            return (r, c, step_idx)
        if not _is_consistent(grid, r, c, d):
            return (r, c, step_idx)
        grid[pos] = str(d)
    return None


def plot_mistake_position_distribution(
    steps_from_end: list[int], output_path: str,
) -> None:
    """Plot histogram of how many steps from the end of the trace the first mistake occurs."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4))
    max_val = max(steps_from_end)
    bins = min(max_val + 1, 50)
    ax.hist(steps_from_end, bins=bins, edgecolor="black", linewidth=0.5)
    ax.set_xlabel("Steps from end of trace")
    ax.set_ylabel("Count")
    ax.set_title(f"First mistake position ({len(steps_from_end)} puzzles with errors)")
    ax.invert_xaxis()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved mistake position distribution to {output_path}")


def plot_first_mistake_heatmap(
    positions: list[tuple[int, int]], output_path: str,
) -> None:
    """Plot 9x9 heatmap of first-mistake positions and save to file."""
    import matplotlib.pyplot as plt

    counts = np.zeros((9, 9), dtype=int)
    for r, c in positions:
        counts[r, c] += 1

    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(counts, cmap="Reds", origin="upper")
    # Annotate cells
    for i in range(9):
        for j in range(9):
            v = counts[i, j]
            if v > 0:
                ax.text(j, i, str(v), ha="center", va="center", fontsize=9,
                        color="white" if v > counts.max() / 2 else "black")
    # Sudoku box lines
    for k in range(0, 10, 3):
        lw = 2
        ax.axhline(k - 0.5, color="black", linewidth=lw)
        ax.axvline(k - 0.5, color="black", linewidth=lw)
    ax.set_xticks(range(9))
    ax.set_yticks(range(9))
    ax.set_title(f"First mistake location ({len(positions)} puzzles with errors)")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved heatmap to {output_path}")


def summarize_stats(all_stats: list[dict]) -> str:
    """Format aggregate evaluation statistics as a string.

    Handles both old-style stats (from evaluate_puzzle) and new-style stats
    (from evaluate_sequence).
    """
    n = len(all_stats)
    avg_acc = np.mean([s["cell_accuracy"] for s in all_stats])
    # Support both key names: "solved" (new) and "puzzle_solved" (old)
    n_solved = sum(s.get("solved", s.get("puzzle_solved", False)) for s in all_stats)
    avg_correct = np.mean([s.get("n_correct", s.get("correct", 0)) for s in all_stats])
    avg_empties = np.mean([s["n_empties"] for s in all_stats])
    lines = [
        f"\n{'='*60}",
        f"Results on {n} puzzles:",
        f"  Cell accuracy:     {avg_acc:.1%} ({avg_correct:.1f}/{avg_empties:.1f} avg)",
        f"  Puzzles solved:    {n_solved}/{n} ({n_solved/n:.1%})",
    ]
    # Old-style detailed stats
    if "wrong_consistent" in all_stats[0]:
        total_wc = sum(s["wrong_consistent"] for s in all_stats)
        total_ic = sum(s["inconsistent"] for s in all_stats)
        total_oc = sum(s["overwrites_clue"] for s in all_stats)
        total_of = sum(s["overwrites_fill"] for s in all_stats)
        total_miss = sum(s["missing"] for s in all_stats)
        lines += [
            f"  Wrong consistent:  {total_wc}",
            f"  Inconsistent:      {total_ic}",
            f"  Clue overwrites:   {total_oc}",
            f"  Fill overwrites:   {total_of}",
            f"  Missing fills:     {total_miss}",
        ]
    # BT diagnostics
    total_pops = sum(s.get("unmatched_pops", 0) for s in all_stats)
    if total_pops > 0:
        lines.append(f"  Unmatched POPs:    {total_pops}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Evaluate model on Sudoku puzzles")
    parser.add_argument("--cache_path", required=True, help="Path to cached activations NPZ")
    parser.add_argument("--n", type=int, default=None, help="Number of puzzles to evaluate (default: all)")
    parser.add_argument("--quiet", action="store_true", help="Only show summary")
    parser.add_argument("--mistake-map", action="store_true", help="Plot first-mistake heatmap")
    parser.add_argument("--mistake-position", action="store_true", help="Plot first-mistake position distribution")
    parser.add_argument("--output", default=None, help="Output path for plots")
    args = parser.parse_args()

    from sudoku.activations import load_probe_dataset, derive_n_clues, sequences_to_traces

    _, puzzles, sequences, n_clues = load_probe_dataset(args.cache_path)
    if n_clues is None:
        n_clues = derive_n_clues(puzzles)

    if args.n is not None:
        puzzles = puzzles[:args.n]
        sequences = sequences[:args.n]
        n_clues = n_clues[:args.n]

    if args.mistake_map or args.mistake_position:
        traces = sequences_to_traces(sequences, n_clues)
        positions = []
        steps_from_end = []
        for puzzle, trace in zip(puzzles, traces):
            result = first_inconsistent_cell(trace, puzzle)
            if result is not None:
                r, c, step_idx = result
                positions.append((r, c))
                steps_from_end.append(len(trace) - step_idx)
        print(f"Found first mistakes in {len(positions)}/{len(puzzles)} puzzles")
        if not positions:
            print("No mistakes found — nothing to plot.")
            return
        if args.mistake_map:
            plot_first_mistake_heatmap(positions, args.output or "first_mistake_heatmap.png")
        if args.mistake_position:
            plot_mistake_position_distribution(steps_from_end, args.output or "first_mistake_position.png")
        return

    all_stats = evaluate_sequences(puzzles, sequences, n_clues, quiet=args.quiet)
    print(summarize_stats(all_stats))


if __name__ == "__main__":
    main()
