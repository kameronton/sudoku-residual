"""Evaluate model traces against ground-truth Sudoku solutions."""

import argparse

import numpy as np

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
    """Format aggregate evaluation statistics as a string."""
    n = len(all_stats)
    avg_acc = np.mean([s["cell_accuracy"] for s in all_stats])
    n_solved = sum(s["puzzle_solved"] for s in all_stats)
    avg_correct = np.mean([s["correct"] for s in all_stats])
    avg_empties = np.mean([s["n_empties"] for s in all_stats])
    total_wc = sum(s["wrong_consistent"] for s in all_stats)
    total_ic = sum(s["inconsistent"] for s in all_stats)
    total_oc = sum(s["overwrites_clue"] for s in all_stats)
    total_of = sum(s["overwrites_fill"] for s in all_stats)
    total_miss = sum(s["missing"] for s in all_stats)
    lines = [
        f"\n{'='*60}",
        f"Results on {n} puzzles:",
        f"  Cell accuracy:     {avg_acc:.1%} ({avg_correct:.1f}/{avg_empties:.1f} avg)",
        f"  Puzzles solved:    {n_solved}/{n} ({n_solved/n:.1%})",
        f"  Wrong consistent:  {total_wc}",
        f"  Inconsistent:      {total_ic}",
        f"  Clue overwrites:   {total_oc}",
        f"  Fill overwrites:   {total_of}",
        f"  Missing fills:     {total_miss}",
    ]
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

    from activations import load_probe_dataset, derive_n_clues, sequences_to_traces

    _, puzzles, sequences, n_clues = load_probe_dataset(args.cache_path)
    if n_clues is None:
        n_clues = derive_n_clues(puzzles)
    traces = sequences_to_traces(sequences, n_clues)

    if args.n is not None:
        puzzles = puzzles[:args.n]
        traces = traces[:args.n]

    if args.mistake_map or args.mistake_position:
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

    all_stats = evaluate_traces(puzzles, traces, quiet=args.quiet)
    print(summarize_stats(all_stats))


if __name__ == "__main__":
    main()
