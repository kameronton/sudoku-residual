"""Evaluate all experiments from cached activations and collect summaries.

Usage:
    uv run python run_eval.py                        # run all
    uv run python run_eval.py --dry-run              # list what would run
    uv run python run_eval.py --filter no_sep        # subset by name
    uv run python run_eval.py --output results.txt   # custom output file
"""

import os
from datetime import datetime

from experiment_config import parse_batch_args, filter_experiments, experiment_dir


def main():
    opts = parse_batch_args()

    runs = filter_experiments(opts["filter"])
    if not runs:
        print("No matching experiments.")
        return

    if opts["dry_run"]:
        for name, _ in runs:
            exp_dir = experiment_dir(name)
            cache = f"{exp_dir}/activations.npz"
            exists = "OK" if os.path.exists(cache) else "MISSING"
            print(f"  {name}: {cache} [{exists}] -> {exp_dir}/eval.txt")
        return

    from data import solve
    from probes import load_probe_dataset, derive_n_clues
    from evaluate import evaluate_puzzle, summarize_stats, sequences_to_traces

    for i, (name, _) in enumerate(runs):
        exp_dir = experiment_dir(name)
        cache_path = f"{exp_dir}/activations.npz"
        eval_path = f"{exp_dir}/eval.txt"

        header = f"[{i+1}/{len(runs)}] {name}"
        print(f"\n{'='*60}\n{header}\n{'='*60}")

        if not os.path.exists(cache_path):
            print(f"  Skipping (missing: {cache_path})")
            continue

        print(f"  Loading {cache_path}...")
        _, puzzles, sequences, n_clues = load_probe_dataset(cache_path)
        if n_clues is None:
            n_clues = derive_n_clues(puzzles)

        traces = sequences_to_traces(sequences, n_clues)

        # Solve puzzles to get ground truth solutions
        print(f"  Solving {len(puzzles)} puzzles for ground truth...")
        solutions = []
        for p in puzzles:
            result = solve(p)
            if result is None:
                raise ValueError(f"Solver failed: {p[:20]}...")
            solutions.append(result[0])

        all_stats = []
        for puzzle, solution, trace in zip(puzzles, solutions, traces):
            stats = evaluate_puzzle(trace, puzzle, solution, verbose=False)
            all_stats.append(stats)

        summary = summarize_stats(all_stats)
        print(summary)

        with open(eval_path, "w") as f:
            f.write(f"Experiment: {name}\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
            f.write(summary + "\n")
        print(f"  Written to {eval_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
