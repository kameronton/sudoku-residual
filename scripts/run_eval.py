"""Evaluate all experiments from cached activations and collect summaries.

Usage:
    uv run python run_eval.py                        # run all
    uv run python run_eval.py --dry-run              # list what would run
    uv run python run_eval.py --filter no_sep        # subset by name
    uv run python run_eval.py --name baseline        # single experiment
    uv run python run_eval.py --name baseline --all-steps  # all checkpoints
"""

import os
from datetime import datetime

from sudoku.experiment_config import parse_batch_args, resolve_runs


def main():
    opts = parse_batch_args()

    runs = resolve_runs(opts)
    if not runs:
        print("No matching experiments.")
        return

    if opts["dry_run"]:
        for name, _, ckpt_step, output_dir in runs:
            cache = f"{output_dir}/activations.npz"
            exists = "OK" if os.path.exists(cache) else "MISSING"
            step_info = f" (step {ckpt_step})" if ckpt_step is not None else ""
            print(f"  {name}{step_info}: {cache} [{exists}] -> {output_dir}/eval.txt")
        return

    from sudoku.activations import load_probe_dataset, derive_n_clues
    from sudoku.evaluate import evaluate_sequences, summarize_stats

    for i, (name, _, ckpt_step, output_dir) in enumerate(runs):
        cache_path = f"{output_dir}/activations.npz"
        eval_path = f"{output_dir}/eval.txt"

        step_info = f" step {ckpt_step}" if ckpt_step is not None else ""
        header = f"[{i+1}/{len(runs)}] {name}{step_info}"
        print(f"\n{'='*60}\n{header}\n{'='*60}")

        if not os.path.exists(cache_path):
            print(f"  Skipping (missing: {cache_path})")
            continue

        print(f"  Loading {cache_path}...")
        _, puzzles, sequences, n_clues = load_probe_dataset(cache_path)
        if n_clues is None:
            n_clues = derive_n_clues(puzzles)

        all_stats = evaluate_sequences(puzzles, sequences, n_clues)

        summary = summarize_stats(all_stats)
        print(summary)

        with open(eval_path, "w") as f:
            f.write(f"Experiment: {name}\n")
            if ckpt_step is not None:
                f.write(f"Checkpoint step: {ckpt_step}\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
            f.write(summary + "\n")
        print(f"  Written to {eval_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
