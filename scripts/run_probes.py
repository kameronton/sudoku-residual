"""Run probes on cached activations for all experiments.

Usage:
    uv run python run_probes.py                          # run all
    uv run python run_probes.py --dry-run                # list what would run
    uv run python run_probes.py --filter no_sep          # subset by name
    uv run python run_probes.py --name baseline          # single experiment
    uv run python run_probes.py --name baseline --all-steps  # all checkpoints
    uv run python run_probes.py --mode candidates        # probe mode
    uv run python run_probes.py --step 5                 # probe at step 5
"""

import os

from sudoku.experiment_config import parse_batch_args, resolve_runs


def main():
    opts = parse_batch_args()
    mode = opts["_extra"].get("mode", "state_filled")
    step = int(opts["_extra"].get("step", 0))
    per_digit = opts["_extra"].get("per-digit", False)
    use_deltas = opts["_extra"].get("use-deltas", False)

    runs = resolve_runs(opts)
    if not runs:
        print("No matching experiments.")
        return

    if opts["dry_run"]:
        for name, _, ckpt_step, output_dir in runs:
            cache = f"{output_dir}/activations.npz"
            exists = "OK" if os.path.exists(cache) else "MISSING"
            step_info = f" (step {ckpt_step})" if ckpt_step is not None else ""
            print(f"  {name}{step_info}: {cache} [{exists}] -> {output_dir}/probe_{mode}_step{step}.png")
        return

    # Import heavy deps only when actually running
    from sudoku.activations import load_probe_dataset, derive_n_clues
    from sudoku.probes import (
        prepare_probe_inputs, run_probe_loop, run_structure_probe_loop, compute_deltas,
        plot_all_layers, plot_all_layers_per_digit, plot_structure, metric_name_for_mode,
    )

    for i, (name, _, ckpt_step, output_dir) in enumerate(runs):
        cache_path = f"{output_dir}/activations.npz"
        output_path = f"{output_dir}/probe_{mode}_step{step}.png"
        if use_deltas:
            output_path = output_path.replace(".png", "_deltas.png")

        step_info = f" step {ckpt_step}" if ckpt_step is not None else ""
        header = f"[{i+1}/{len(runs)}] {name}{step_info}"
        print(f"\n{'='*60}\n{header}\n{'='*60}")

        if not os.path.exists(cache_path):
            print(f"  Skipping (missing: {cache_path})")
            continue

        if os.path.exists(output_path):
            print(f"  Skipping (already exists: {output_path})")
            continue

        print(f"  Loading {cache_path}...")
        activations, puzzles, sequences, n_clues = load_probe_dataset(cache_path)
        if n_clues is None:
            n_clues = derive_n_clues(puzzles)

        activations, probe_grids, probe_positions = prepare_probe_inputs(
            activations, puzzles, sequences, n_clues, step,
        )

        if not probe_grids:
            print("  No puzzles remaining after filtering.")
            continue

        if use_deltas:
            activations = compute_deltas(activations)

        print(f"  Running probes ({mode}, step={step}, {len(probe_grids)} puzzles)...")
        if mode == "structure":
            all_scores = run_structure_probe_loop(activations, probe_grids, probe_positions)
            plot_structure(all_scores, output_path, show=False)
        else:
            all_accuracies, all_per_digit = run_probe_loop(
                activations, probe_grids, probe_positions, mode=mode,
            )
            metric = metric_name_for_mode(mode)
            if per_digit and all_per_digit:
                plot_all_layers_per_digit(all_per_digit, output_path.replace(".png", "_per_digit.png"), show=False)
            else:
                plot_all_layers(all_accuracies, output_path, metric_name=metric, show=False)

    print(f"\nDone.")


if __name__ == "__main__":
    main()
