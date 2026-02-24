"""Run probes on cached activations for all experiments.

Usage:
    uv run python run_probes.py                          # run all
    uv run python run_probes.py --dry-run                # list what would run
    uv run python run_probes.py --filter no_sep          # subset by name
    uv run python run_probes.py --mode candidates        # probe mode
    uv run python run_probes.py --step 5                 # probe at step 5
"""

import os

from experiment_config import parse_batch_args, filter_experiments


def main():
    opts = parse_batch_args()
    mode = opts["_extra"].get("mode", "state_filled")
    step = int(opts["_extra"].get("step", 0))

    runs = filter_experiments(opts["filter"])
    if not runs:
        print("No matching experiments.")
        return

    if opts["dry_run"]:
        for name, _ in runs:
            cache = f"activations/{name}.npz"
            exists = "OK" if os.path.exists(cache) else "MISSING"
            print(f"  {name}: {cache} [{exists}] -> probes/{name}_{mode}_step{step}.png")
        return

    os.makedirs("probes", exist_ok=True)

    # Import heavy deps only when actually running
    from data import SEP_TOKEN
    from probes import (
        load_probe_dataset, derive_n_clues, run_probe_loop,
        plot_all_layers, anchor_positions, build_grid_at_step,
    )
    from evaluate import sequences_to_traces

    for i, (name, _) in enumerate(runs):
        cache_path = f"activations/{name}.npz"
        output_path = f"probes/{name}_{mode}_step{step}.png"

        header = f"[{i+1}/{len(runs)}] {name}"
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

        # Detect anchor mode: use "sep" if sequences contain SEP token, else "last_clue"
        has_sep = any(SEP_TOKEN in seq for seq in sequences[:10])
        anchor = "sep" if has_sep else "last_clue"
        print(f"  Anchor mode: {anchor}")

        traces = sequences_to_traces(sequences, n_clues)
        anchor_pos = anchor_positions(n_clues, anchor)

        # Filter puzzles with enough trace steps
        keep = [j for j, (t, ap) in enumerate(zip(traces, anchor_pos))
                if len(t) >= step and ap + step >= 0]
        if len(keep) < len(puzzles):
            print(f"  Filtered to {len(keep)}/{len(puzzles)} puzzles")
            activations = activations[keep]
            puzzles = [puzzles[j] for j in keep]
            sequences = [sequences[j] for j in keep]
            n_clues = n_clues[keep]
            anchor_pos = [anchor_pos[j] for j in keep]

        if not puzzles:
            print("  No puzzles remaining after filtering.")
            continue

        probe_positions = [ap + step for ap in anchor_pos]

        if step == 0 and anchor == "sep":
            probe_grids = puzzles
        else:
            probe_grids = [build_grid_at_step(seq, pos)
                           for seq, pos in zip(sequences, probe_positions)]

        print(f"  Running probes ({mode}, step={step}, {len(puzzles)} puzzles)...")
        all_accuracies, all_per_digit = run_probe_loop(
            activations, probe_grids, probe_positions, mode=mode,
        )

        metric = "F1" if mode == "candidates" else "Accuracy"
        if mode == "candidates" and all_per_digit:
            from probes import plot_all_layers_per_digit
            plot_all_layers_per_digit(all_per_digit, output_path, show=False)
        else:
            plot_all_layers(all_accuracies, output_path, metric_name=metric, show=False)

    print(f"\nDone. Results in probes/")


if __name__ == "__main__":
    main()
