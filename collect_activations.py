"""Collect activations for all experiments defined in experiment_config.py.

Usage:
    uv run python collect_activations.py                    # run all
    uv run python collect_activations.py --dry-run          # print commands
    uv run python collect_activations.py --filter no_sep    # subset by name
    uv run python collect_activations.py --n_puzzles 1000   # override puzzle count
"""

import os

from experiment_config import COMMON, parse_batch_args, filter_experiments, experiment_dir

DEFAULT_N_PUZZLES = 6400
DEFAULT_BATCH_SIZE = 64


def main():
    opts = parse_batch_args()
    n_puzzles = int(opts["_extra"].get("n_puzzles", DEFAULT_N_PUZZLES))
    batch_size = int(opts["_extra"].get("batch_size", DEFAULT_BATCH_SIZE))

    runs = filter_experiments(opts["filter"])
    if not runs:
        print("No matching experiments.")
        return

    if opts["dry_run"]:
        for name, overrides in runs:
            exp_dir = experiment_dir(name)
            ckpt_dir = overrides.get("ckpt_dir", f"{exp_dir}/checkpoint")
            traces_path = overrides.get("traces_path", COMMON["traces_path"])
            cache_path = f"{exp_dir}/activations.npz"
            print(f"{name}: ckpt_dir={ckpt_dir} traces={traces_path} -> {cache_path}")
        return

    # Import here so --dry-run works without JAX installed
    from probes import generate_probe_dataset

    for i, (name, overrides) in enumerate(runs):
        exp_dir = experiment_dir(name)
        ckpt_dir = overrides.get("ckpt_dir", f"{exp_dir}/checkpoint")
        traces_path = overrides.get("traces_path", COMMON["traces_path"])
        cache_path = f"{exp_dir}/activations.npz"
        os.makedirs(exp_dir, exist_ok=True)

        header = f"[{i+1}/{len(runs)}] {name}"
        print(f"\n{'='*60}\n{header}\n{'='*60}")

        if os.path.exists(cache_path):
            print(f"  Skipping (already exists: {cache_path})")
            continue

        generate_probe_dataset(
            ckpt_dir=ckpt_dir,
            traces_path=traces_path,
            n_puzzles=n_puzzles,
            batch_size=batch_size,
            cache_path=cache_path,
            compress=False,
        )

    print(f"\nAll {len(runs)} activations collected.")


if __name__ == "__main__":
    main()
