"""Collect activations for experiments discovered from results/*/config.json.

Usage:
    uv run python collect_activations.py                             # run all
    uv run python collect_activations.py --dry-run                   # print what would run
    uv run python collect_activations.py --filter no_sep             # subset by name
    uv run python collect_activations.py --name baseline             # single experiment
    uv run python collect_activations.py --name baseline --all-steps # all checkpoints
    uv run python collect_activations.py --n_puzzles 1000            # override puzzle count
    uv run python collect_activations.py --traces-only               # skip activation collection
"""

import os

from sudoku.experiment_config import parse_batch_args, resolve_runs

DEFAULT_N_PUZZLES = 6400
DEFAULT_BATCH_SIZE = 64


def main():
    opts = parse_batch_args()
    n_puzzles = int(opts["_extra"].get("n_puzzles", DEFAULT_N_PUZZLES))
    batch_size = int(opts["_extra"].get("batch_size", DEFAULT_BATCH_SIZE))
    traces_only = opts["traces_only"]

    runs = resolve_runs(opts)
    if not runs:
        print("No matching experiments.")
        return

    if opts["dry_run"]:
        for name, cfg, ckpt_step, output_dir in runs:
            step_info = f" (step {ckpt_step})" if ckpt_step is not None else ""
            print(f"{name}{step_info}: ckpt_dir={cfg['ckpt_dir']} traces={cfg['traces_path']} -> {output_dir}/activations.npz")
        return

    # Import here so --dry-run works without JAX installed
    from sudoku.activations import generate_probe_dataset

    for i, (name, cfg, ckpt_step, output_dir) in enumerate(runs):
        cache_path = f"{output_dir}/activations.npz"
        os.makedirs(output_dir, exist_ok=True)

        step_info = f" step {ckpt_step}" if ckpt_step is not None else ""
        header = f"[{i+1}/{len(runs)}] {name}{step_info}"
        print(f"\n{'='*60}\n{header}\n{'='*60}")

        if os.path.exists(cache_path):
            import numpy as _np
            if "activations" in _np.load(cache_path, allow_pickle=False).files:
                print(f"  Skipping (activations already exist: {cache_path})")
                continue
            # File exists but contains only traces — fall through to collect activations

        generate_probe_dataset(
            ckpt_dir=cfg["ckpt_dir"],
            traces_path=cfg["traces_path"],
            n_puzzles=n_puzzles,
            batch_size=batch_size,
            cache_path=cache_path,
            compress=False,
            ckpt_step=ckpt_step,
            traces_only=traces_only,
        )

    print(f"\nAll {len(runs)} activations collected.")


if __name__ == "__main__":
    main()
