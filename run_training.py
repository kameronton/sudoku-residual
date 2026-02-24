"""Run training for all experiments sequentially.

Usage:
    uv run python run_training.py              # run all pending experiments
    uv run python run_training.py --dry-run    # print commands without running
    uv run python run_training.py --filter L12 # only runs whose name contains "L12"
"""

import subprocess
import sys

from experiment_config import COMMON, parse_batch_args, filter_experiments, experiment_dir


def build_args(name: str, overrides: dict) -> list[str]:
    cfg = {**COMMON, **overrides}
    exp_dir = experiment_dir(name)
    cfg["ckpt_dir"] = overrides.get("ckpt_dir", f"{exp_dir}/checkpoint")
    cfg["log_path"] = overrides.get("log_path", f"{exp_dir}/train_log.json")
    args = ["uv", "run", "python", "training.py"]
    for k, v in cfg.items():
        if isinstance(v, bool):
            if v:
                args.append(f"--{k}")
        else:
            args += [f"--{k}", str(v)]
    return args


def main():
    opts = parse_batch_args()
    runs = filter_experiments(opts["filter"])
    if not runs:
        print("No matching experiments.")
        return

    for i, (name, overrides) in enumerate(runs):
        args = build_args(name, overrides)
        header = f"[{i+1}/{len(runs)}] {name}"
        if overrides:
            header += f"  {overrides}"
        print(f"\n{'='*60}\n{header}\n{'='*60}")
        if opts["dry_run"]:
            print(" ".join(args))
        else:
            result = subprocess.run(args)
            if result.returncode != 0:
                print(f"FAILED: {name} (exit {result.returncode})")
                sys.exit(1)

    print(f"\nAll {len(runs)} experiments finished.")


if __name__ == "__main__":
    main()
