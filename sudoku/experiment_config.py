"""Shared experiment definitions and helpers for batch runner scripts.

All runner scripts (run_training.py, collect_activations.py, run_probes.py,
run_eval.py) import EXPERIMENTS and COMMON from here.
"""

import os
import sys

# ── Experiment definitions ───────────────────────────────────────────
# Load from experiments_local.py if present (git-ignored, for Colab/local overrides),
# otherwise fall back to the committed defaults in default_experiments.py.
try:
    from experiments_local import COMMON, EXPERIMENTS
    print("experiment_config: loaded from experiments_local.py", flush=True)
except ImportError:
    from default_experiments import COMMON, EXPERIMENTS


RESULTS_DIR = "results"


def experiment_dir(name: str) -> str:
    """Return the output directory for a given experiment."""
    return f"{RESULTS_DIR}/{name}"


def parse_batch_args(argv: list[str] | None = None) -> dict:
    """Parse common batch runner flags: --dry-run, --filter, and extra key=value pairs.

    Returns dict with keys: dry_run (bool), filter (str|None), plus any
    extra flags parsed by the caller.
    """
    if argv is None:
        argv = sys.argv[1:]
    result = {"dry_run": False, "filter": None, "name": None, "all_steps": False, "traces_only": False, "_extra": {}}
    i = 0
    while i < len(argv):
        if argv[i] == "--dry-run":
            result["dry_run"] = True
            i += 1
        elif argv[i] == "--all-steps":
            result["all_steps"] = True
            i += 1
        elif argv[i] == "--traces-only":
            result["traces_only"] = True
            i += 1
        elif argv[i] == "--filter" and i + 1 < len(argv):
            result["filter"] = argv[i + 1]
            i += 2
        elif argv[i] == "--name" and i + 1 < len(argv):
            result["name"] = argv[i + 1]
            i += 2
        elif argv[i].startswith("--") and i + 1 < len(argv) and not argv[i + 1].startswith("--"):
            key = argv[i].lstrip("-")
            result["_extra"][key] = argv[i + 1]
            i += 2
        elif argv[i].startswith("--"):
            key = argv[i].lstrip("-")
            result["_extra"][key] = True
            i += 1
        else:
            i += 1
    return result


def filter_experiments(filt: str | None = None, name: str | None = None) -> list[tuple[str, dict]]:
    """Return experiments from EXPERIMENTS matching filter (substring) or name (exact).

    Used by run_training.py to decide what to train.
    """
    if name is not None:
        return [(n, o) for n, o in EXPERIMENTS if n == name]
    return [(n, o) for n, o in EXPERIMENTS if filt is None or filt in n]


def discover_experiments(filt: str | None = None, name: str | None = None) -> list[tuple[str, dict]]:
    """Discover experiments from results/*/config.json on disk.

    Returns list of (name, config_dict) sorted by name.
    Used by downstream scripts (collect_activations, run_probes, run_eval).
    """
    import json
    experiments = []
    if not os.path.isdir(RESULTS_DIR):
        return []
    for entry in sorted(os.listdir(RESULTS_DIR)):
        config_path = f"{RESULTS_DIR}/{entry}/config.json"
        if not os.path.isfile(config_path):
            continue
        if name is not None and entry != name:
            continue
        if filt is not None and filt not in entry:
            continue
        with open(config_path) as f:
            cfg = json.load(f)
        experiments.append((entry, cfg))
    return experiments


def list_checkpoint_steps(name: str) -> list[int]:
    """Return sorted list of all checkpoint steps for an experiment."""
    ckpt_dir = f"{experiment_dir(name)}/checkpoint"
    if not os.path.isdir(ckpt_dir):
        return []
    import orbax.checkpoint as ocp
    mgr = ocp.CheckpointManager(os.path.abspath(ckpt_dir))
    return sorted(mgr.all_steps())


def resolve_runs(opts: dict) -> list[tuple[str, dict, int | None, str]]:
    """Resolve (name, config, ckpt_step, output_dir) for each run.

    Discovers experiments from results/*/config.json on disk.
    Default mode: one run per experiment at latest checkpoint.
    --all-steps mode: one run per checkpoint step for a single --name experiment.
    """
    if opts["all_steps"]:
        if opts["name"] is None:
            print("Error: --all-steps requires --name")
            sys.exit(1)
        experiments = discover_experiments(name=opts["name"])
        if not experiments:
            print(f"No config.json found for {opts['name']}")
            return []
        name, cfg = experiments[0]
        steps = list_checkpoint_steps(name)
        if not steps:
            print(f"No checkpoints found for {name}")
            return []
        exp_dir = experiment_dir(name)
        return [(name, cfg, step, f"{exp_dir}/steps/{step}") for step in steps]
    else:
        experiments = discover_experiments(opts["filter"], opts["name"])
        return [(n, cfg, None, experiment_dir(n)) for n, cfg in experiments]
