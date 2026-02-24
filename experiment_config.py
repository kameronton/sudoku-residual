"""Shared experiment definitions and helpers for batch runner scripts.

All runner scripts (run_training.py, collect_activations.py, run_probes.py,
run_eval.py) import EXPERIMENTS and COMMON from here.
"""

import sys

# ── Shared defaults (override per-run as needed) ─────────────────────
COMMON = dict(
    traces_path="traces_unbiased.npz",
    dtype="bfloat16",
    batch_size=512,
    num_tokens=218_700_000,
    lr=0.001,
    eval_every=1000,
    num_checkpoints=1,
    n_layers=8,
    n_heads=8,
    d_model=576,
    d_ff=3456,
    schedule_type="linear",
    loss_mask="after_clues",
    weight_decay=0.1,
    warmup_tokens=1_000_000,
    seed=42,
)

# ── Experiment definitions ───────────────────────────────────────────
# Each entry: (name, {overrides})
# name is used for ckpt_dir and log_path automatically.
EXPERIMENTS = [
    ("baseline", {}),
    ("baseline_mask_all", {"loss_mask": "all"}),
    ("baseline_no_pos_emb", {"no_pos_emb": True}),
    ("baseline_mask_all_no_pos_emb", {"loss_mask": "all", "no_pos_emb": True}),
    ("no_sep", {"traces_path": "traces_unbiased_no_sep.npz"}),
    ("no_sep_no_pos_emb", {"traces_path": "traces_unbiased_no_sep.npz", "no_pos_emb": True}),
    ("no_sep_mask_all", {"traces_path": "traces_unbiased_no_sep.npz", "loss_mask": "all"}),
    ("no_sep_mask_all_no_pos_emb", {"traces_path": "traces_unbiased_no_sep.npz", "loss_mask": "all", "no_pos_emb": True}),
]


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
    result = {"dry_run": False, "filter": None, "_extra": {}}
    i = 0
    while i < len(argv):
        if argv[i] == "--dry-run":
            result["dry_run"] = True
            i += 1
        elif argv[i] == "--filter" and i + 1 < len(argv):
            result["filter"] = argv[i + 1]
            i += 2
        elif argv[i].startswith("--") and i + 1 < len(argv):
            key = argv[i].lstrip("-")
            result["_extra"][key] = argv[i + 1]
            i += 2
        else:
            i += 1
    return result


def filter_experiments(filt: str | None = None) -> list[tuple[str, dict]]:
    """Return experiments matching filter string (or all if None)."""
    return [(n, o) for n, o in EXPERIMENTS if filt is None or filt in n]
