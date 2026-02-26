"""Default experiment definitions — committed baseline.

To override without modifying this file, create experiments_local.py
(git-ignored) in the same directory. See README or CLAUDE.md for the
expected format.
"""

COMMON = dict(
    traces_path="traces_unbiased_no_sep.npz",
    dtype="bfloat16",
    batch_size=512,
    num_epochs=1,
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

EXPERIMENTS = [
    ("baseline",                          {}),
]
