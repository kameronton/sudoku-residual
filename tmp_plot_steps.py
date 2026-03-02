"""Temporary script: plot cell/grid accuracy vs epoch for both experiments."""

import json
import re
import os
import warnings
import matplotlib.pyplot as plt
import numpy as np
import optax
from scipy.optimize import curve_fit

experiments = {

    "3M-lr-cosine": {"color": "C0", "label": "cosine: 1e-3 -> 1e-5"},
    "3M-lr1e3":     {"color": "C1", "label": "linear: 1e-3 -> 1e-5"},
    "3M-lr5e4":     {"color": "C2", "label": "linear: 5e-4 -> 5e-6"},

}
steps_per_epoch = 41472   # n_train // batch_size, ~2.7M / 512

# Extrapolate to this many tokens (3x current max as a reasonable horizon).
EXTRAP_FACTOR = 3


def _power_sat(x, L, a, b):
    """Saturating power law:  y = L - a * x^{-b}

    Standard neural scaling-law form (Kaplan et al., Hoffmann et al.).
    Asymptote L is the ceiling; b ~ 0.3-0.8 for typical learning curves.
    Preferred over exponential saturation because training curves are
    approximately linear in log-log space.
    """
    return L - a * x ** (-b)


def fit_and_extrapolate(tokens_arr, values_arr, extrap_max, color, ax,
                        marker_style, is_dashed):
    """Fit saturating power law and plot extrapolation beyond last data point."""
    x = np.asarray(tokens_arr, dtype=float)
    y = np.asarray(values_arr, dtype=float)

    # Initial guess: ceiling slightly above last observed value.
    L0 = min(y[-1] * 1.15 + 1.0, 99.0)
    # a0 such that the model passes near the first point: L0 - a0 * x[0]^{-b0} ≈ y[0]
    b0 = 0.5
    a0 = max((L0 - y[0]) * x[0] ** b0, 1e-3)

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            popt, _ = curve_fit(
                _power_sat, x, y,
                p0=[L0, a0, b0],
                bounds=([y[-1], 1e-6, 0.05], [100.0, np.inf, 3.0]),
                maxfev=20000,
            )
        L, a, b = popt
    except Exception as e:
        print(f"  fit failed ({e}); skipping extrapolation")
        return

    linestyle = "--" if is_dashed else "-"

    # Extrapolation region only (from last observed point onward).
    x_ext = np.linspace(x[-1], extrap_max, 200)
    y_ext = _power_sat(x_ext, *popt)
    y_ext = np.clip(y_ext, 0, 100)
    ax.plot(x_ext, y_ext, linestyle=linestyle, color=color, alpha=0.45,
            linewidth=1.4)

    # Annotate the asymptote value right at the end of the extrapolated line.
    ax.text(
        extrap_max, y_ext[-1],
        f" →{L:.0f}%",
        fontsize=6.5,
        color=color,
        alpha=0.7,
        va="center",
    )
    print(f"    asymptote L={L:.1f}%  b={b:.3f}  a={a:.3e}")


def _load_exp_config(exp_name):
    """Return (train_cfg, tokens_per_step) from config.json + train_log.json, or None."""
    config_path = f"results/{exp_name}/config.json"
    log_path    = f"results/{exp_name}/train_log.json"
    if not os.path.isfile(config_path) or not os.path.isfile(log_path):
        return None, None
    with open(config_path) as f:
        train_cfg = json.load(f)
    with open(log_path) as f:
        log = json.load(f)
    return train_cfg, log.get("tokens_per_step")


def _make_schedule(train_cfg, total_steps, warmup_steps):
    schedule_type = train_cfg.get("schedule_type", "linear")
    schedule_frac = train_cfg.get("schedule_frac", 1.0)
    schedule_steps = max(warmup_steps + 1, round(total_steps * schedule_frac))
    lr = train_cfg["lr"]
    if schedule_type == "cosine":
        return optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=lr,
            warmup_steps=warmup_steps,
            decay_steps=schedule_steps,
            end_value=lr * 0.1,
        )
    else:
        return optax.linear_onecycle_schedule(
            transition_steps=schedule_steps,
            peak_value=lr,
            pct_start=warmup_steps / max(schedule_steps, 1),
            div_factor=10.0,
            final_div_factor=100.0,
        )


def plot_lr_schedule(exp_name, train_cfg, tokens_per_step, color, ax_lr):
    num_tokens      = train_cfg["num_tokens"]
    num_checkpoints = train_cfg.get("num_checkpoints", 0)
    warmup_tokens   = train_cfg.get("warmup_tokens", 1_000_000)

    total_steps_ideal = num_tokens // tokens_per_step
    if num_checkpoints > 0:
        ckpt_every  = max(1, total_steps_ideal // num_checkpoints)
        total_steps = round(total_steps_ideal / ckpt_every) * ckpt_every
    else:
        total_steps = total_steps_ideal
    warmup_steps = warmup_tokens // tokens_per_step

    schedule = _make_schedule(train_cfg, total_steps, warmup_steps)
    steps    = np.arange(total_steps + 1)
    lrs      = np.array([schedule(s) for s in steps])
    tokens   = steps * tokens_per_step

    ax_lr.plot(tokens, lrs, color=color, linewidth=1.5)


fig, (ax, ax_lr) = plt.subplots(2, 1, figsize=(10, 9), sharex=False)
fig.subplots_adjust(hspace=0.35)

all_max_tokens = []

for exp_name, cfg in experiments.items():
    steps_dir = f"results/{exp_name}/steps"
    if not os.path.isdir(steps_dir):
        print(f"Skipping {exp_name}: no steps dir")
        continue

    data = []
    for step_name in sorted(os.listdir(steps_dir), key=int):
        eval_path = f"{steps_dir}/{step_name}/eval.txt"
        if not os.path.isfile(eval_path):
            continue
        text = open(eval_path).read()
        cell_match = re.search(r"Cell accuracy:\s+([\d.]+)%", text)
        grid_match = re.search(r"Puzzles solved:\s+\d+/\d+ \(([\d.]+)%\)", text)
        if cell_match and grid_match:
            step = int(step_name)
            tokens = step * steps_per_epoch
            data.append((tokens, float(cell_match.group(1)), float(grid_match.group(1))))

    if not data:
        print(f"Skipping {exp_name}: no data")
        continue

    tokens, cell_acc, grid_acc = zip(*data)
    all_max_tokens.append(max(tokens))
    ax.plot(tokens, cell_acc, "o-", color=cfg["color"], label=f"{cfg['label']} (cell)")
    ax.plot(tokens, grid_acc, "s--", color=cfg["color"], label=f"{cfg['label']} (grid)")

    train_cfg, tokens_per_step = _load_exp_config(exp_name)
    if train_cfg is not None and tokens_per_step:
        plot_lr_schedule(exp_name, train_cfg, tokens_per_step, cfg["color"], ax_lr)

    if len(data) >= 4:
        extrap_max = max(tokens) * EXTRAP_FACTOR
        print(f"{exp_name} cell:")
        # fit_and_extrapolate(tokens, cell_acc, extrap_max, cfg["color"], ax,
                            # marker_style="o", is_dashed=False)
        print(f"{exp_name} grid:")
        # fit_and_extrapolate(tokens, grid_acc, extrap_max, cfg["color"], ax,
                            # marker_style="s", is_dashed=True)

ax.set_xlabel("Tokens")
ax.set_ylabel("Accuracy (%)")
ax.set_yticks(np.arange(0, 100, 5))
ax.set_title("Accuracy over training  (faded lines: power-law extrapolation)")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
ax.set_ylim(0, 100)

ax_lr.set_xlabel("Tokens")
ax_lr.set_ylabel("Learning rate")
ax_lr.set_title("LR schedule per experiment")
ax_lr.grid(True, alpha=0.3)
# Add per-experiment legend entries (label taken from experiments dict)
for exp_name, cfg in experiments.items():
    ax_lr.plot([], [], color=cfg["color"], label=cfg["label"])
ax_lr.legend(fontsize=8)

out = "results/accuracy_vs_epoch.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved to {out}")
