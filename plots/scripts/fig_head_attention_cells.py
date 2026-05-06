"""Figure: mean attention from [SEP] per cell, one 9×9 grid per head (layers × heads).

Usage:
    uv run python plots/scripts/fig_head_attention_cells.py                         # compute + save + plot
    uv run python plots/scripts/fig_head_attention_cells.py --data plots/data/fig_head_attention_cells.npz  # plot only
    uv run python plots/scripts/fig_head_attention_cells.py --cache results/3M-backtracking-packing/activations.npz
    uv run python plots/scripts/fig_head_attention_cells.py --step 1 --n-puzzles 3200
"""

import argparse
import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from sudoku.activations import load_probe_dataset, load_checkpoint, derive_n_clues
from sudoku.data import PAD_TOKEN, SEP_TOKEN

sns.set_theme(style="ticks", font="Avenir", context="paper")

DEFAULT_CACHE = "results/3M-backtracking-packing/activations.npz"
OUTPUT        = "plots/figures/fig_head_attention_cells.pdf"
DATA_PATH     = "plots/data/fig_head_attention_cells.npz"


def compute_data(args) -> dict:
    _, puzzles, sequences, n_clues, _ = load_probe_dataset(args.cache)
    if n_clues is None:
        n_clues = derive_n_clues(puzzles)

    ckpt_dir = args.cache.replace("activations.npz", "checkpoint")
    params, model = load_checkpoint(ckpt_dir)

    cfg      = model.config
    n_layers = cfg.n_layers
    n_heads  = cfg.n_heads
    T        = cfg.max_seq_len

    @jax.jit
    def get_attn_weights(tokens):
        _, intermediates = model.apply({"params": params}, tokens, return_intermediates=True)
        return jnp.stack(
            [intermediates[f"layer_{i}_attn_weights"] for i in range(n_layers)],
            axis=1,
        )

    acc_cell     = np.zeros((n_layers, n_heads, 81), dtype=np.float64)
    acc_cell_cnt = np.zeros(81, dtype=np.int64)
    acc_sep      = np.zeros((n_layers, n_heads), dtype=np.float64)
    acc_self     = np.zeros((n_layers, n_heads), dtype=np.float64)
    n_valid      = 0

    n_total = min(args.n_puzzles, len(sequences)) if args.n_puzzles else len(sequences)
    query_label = "SEP" if args.step == 0 else f"SEP+{args.step}"

    for batch_start in tqdm(range(0, n_total, args.batch_size), desc=f"{query_label} attention"):
        batch_end = min(batch_start + args.batch_size, n_total)
        B         = batch_end - batch_start

        tokens_np = np.full((B, T), PAD_TOKEN, dtype=np.int32)
        for i, s in enumerate(sequences[batch_start:batch_end]):
            L = min(len(s), T)
            tokens_np[i, :L] = s[:L]

        attn_w = np.array(get_attn_weights(jnp.array(tokens_np)))  # (B, n_layers, n_heads, T, T)

        for b in range(B):
            toks = tokens_np[b]
            nc   = int(n_clues[batch_start + b])

            if nc >= T or toks[nc] != SEP_TOKEN:
                continue
            query_pos = nc + args.step
            if query_pos >= T or toks[query_pos] == PAD_TOKEN:
                continue
            n_valid += 1

            attn_q = attn_w[b, :, :, query_pos, :]   # (n_layers, n_heads, T)

            acc_sep  += attn_q[:, :, nc]
            acc_self += attn_q[:, :, query_pos]

            # Fill tokens causally available (positions 0..query_pos-1, excluding PAD/SEP)
            key_pos  = np.arange(query_pos, dtype=np.int32)
            is_fill  = (toks[key_pos] >= 0) & (toks[key_pos] <= 728)
            fill_pos = key_pos[is_fill]
            if len(fill_pos) == 0:
                continue

            key_sum   = attn_q[:, :, key_pos].sum(-1, keepdims=True)
            fill_attn = attn_q[:, :, fill_pos] / np.maximum(key_sum, 1e-12)

            t32      = toks[fill_pos].astype(np.int32)
            rows     = t32 // 81
            cols     = (t32 % 81) // 9
            cell_idx = rows * 9 + cols

            acc_cell[:, :, cell_idx] += fill_attn
            acc_cell_cnt[cell_idx]   += 1

    print(f"Used {n_valid}/{n_total} puzzles (query = {query_label})")

    mean_cell = acc_cell / np.maximum(acc_cell_cnt, 1)[None, None, :]
    mean_cell[:, :, acc_cell_cnt == 0] = np.nan
    mean_sep  = acc_sep  / max(n_valid, 1)
    mean_self = acc_self / max(n_valid, 1)

    return dict(
        cell_grid=mean_cell.reshape(n_layers, n_heads, 9, 9),
        mean_sep=mean_sep,
        mean_self=mean_self,
        n_valid=n_valid,
        n_puzzles=n_total,
        step=args.step,
    )


def plot(data: dict):
    cell_grid = data["cell_grid"]                   # (n_layers, n_heads, 9, 9)
    mean_sep  = data["mean_sep"]                    # (n_layers, n_heads)
    mean_self = data["mean_self"]
    n_valid   = int(data["n_valid"])
    step      = int(data["step"])
    n_layers, n_heads = cell_grid.shape[:2]

    query_label = "SEP" if step == 0 else f"SEP+{step}"

    cmap = plt.get_cmap("YlOrRd").copy()
    cmap.set_bad(color="lightgrey")

    fig, axes = plt.subplots(n_layers, n_heads,
                             figsize=(n_heads * 1.5, n_layers * 1.3),
                             squeeze=False)

    for li in range(n_layers):
        for hi in range(n_heads):
            ax   = axes[li][hi]
            g    = cell_grid[li, hi]
            vmax = np.nanmax(g) if not np.all(np.isnan(g)) else 1.0
            ax.imshow(g, vmin=0, vmax=vmax, cmap=cmap, interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])

            for v in [2.5, 5.5]:
                ax.axvline(v, color="black", linewidth=0.6)
                ax.axhline(v, color="black", linewidth=0.6)

            if li == 0:
                ax.set_title(f"H{hi+1}", fontsize=7)
            if hi == 0:
                ax.set_ylabel(f"L{li+1}", fontsize=7)

            s_pct = mean_sep[li, hi]
            ax.text(0.5, -0.1, f"END_CLUES: {s_pct:.0%}",
                    transform=ax.transAxes, ha="center", va="bottom",
                    fontsize=4.5, color="black",
                    bbox=dict(facecolor="white", alpha=0.65, pad=0.8, edgecolor="none"))

    fig.tight_layout()
    fig.savefig(OUTPUT, bbox_inches="tight")
    print(f"Saved {OUTPUT}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache",      default=DEFAULT_CACHE)
    parser.add_argument("--step",       type=int, default=0,   help="0=SEP token, N=SEP+N-th trace token")
    parser.add_argument("--n-puzzles",  type=int, default=6400)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--data",       default=None, help="load precomputed .npz instead of running")
    args = parser.parse_args()

    if args.data:
        raw  = np.load(args.data, allow_pickle=True)
        data = {k: raw[k] for k in raw}
    else:
        data = compute_data(args)
        np.savez(DATA_PATH, **data)
        print(f"Saved data to {DATA_PATH}")

    plot(data)


if __name__ == "__main__":
    main()
