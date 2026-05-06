"""Figure: head DLA for a digit across the board, conditioned on a cell's clue state.

For every (cell, digit) pair, splits puzzles by whether the cell holds that digit as
a clue, then computes the head's mean DLA to all 81 placement tokens for that digit.

  present: the cell has the digit as a clue
  absent:  the cell does not (empty or holds a different digit)

All 729 (cell × digit) conditions are computed in a single forward-pass run and saved
to one .npz (~30 MB). Re-plotting for any combination of cell / digit / layer / head
is then free via --data.

Usage:
    uv run python plots/scripts/fig_head_dla_cell.py --layer 2 --head 5 --row 3 --col 4 --digit 7
    uv run python plots/scripts/fig_head_dla_cell.py --data plots/data/fig_head_dla_cell.npz --layer 2 --head 5 --row 3 --col 4 --digit 7
"""

import argparse
import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
from tqdm import tqdm

from sudoku.activations import load_probe_dataset, load_checkpoint, derive_n_clues
from sudoku.data import PAD_TOKEN, SEP_TOKEN

sns.set_theme(style="ticks", font="Avenir", context="paper")

DEFAULT_CACHE = "results/3M-backtracking-packing/activations.npz"
DATA_PATH     = "plots/data/fig_head_dla_cell.npz"


def _ln(x, scale, bias, eps=1e-5):
    m = x.mean(-1, keepdims=True)
    v = x.var(-1, keepdims=True)
    return (x - m) / jnp.sqrt(v + eps) * scale + bias


def compute_data(args) -> dict:
    _, puzzles, sequences, n_clues, _ = load_probe_dataset(args.cache)
    if n_clues is None:
        n_clues = derive_n_clues(puzzles)

    ckpt_dir = args.cache.replace("activations.npz", "checkpoint")
    params, model = load_checkpoint(ckpt_dir)

    cfg      = model.config
    n_layers = cfg.n_layers
    n_heads  = cfg.n_heads
    d_model  = cfg.d_model
    T        = cfg.max_seq_len
    head_dim = d_model // n_heads

    @jax.jit
    def forward_gather_heads(tokens, query_pos):
        B     = tokens.shape[0]
        b_idx = jnp.arange(B)
        x     = (params["token_emb"]["embedding"][tokens]
                 + params["pos_emb"]["embedding"][jnp.arange(T)[None, :]])

        head_outs_layers = []
        for li in range(n_layers):
            blk  = params[f"block_{li}"]
            attn = blk["CausalSelfAttention_0"]

            h   = _ln(x, blk["LayerNorm_0"]["scale"], blk["LayerNorm_0"]["bias"])
            qkv = h @ attn["qkv"]["kernel"] + attn["qkv"]["bias"]
            q   = qkv[..., :d_model         ].reshape(B, T, n_heads, head_dim).transpose(0, 2, 1, 3)
            k   = qkv[..., d_model:2*d_model].reshape(B, T, n_heads, head_dim).transpose(0, 2, 1, 3)
            v   = qkv[..., 2*d_model:       ].reshape(B, T, n_heads, head_dim).transpose(0, 2, 1, 3)

            scores = (q @ k.transpose(0, 1, 3, 2)) * (head_dim ** -0.5)
            mask   = jnp.tril(jnp.ones((T, T), dtype=bool))
            scores = jnp.where(mask[None, None], scores, jnp.finfo(scores.dtype).min)
            z      = jax.nn.softmax(scores, axis=-1) @ v

            W_O_h = attn["proj"]["kernel"].reshape(n_heads, head_dim, d_model)
            z_q   = z[b_idx[:, None], jnp.arange(n_heads)[None, :], query_pos[:, None]]
            head_outs_layers.append(jnp.einsum("bhd,hde->bhe", z_q, W_O_h))

            z_cat = z.transpose(0, 2, 1, 3).reshape(B, T, d_model)
            x = x + z_cat @ attn["proj"]["kernel"] + attn["proj"]["bias"]
            h2 = _ln(x, blk["LayerNorm_1"]["scale"], blk["LayerNorm_1"]["bias"])
            h2 = jax.nn.gelu(h2 @ blk["Dense_0"]["kernel"] + blk["Dense_0"]["bias"])
            x  = x + h2 @ blk["Dense_1"]["kernel"] + blk["Dense_1"]["bias"]

        head_outs = jnp.stack(head_outs_layers, axis=1)
        resid     = x[b_idx, query_pos]
        return head_outs, resid

    n_total     = min(args.n_puzzles, len(sequences)) if args.n_puzzles else len(sequences)
    query_label = "SEP" if args.step == 0 else f"SEP+{args.step}"

    W_U  = np.asarray(params["lm_head"]["kernel"], dtype=np.float32)
    ln_W = np.asarray(params["LayerNorm_0"]["scale"], dtype=np.float32)[:, None] * W_U

    # acc_digit[d]: sum of DLA board for digit d over all valid puzzles — same for all cells
    # shape (9, n_layers, n_heads, 81)
    acc_digit   = np.zeros((9, n_layers, n_heads, 81), dtype=np.float64)
    # acc_present[cell, d]: sum only over puzzles where that (cell, d) is a clue
    # shape (81, 9, n_layers, n_heads, 81)
    acc_present = np.zeros((81, 9, n_layers, n_heads, 81), dtype=np.float64)
    cnt_present = np.zeros((81, 9), dtype=np.int64)
    n_valid     = 0

    for batch_start in tqdm(range(0, n_total, args.batch_size), desc=f"DLA ({query_label})"):
        batch_end = min(batch_start + args.batch_size, n_total)
        B         = batch_end - batch_start

        tokens_np   = np.full((args.batch_size, T), PAD_TOKEN, dtype=np.int32)
        qpos_np     = np.zeros(args.batch_size, dtype=np.int32)
        local_valid = np.zeros(args.batch_size, dtype=bool)

        for i in range(B):
            s  = sequences[batch_start + i]
            L  = min(len(s), T)
            tokens_np[i, :L] = s[:L]
            nc = int(n_clues[batch_start + i])
            qp = nc + args.step
            if nc < T and tokens_np[i, nc] == SEP_TOKEN and qp < T and tokens_np[i, qp] != PAD_TOKEN:
                qpos_np[i]     = qp
                local_valid[i] = True

        h_q, r_f     = forward_gather_heads(jnp.array(tokens_np), jnp.array(qpos_np))
        head_outs_np = np.asarray(h_q)[:B]
        resid_np     = np.asarray(r_f)[:B]

        resid_std = resid_np.std(axis=-1, keepdims=True)[:, None, None, :]
        dla       = (head_outs_np - head_outs_np.mean(axis=-1, keepdims=True)) / resid_std @ ln_W
        # dla_flat: (B, n_layers, n_heads, 81, 9) — DLA per board cell per digit
        dla_flat  = dla[..., :729].reshape(B, n_layers, n_heads, 81, 9)

        for b in range(B):
            if not local_valid[b]:
                continue
            n_valid += 1

            # dla_d: (9, n_layers, n_heads, 81) — reorder digit to front for indexing
            dla_d = dla_flat[b].transpose(3, 0, 1, 2)
            acc_digit += dla_d

            nc        = int(n_clues[batch_start + b])
            clue_toks = tokens_np[b, :nc]
            clue_toks = clue_toks[(clue_toks >= 0) & (clue_toks <= 728)]
            if len(clue_toks) == 0:
                continue

            cell_arr  = clue_toks // 9   # (n_clues,) cell index 0-80
            digit_arr = clue_toks % 9    # (n_clues,) digit index 0-8

            # dla_d[digit_arr]: (n_clues, n_layers, n_heads, 81)
            np.add.at(acc_present, (cell_arr, digit_arr), dla_d[digit_arr])
            np.add.at(cnt_present, (cell_arr, digit_arr), 1)

    print(f"Used {n_valid}/{n_total} puzzles (query = {query_label})")

    cnt_absent = n_valid - cnt_present   # (81, 9)

    # mean_present[cell, digit, layer, head, board_cell]
    mean_present = (acc_present / np.maximum(cnt_present, 1)[:, :, None, None, None])
    # mean_absent: subtract the present accumulator from the digit total
    mean_absent = (
        (acc_digit[None, :] - acc_present)           # (81, 9, L, H, 81)
        / np.maximum(cnt_absent, 1)[:, :, None, None, None]
    )

    # Reshape board dim to 9×9 for plotting
    return dict(
        present     = mean_present.reshape(81, 9, n_layers, n_heads, 9, 9).astype(np.float32),
        absent      = mean_absent .reshape(81, 9, n_layers, n_heads, 9, 9).astype(np.float32),
        cnt_present = cnt_present,
        cnt_absent  = cnt_absent,
        n_valid     = n_valid,
        step        = args.step,
    )


def plot(data: dict, args):
    R, C, D   = args.row, args.col, args.digit
    layer, head = args.layer, args.head
    cell_idx  = R * 9 + C
    d_idx     = D - 1

    present = data["present"][cell_idx, d_idx, layer, head]   # (9, 9)
    absent  = data["absent"] [cell_idx, d_idx, layer, head]   # (9, 9)
    n_p     = int(data["cnt_present"][cell_idx, d_idx])
    n_a     = int(data["cnt_absent"] [cell_idx, d_idx])
    step    = int(data["step"])
    query_label = "SEP" if step == 0 else f"SEP+{step}"

    vmax = max(np.abs(present).max(), np.abs(absent).max())
    cmap = "RdBu_r"

    fig, (ax_p, ax_a) = plt.subplots(1, 2, figsize=(7, 3.5))

    for ax, arr, title in [
        (ax_p, present, f"Digit {D} present at ({R},{C})  [n={n_p}]"),
        (ax_a, absent,  f"Digit {D} absent from ({R},{C})  [n={n_a}]"),
    ]:
        im = ax.imshow(arr, vmin=-vmax, vmax=vmax, cmap=cmap, interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([])
        for v in [2.5, 5.5]:
            ax.axvline(v, color="black", linewidth=0.8)
            ax.axhline(v, color="black", linewidth=0.8)
        ax.add_patch(patches.Rectangle(
            (C - 0.5, R - 0.5), 1, 1,
            fill=False, edgecolor="black", linewidth=2.5,
        ))
        ax.set_title(title, fontsize=8)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(
        f"L{layer} H{head} DLA for digit {D} at [{query_label}]",
        fontsize=10,
    )
    sns.despine(fig, left=True, bottom=True)
    fig.tight_layout()

    outfile = f"plots/figures/fig_head_dla_cell_r{R}c{C}d{D}_L{layer}H{head}.pdf"
    fig.savefig(outfile, bbox_inches="tight")
    print(f"Saved {outfile}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache",      default=DEFAULT_CACHE)
    parser.add_argument("--step",       type=int, default=0)
    parser.add_argument("--n-puzzles",  type=int, default=6400)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--data",       default=None, help="load precomputed .npz instead of running")
    parser.add_argument("--row",        type=int, required=True, help="conditioning cell row (0-based)")
    parser.add_argument("--col",        type=int, required=True, help="conditioning cell col (0-based)")
    parser.add_argument("--digit",      type=int, required=True, help="digit 1-9")
    parser.add_argument("--layer",      type=int, required=True)
    parser.add_argument("--head",       type=int, required=True)
    args = parser.parse_args()

    if args.data:
        raw  = np.load(args.data, allow_pickle=True)
        data = {k: raw[k] for k in raw}
    else:
        data = compute_data(args)
        np.savez(DATA_PATH, **data)
        print(f"Saved data to {DATA_PATH}")

    plot(data, args)


if __name__ == "__main__":
    main()
