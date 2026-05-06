"""Figure: head DLA conditioned on digit presence at the substructure level.

For a given head, substructure type (rows/cols/boxes), and instance index, splits
puzzles by whether each digit is present or absent in that substructure instance
among the clues, then plots the mean DLA board for the digit averaged over all 9 digits.

  present: digit present somewhere in the instance → expect suppression within it
  absent:  digit absent from the instance          → expect promotion within it

All (type × instance × digit × layer × head) combinations are precomputed in one
forward-pass run and saved to ~10 MB. Re-plotting is free via --data.

Usage:
    uv run python plots/scripts/fig_head_dla_substructure.py --layer 5 --head 7 --sub rows --instance 4
    uv run python plots/scripts/fig_head_dla_substructure.py --data plots/data/fig_head_dla_substructure.npz --layer 5 --head 7 --sub rows --instance 4
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
DATA_PATH     = "plots/data/fig_head_dla_substructure.npz"

CELL_TO = {
    "rows":  np.arange(81) // 9,
    "cols":  np.arange(81) % 9,
    "boxes": (np.arange(81) // 9 // 3) * 3 + (np.arange(81) % 9 // 3),
}

# Rectangle (x, y, w, h) in imshow coordinates to outline the instance
def _sub_rect(sub, instance):
    if sub == "rows":
        return (-0.5, instance - 0.5, 9, 1)
    if sub == "cols":
        return (instance - 0.5, -0.5, 1, 9)
    # boxes: instance = band*3 + stack
    r0, c0 = 3 * (instance // 3), 3 * (instance % 3)
    return (c0 - 0.5, r0 - 0.5, 3, 3)


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

    # acc_digit[d]: sum of DLA board for digit d over all valid puzzles — shared across instances
    # shape: (9_digit, n_layers, n_heads, 9_boardrow, 9_boardcol)
    acc_digit = np.zeros((9, n_layers, n_heads, 9, 9), dtype=np.float64)

    # acc_present[stype]: sum over puzzles where (instance, digit) is a clue
    # shape: (9_inst, 9_digit, n_layers, n_heads, 9_boardrow, 9_boardcol)
    acc_present = {s: np.zeros((9, 9, n_layers, n_heads, 9, 9), dtype=np.float64) for s in CELL_TO}
    cnt_present = {s: np.zeros((9, 9), dtype=np.int64) for s in CELL_TO}
    n_valid = 0

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
        # (B, L, H, 9_boardrow, 9_boardcol, 9_digit) — digit last
        dla_board = dla[..., :729].reshape(B, n_layers, n_heads, 9, 9, 9)

        for b in range(B):
            if not local_valid[b]:
                continue
            n_valid += 1

            # (9_digit, L, H, 9_boardrow, 9_boardcol) — digit first for indexing
            dla_d = dla_board[b].transpose(4, 0, 1, 2, 3)
            acc_digit += dla_d

            nc        = int(n_clues[batch_start + b])
            clue_toks = tokens_np[b, :nc]
            clue_toks = clue_toks[(clue_toks >= 0) & (clue_toks <= 728)]
            if len(clue_toks) == 0:
                continue

            cell_arr  = clue_toks // 9   # cell index 0-80
            digit_arr = clue_toks % 9    # digit index 0-8

            # dla_d[digit_arr]: (n_clues, L, H, 9, 9)
            # In a valid sudoku each (inst, digit) pair is unique per substructure type,
            # so repeated-index issues with += do not arise here.
            for stype, cell_to_inst in CELL_TO.items():
                inst_arr = cell_to_inst[cell_arr]
                np.add.at(acc_present[stype], (inst_arr, digit_arr), dla_d[digit_arr])
                np.add.at(cnt_present[stype], (inst_arr, digit_arr), 1)

    print(f"Used {n_valid}/{n_total} puzzles (query = {query_label})")

    result = dict(n_valid=n_valid, step=args.step)
    for stype in CELL_TO:
        cnt_abs  = n_valid - cnt_present[stype]          # (9, 9)
        denom_p  = np.maximum(cnt_present[stype], 1)[:, :, None, None, None, None]
        denom_a  = np.maximum(cnt_abs,            1)[:, :, None, None, None, None]

        mean_p = acc_present[stype] / denom_p
        mean_a = (acc_digit[None, :] - acc_present[stype]) / denom_a

        # Save as (L, H, 9_inst, 9_digit, 9, 9)
        result[f"present_{stype}"]     = mean_p.transpose(2, 3, 0, 1, 4, 5).astype(np.float32)
        result[f"absent_{stype}"]      = mean_a.transpose(2, 3, 0, 1, 4, 5).astype(np.float32)
        result[f"cnt_present_{stype}"] = cnt_present[stype]
        result[f"cnt_absent_{stype}"]  = cnt_abs

    return result


def plot(data: dict, args):
    layer, head = args.layer, args.head
    sub, inst   = args.sub, args.instance
    step        = int(data["step"])
    query_label = "SEP" if step == 0 else f"SEP+{step}"

    # (9_digit, 9, 9) → average over digits → (9, 9)
    present = data[f"present_{sub}"][layer, head, inst].mean(axis=0)
    absent  = data[f"absent_{sub}"] [layer, head, inst].mean(axis=0)
    n_p     = int(data[f"cnt_present_{sub}"][inst].sum())
    n_a     = int(data[f"cnt_absent_{sub}"] [inst].sum())

    vmax = max(np.abs(present).max(), np.abs(absent).max())
    rect = _sub_rect(sub, inst)

    fig, (ax_p, ax_a) = plt.subplots(1, 2, figsize=(7, 3.5))

    sub_label = sub[:-1]   # "row", "col", "box"
    for ax, arr, title in [
        (ax_p, present, f"Digit present in {sub_label} {inst}  [n≈{n_p//9}]"),
        (ax_a, absent,  f"Digit absent from {sub_label} {inst}  [n≈{n_a//9}]"),
    ]:
        im = ax.imshow(arr, vmin=-vmax, vmax=vmax, cmap="RdBu_r", interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([])
        for v in [2.5, 5.5]:
            ax.axvline(v, color="black", linewidth=0.8)
            ax.axhline(v, color="black", linewidth=0.8)
        ax.add_patch(patches.Rectangle(
            (rect[0], rect[1]), rect[2], rect[3],
            fill=False, edgecolor="black", linewidth=2.5,
        ))
        ax.set_title(title, fontsize=8)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(
        f"L{layer} H{head} — mean DLA over all digits, conditioned on {sub_label} {inst} at [{query_label}]",
        fontsize=9,
    )
    sns.despine(fig, left=True, bottom=True)
    fig.tight_layout()

    outfile = f"plots/figures/fig_head_dla_substructure_L{layer}H{head}_{sub}{inst}.pdf"
    fig.savefig(outfile, bbox_inches="tight")
    print(f"Saved {outfile}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache",      default=DEFAULT_CACHE)
    parser.add_argument("--step",       type=int, default=0)
    parser.add_argument("--n-puzzles",  type=int, default=6400)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--data",       default=None, help="load precomputed .npz instead of running")
    parser.add_argument("--layer",      type=int, required=True)
    parser.add_argument("--head",       type=int, required=True)
    parser.add_argument("--sub",        choices=["rows", "cols", "boxes"], required=True)
    parser.add_argument("--instance",   type=int, required=True, help="substructure index 0-8")
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
