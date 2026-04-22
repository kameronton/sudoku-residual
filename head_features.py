"""
head_features.py — Attention head analysis at an arbitrary query token.

For each of the 64 heads, compute how much attention the query token pays to each of
the 33 sudoku substructures (9 rows + 9 cols + 9 boxes + 3 bands + 3 stacks) formed
by all fill tokens causally available to the query.

Two modes depending on QUERY_STEP:
  0 → query = [SEP]; keys = clue fill tokens only (trace not yet visible).
      EXCLUDE_SELF has no effect (SEP is not a fill token).
  N → query = SEP+N-th trace token; keys = all fill tokens before query_pos,
      i.e. both clues *and* trace placements already made.
      EXCLUDE_SELF removes the query token itself from the key pool (now meaningful).
      EXCLUDE_SEP removes the [SEP] token from attention before scoring.

  Raw score:   mean over puzzles of Σ attn(query → fill tokens in substructure S)
  Enrichment:  raw / (n_fills_in_S / n_total_keys)  — removes density imbalance

Outputs:
  head_features_raw.png     64×33 heatmap of mean raw attention to each substructure
  head_features_enrich.png  64×33 heatmap of enrichment (× uniform-fill baseline)
  head_features_cells.png   per-head 9×9 attention grid + SEP attention column

Run:  uv run python head_features.py
"""

import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from tqdm import tqdm

from sudoku.activations import load_probe_dataset, load_checkpoint
from sudoku.data import PAD_TOKEN, SEP_TOKEN

# ── Config ────────────────────────────────────────────────────────────────────
CACHE_PATH       = "results/3M-backtracking-packing/activations.npz"
CKPT_DIR         = "results/3M-backtracking-packing/checkpoint"

N_PUZZLES        = 6400
BATCH_SIZE       = 32
QUERY_STEP       = 5      # 0 = SEP token, 1 = first trace token after SEP, …
EXCLUDE_SEP      = False  # if True, exclude [SEP] attention before scoring
EXCLUDE_SELF     = True   # exclude the query token itself from the key pool (QUERY_STEP>0 only)
ENRICH_THRESHOLD = 3.0    # × baseline → blue outline in plot

# ── Load ──────────────────────────────────────────────────────────────────────
_, puzzles, sequences, n_clues, _ = load_probe_dataset(CACHE_PATH)
params, model = load_checkpoint(CKPT_DIR)

cfg      = model.config
n_layers = cfg.n_layers
n_heads  = cfg.n_heads
T        = cfg.max_seq_len

print(f"n_layers={n_layers}, n_heads={n_heads}, max_seq_len={T}")

if n_clues is None:
    from sudoku.activations import derive_n_clues
    n_clues = derive_n_clues(puzzles)


@jax.jit
def get_attn_weights(tokens):
    """tokens: (B, T) int32 → (B, n_layers, n_heads, T, T) float32."""
    _, intermediates = model.apply({"params": params}, tokens, return_intermediates=True)
    return jnp.stack(
        [intermediates[f"layer_{i}_attn_weights"] for i in range(n_layers)],
        axis=1,
    )


# ── Accumulators ──────────────────────────────────────────────────────────────
# 33 substructures: 0-8 = rows, 9-17 = cols, 18-26 = boxes, 27-29 = bands, 30-32 = stacks
acc_raw    = np.zeros((n_layers, n_heads, 33), dtype=np.float64)
acc_enrich = np.zeros((n_layers, n_heads, 33), dtype=np.float64)
acc_count  = np.zeros(33, dtype=np.int64)   # puzzles contributing per substructure

# Per-cell accumulators: cell index = row*9 + col
acc_cell     = np.zeros((n_layers, n_heads, 81), dtype=np.float64)
acc_cell_cnt = np.zeros(81, dtype=np.int64)   # puzzles where cell appeared as clue

# SEP attention accumulator
acc_sep      = np.zeros((n_layers, n_heads), dtype=np.float64)

n_valid = 0   # puzzles with a valid query token

n_total = min(N_PUZZLES, len(sequences)) if N_PUZZLES else len(sequences)

query_label = "SEP" if QUERY_STEP == 0 else f"SEP+{QUERY_STEP}"

for batch_start in tqdm(range(0, n_total, BATCH_SIZE), desc=f"{query_label} attention"):
    batch_end  = min(batch_start + BATCH_SIZE, n_total)
    B          = batch_end - batch_start

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

        query_pos = nc + QUERY_STEP
        if query_pos >= T or toks[query_pos] == PAD_TOKEN:
            continue
        n_valid += 1

        # Attention from query token to all tokens: (n_layers, n_heads, T)
        attn_q = attn_w[b, :, :, query_pos, :]

        # Always track SEP attention for the cell plot column
        acc_sep += attn_q[:, :, nc]

        # Key pool: all positions 0..query_pos, then optionally drop SEP / self.
        key_pos = np.arange(query_pos + 1, dtype=np.int32)
        if EXCLUDE_SEP:
            key_pos = key_pos[key_pos != nc]
        if EXCLUDE_SELF:
            key_pos = key_pos[key_pos != query_pos]

        total_keys = len(key_pos)   # enrichment denominator over the full pool

        # Fill tokens for substructure/cell analysis (SEP has no grid position)
        is_fill  = (toks[key_pos] >= 0) & (toks[key_pos] <= 728)
        fill_pos = key_pos[is_fill]
        if len(fill_pos) == 0:
            continue

        key_sum   = attn_q[:, :, key_pos].sum(-1, keepdims=True)  # total attn over kept pool
        clue_attn = attn_q[:, :, fill_pos] / np.maximum(key_sum, 1e-12)

        t32    = toks[fill_pos].astype(np.int32)
        rows   = t32 // 81
        cols   = (t32 % 81) // 9
        boxes  = (rows // 3) * 3 + cols // 3
        bands  = rows // 3
        stacks = cols // 3

        # for si in range(9):
        #     for attr, offset in [(rows, 0), (cols, 9), (boxes, 18)]:
        #         local = np.where(attr == si)[0]
        #         if len(local) == 0:
        #             continue
        #         raw      = clue_attn[:, :, local].sum(-1)
        #         baseline = len(local) / total_keys
        #         acc_raw   [:, :, offset + si] += raw
        #         acc_enrich[:, :, offset + si] += raw / baseline
        #         acc_count [offset + si]        += 1

        # for si in range(3):
        #     for attr, offset in [(bands, 27), (stacks, 30)]:
        #         local = np.where(attr == si)[0]
        #         if len(local) == 0:
        #             continue
        #         raw      = clue_attn[:, :, local].sum(-1)
        #         baseline = len(local) / total_keys
        #         acc_raw   [:, :, offset + si] += raw
        #         acc_enrich[:, :, offset + si] += raw / baseline
        #         acc_count [offset + si]        += 1

        # Per-cell: attention to each individual fill token
        cell_idx = rows * 9 + cols                    # (n_fills_in_pool,)
        acc_cell[:, :, cell_idx] += clue_attn         # (n_layers, n_heads, n_fills_in_pool)
        acc_cell_cnt[cell_idx]   += 1

print(f"Used {n_valid}/{n_total} puzzles (had valid query token at {query_label})")


# ── Averages ──────────────────────────────────────────────────────────────────
mean_raw    = acc_raw    / np.maximum(acc_count, 1)[None, None, :]
mean_enrich = acc_enrich / np.maximum(acc_count, 1)[None, None, :]
mean_sep    = np.zeros((n_layers, n_heads)) if EXCLUDE_SEP else acc_sep / max(n_valid, 1)

n_heads_total = n_layers * n_heads
raw2d = mean_raw.reshape(n_heads_total, 33)
enr2d = mean_enrich.reshape(n_heads_total, 33)
peaked = np.zeros_like(enr2d, dtype=bool)
top4 = np.argsort(enr2d, axis=1)[:, -4:]
peaked[np.arange(n_heads_total)[:, None], top4] = True


# ── Plot ──────────────────────────────────────────────────────────────────────
col_labels = (
    [f"r{i}" for i in range(9)]
    + [f"c{i}" for i in range(9)]
    + [f"b{i}" for i in range(9)]
    + [f"band{i}" for i in range(3)]
    + [f"stk{i}" for i in range(3)]
)
row_labels = [f"L{li}H{hi}" for li in range(n_layers) for hi in range(n_heads)]


# def heatmap(data, title, outfile, peaked_mask=None, vmax=None, cmap="YlOrRd"):
#     nrows, ncols = data.shape
#     fig, ax = plt.subplots(figsize=(ncols * 0.45, nrows * 0.2 + 0.5))

#     im = ax.imshow(data, aspect='auto', cmap=cmap,
#                    vmin=0, vmax=(vmax if vmax is not None else float(data.max())),
#                    interpolation='nearest')
#     plt.colorbar(im, ax=ax, fraction=0.02, pad=0.02)

#     ax.set_xticks(range(ncols))
#     ax.set_xticklabels(col_labels, rotation=45, ha='right', fontsize=8)
#     ax.set_yticks(range(nrows))
#     ax.set_yticklabels(row_labels, fontsize=6.5)

#     # Separators between substructure groups and between layers
#     for x in [8.5, 17.5, 26.5, 29.5]:
#         ax.axvline(x, color='black', linewidth=1.0)
#     for li in range(1, n_layers):
#         ax.axhline(li * n_heads - 0.5, color='gray', linewidth=0.5, alpha=0.6)

#     if peaked_mask is not None:
#         for r, c in zip(*np.where(peaked_mask)):
#             ax.add_patch(plt.Rectangle(
#                 (c - 0.5, r - 0.5), 1, 1,
#                 fill=False, edgecolor='dodgerblue', linewidth=1.5,
#             ))

#     ax.set_title(title, fontsize=10, pad=8)
#     fig.tight_layout()
#     fig.savefig(outfile, dpi=150, bbox_inches='tight')
#     print(f"Saved {outfile}")
#     plt.pause(0.1)


# excl_note = " (SEP excluded from clue pool)" if EXCLUDE_SEP else ""

# heatmap(
#     raw2d,
#     f"Raw attention from [{query_label}] to substructures — {n_valid} puzzles{excl_note}\n"
#     f"Blue = top-4 enrichment per head",
#     "head_features_raw.png",
#     peaked_mask=peaked,
# )

# heatmap(
#     enr2d,
#     f"Enrichment (÷ clue-density baseline) — {n_valid} puzzles{excl_note}\n"
#     f"Blue = top-4 per head",
#     "head_features_enrich.png",
#     peaked_mask=peaked,
#     cmap="PuRd",
# )

# # ── Summary ───────────────────────────────────────────────────────────────────
# print("\nTop-4 enrichment per head:")
# sub_names = ([f"row{i}" for i in range(9)] + [f"col{i}" for i in range(9)]
#              + [f"box{i}" for i in range(9)] + [f"band{i}" for i in range(3)]
#              + [f"stk{i}" for i in range(3)])
# for si, name in enumerate(sub_names):
#     heads = [
#         f"L{li}H{hi}({enr2d[li*n_heads+hi, si]:.1f}x)"
#         for li in range(n_layers) for hi in range(n_heads)
#         if peaked[li * n_heads + hi, si]
#     ]
#     if heads:
#         print(f"  {name}: {', '.join(heads)}")

# print("\nDone.")

# ── Per-cell attention grid ───────────────────────────────────────────────────
mean_cell = acc_cell / np.maximum(acc_cell_cnt, 1)[None, None, :]  # (n_layers, n_heads, 81)
# cells never seen as clues → NaN so they render as a distinct colour
mean_cell[:, :, acc_cell_cnt == 0] = np.nan

cell_grid = mean_cell.reshape(n_layers, n_heads, 9, 9)  # (n_layers, n_heads, row, col)

# Combined (9, 10) image: cols 0-8 = board, col 9 = SEP attention
sep_col = mean_sep[:, :, np.newaxis, np.newaxis] * np.ones((n_layers, n_heads, 9, 1))
combined = np.concatenate([cell_grid, sep_col], axis=-1)  # (n_layers, n_heads, 9, 10)

fig, axes = plt.subplots(n_layers, n_heads,
                         figsize=(n_heads * 1.5, n_layers * 1.3),
                         squeeze=False)

cmap = plt.get_cmap("YlOrRd").copy()
cmap.set_bad(color="lightgrey")   # NaN cells

for li in range(n_layers):
    for hi in range(n_heads):
        ax  = axes[li][hi]
        g   = combined[li, hi]                       # (9, 10)
        vmax = np.nanmax(g) if not np.all(np.isnan(g)) else 1.0
        ax.imshow(g, vmin=0, vmax=vmax, cmap=cmap, interpolation='nearest')
        ax.set_xticks([])
        ax.set_yticks([])
        # Box separators within the board
        for v in [2.5, 5.5]:
            ax.axvline(v, color='black', linewidth=0.6)
            ax.axhline(v, color='black', linewidth=0.6)
        # Separator between board and SEP column
        ax.axvline(8.5, color='navy', linewidth=1.0)
        # SEP label at top of first row
        if li == 0:
            ax.set_title(f"H{hi}", fontsize=7)
            ax.text(9, -0.8, "S", ha='center', va='bottom', fontsize=5, color='navy')
        if hi == 0:
            ax.set_ylabel(f"L{li}", fontsize=7)

excl_note_cell = " | SEP excl. from clue pool" if EXCLUDE_SEP else ""
fig.suptitle(
    f"Mean attention from [{query_label}] per cell — {n_valid} puzzles{excl_note_cell}\n"
    f"(col 10 = SEP attention)",
    fontsize=10, y=1.01,
)
fig.tight_layout()
fig.savefig("head_features_cells.png", dpi=150, bbox_inches='tight')
plt.pause(0.1)
input()
print("Saved head_features_cells.png")
