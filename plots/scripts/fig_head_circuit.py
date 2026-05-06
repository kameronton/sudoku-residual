"""Figure: attention pattern + conditional DLA for a single head and substructure.

Combines fig_head_attention_cells and fig_head_dla_substructure into one 3-panel figure:
  1. Mean attention from [SEP] to each board cell
  2. Mean DLA over digits when digit IS present in the substructure instance
  3. Mean DLA over digits when digit is ABSENT from the substructure instance

Requires precomputed data from both source scripts.

Usage:
    uv run python plots/scripts/fig_head_circuit.py \
        --attn-data plots/data/fig_head_attention_cells.npz \
        --dla-data  plots/data/fig_head_dla_substructure.npz \
        --layer 5 --head 7 --sub rows --instance 4
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns

sns.set_theme(style="ticks", context="paper")

plt.rcParams.update({
    'font.size': 10,
    'axes.titlesize': 10,
    'axes.labelsize': 10,
    'xtick.labelsize': 8,     # Will become 8pt
    'ytick.labelsize': 8,
    'legend.fontsize': 9,     # Will become 9pt
    'font.family': 'serif',
    'pdf.fonttype': 42
})


def _sub_rect(sub, instance):
    if sub == "rows":
        return (-0.5, instance - 0.5, 9, 1)
    if sub == "cols":
        return (instance - 0.5, -0.5, 1, 9)
    r0, c0 = 3 * (instance // 3), 3 * (instance % 3)
    return (c0 - 0.5, r0 - 0.5, 3, 3)


def _add_grid(ax):
    for v in [2.5, 5.5]:
        ax.axvline(v, color="black", linewidth=0.8)
        ax.axhline(v, color="black", linewidth=0.8)
    ax.set_xticks([]); ax.set_yticks([])


def _outline(ax, sub, instances):
    for inst in instances:
        x, y, w, h = _sub_rect(sub, inst)
        ax.add_patch(patches.Rectangle(
            (x, y), w, h, fill=False, edgecolor="black", linewidth=2.5,
        ))


def plot(attn_data, dla_data, args):
    layer, head = args.layer - 1, args.head - 1
    sub         = args.sub
    instances   = [int(x) - 1 for x in args.instance.split(",")]
    sub_label   = {"rows": "row", "cols": "col", "boxes": "box"}[sub]
    inst_label  = args.instance   # already 1-indexed as given by the user

    # ── Attention ─────────────────────────────────────────────────────────────
    attn  = attn_data["cell_grid"][layer, head]          # (9, 9)
    s_pct = float(attn_data["mean_sep"][layer, head])

    # ── DLA ───────────────────────────────────────────────────────────────────
    # present/absent: (9_inst, 9_digit, 9, 9) or (9_digit, 9, 9) depending on version
    p_key = f"present_{sub}"
    a_key = f"absent_{sub}"

    p_arr = dla_data[p_key]   # (L, H, 9_inst, 9_digit, 9, 9)
    a_arr = dla_data[a_key]

    if len(instances) == 1:
        present = p_arr[layer, head, instances[0]].mean(axis=0)   # avg over digits → (9,9)
        absent  = a_arr[layer, head, instances[0]].mean(axis=0)
        n_p = int(dla_data[f"cnt_present_{sub}"][instances[0]].sum())
        n_a = int(dla_data[f"cnt_absent_{sub}"] [instances[0]].sum())
    else:
        # joint condition stored with key encoding, e.g. present_rows_0_1_2
        joint_key_p = f"present_{sub}_{'_'.join(map(str, instances))}"
        joint_key_a = f"absent_{sub}_{'_'.join(map(str, instances))}"
        present = dla_data[joint_key_p][layer, head].mean(axis=0)
        absent  = dla_data[joint_key_a][layer, head].mean(axis=0)
        n_p = int(dla_data[f"cnt_present_{sub}_{'_'.join(map(str, instances))}"].sum())
        n_a = int(dla_data[f"cnt_absent_{sub}_{'_'.join(map(str, instances))}"].sum())

    step        = int(dla_data["step"])
    query_label = "CLUES_END" if step == 0 else f"SEP+{step}"
    vmax_dla    = max(np.abs(present).max(), np.abs(absent).max())

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(9, 3.5))
    ax_attn, ax_p, ax_a = axes

    # Panel 1: attention
    cmap_attn = plt.get_cmap("YlOrRd").copy()
    cmap_attn.set_bad(color="lightgrey")
    vmax_attn = np.nanmax(attn) if not np.all(np.isnan(attn)) else 1.0
    im_a = ax_attn.imshow(attn, vmin=0, vmax=vmax_attn, cmap=cmap_attn, interpolation="nearest")
    _add_grid(ax_attn)
    _outline(ax_attn, sub, instances)
    ax_attn.set_title(f"Attention from [{query_label}]")#, fontsize=8)
    ax_attn.text(0.5, -0.04, f"[CLUES_END]: {s_pct:.0%}",
                 transform=ax_attn.transAxes, ha="center", va="top")#, fontsize=7)
    plt.colorbar(im_a, ax=ax_attn, fraction=0.046, pad=0.04)

    # Panels 2 & 3: DLA
    for ax, arr, title in [
        (ax_p, present, f"Digit present in {sub_label} {{{inst_label}}}  [n={n_p}]"),
        (ax_a, absent,  f"Digit absent from {sub_label} {{{inst_label}}}  [n={n_a}]"),
    ]:
        print(f"Present: {n_p}, {n_p//9}, Absent: {n_a}, {n_a//9}")
        im = ax.imshow(arr, vmin=-vmax_dla, vmax=vmax_dla, cmap="RdBu_r", interpolation="nearest")
        _add_grid(ax)
        _outline(ax, sub, instances)
        ax.set_title(title)#, fontsize=8)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # fig.suptitle(
    #     f"L{args.layer} H{args.head} — {sub_label} {{{inst_label}}} at [{query_label}]",
    #     fontsize=10,
    # )
    sns.despine(fig, left=True, bottom=True)
    fig.tight_layout()

    outfile = f"plots/figures/fig_head_circuit_L{args.layer}H{args.head}_{sub}{inst_label}.pdf"
    fig.savefig(outfile, bbox_inches="tight")
    print(f"Saved {outfile}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--attn-data", required=True, help="fig_head_attention_cells.npz")
    parser.add_argument("--dla-data",  required=True, help="fig_head_dla_substructure.npz")
    parser.add_argument("--layer",    type=int, required=True)
    parser.add_argument("--head",     type=int, required=True)
    parser.add_argument("--sub",      choices=["rows", "cols", "boxes"], required=True)
    parser.add_argument("--instance", required=True,
                        help="instance index or comma-separated set, e.g. 4 or 0,1,2")
    args = parser.parse_args()

    attn_data = {k: v for k, v in np.load(args.attn_data, allow_pickle=True).items()}
    dla_data  = {k: v for k, v in np.load(args.dla_data,  allow_pickle=True).items()}

    plot(attn_data, dla_data, args)


if __name__ == "__main__":
    main()
