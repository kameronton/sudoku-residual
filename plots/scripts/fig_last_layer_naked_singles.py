"""Figure: naked-single margin distribution + logit lens.

Panel 1 — distribution of within-cell logit margin at the final layer over all
    unique naked-single states (exactly one NS, next token = forced placement).
Panel 2 — logit-lens trajectory for the 9 digit tokens of the forced cell,
    for a user-specified position (--puzzle / --pos).

Usage:
    uv run python plots/scripts/fig_last_layer_naked_singles.py
    uv run python plots/scripts/fig_last_layer_naked_singles.py --data plots/data/fig_last_layer_naked_singles.csv --puzzle 0 --pos 25
    uv run python plots/scripts/fig_last_layer_naked_singles.py --puzzle 12 --pos 47
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm

from sudoku.activations import load_checkpoint
from sudoku.probes.session import ProbeSession, ActivationIndex
from sudoku.data import decode_fill, PAD_TOKEN
from sudoku.data_bt import PUSH_TOKEN, POP_TOKEN, SUCCESS_TOKEN, PAD_TOKEN_BT

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

DEFAULT_CACHE  = "results/3M-backtracking-packing/activations.npz"
OUTPUT_MARGIN  = "plots/figures/fig_last_layer_ns_margin.pdf"
OUTPUT_LENS    = "plots/figures/fig_last_layer_ns_lens.pdf"
DATA_PATH      = "plots/data/fig_last_layer_naked_singles.csv"

_PAD     = {PAD_TOKEN, PAD_TOKEN_BT}
_CONTROL = {PUSH_TOKEN, POP_TOKEN, SUCCESS_TOKEN}


# ── Board helpers ─────────────────────────────────────────────────────────────

def _seq_len(seq) -> int:
    for i, t in enumerate(seq):
        if int(t) in _PAD:
            return i
    return len(seq)


def _replay(seq, sp: int) -> dict[int, int]:
    """Return board state (cell→digit) after processing seq[0:sp+1]."""
    board: dict[int, int] = {}
    stack: list[dict] = []
    for tok in seq[:sp + 1]:
        tok = int(tok)
        if 0 <= tok <= 728:
            r, c, d = decode_fill(tok)
            board[r * 9 + c] = d
        elif tok == PUSH_TOKEN:
            stack.append(dict(board))
        elif tok == POP_TOKEN and stack:
            board = stack.pop()
    return board


def _candidates(board: dict[int, int]) -> list[int]:
    row = [0] * 9; col = [0] * 9; box = [0] * 9
    for cell, d in board.items():
        r, c = divmod(cell, 9)
        bit = 1 << (d - 1)
        row[r] |= bit; col[c] |= bit; box[(r // 3) * 3 + c // 3] |= bit
    full = (1 << 9) - 1
    out = [0] * 81
    for cell in range(81):
        if cell in board:
            continue
        r, c = divmod(cell, 9)
        out[cell] = full & ~row[r] & ~col[c] & ~box[(r // 3) * 3 + c // 3]
    return out


# ── Readout helpers ────────────────────────────────────────────────────────────

def _final_logits(x: np.ndarray, params) -> np.ndarray:
    scale = np.asarray(params["LayerNorm_0"]["scale"], dtype=np.float32)
    bias  = np.asarray(params["LayerNorm_0"]["bias"],  dtype=np.float32)
    W     = np.asarray(params["lm_head"]["kernel"],    dtype=np.float32)
    b     = np.asarray(params["lm_head"]["bias"],      dtype=np.float32)
    x = np.asarray(x, dtype=np.float32)
    ln = (x - x.mean(-1, keepdims=True)) / np.sqrt(x.var(-1, keepdims=True) + 1e-6) * scale + bias
    return ln @ W + b


# ── Data computation ───────────────────────────────────────────────────────────

def _scan_ns(
    session: ProbeSession,
    *,
    require_unique: bool = False,
    filter_branches: bool = True,
) -> tuple[list[dict], list[dict]]:
    """Single pass over all sequences.

    Args:
        require_unique:  if True, all_ns_pairs only includes states with
                         exactly one NS cell (mirrors the unique_ns criterion).
        filter_branches: if True, exclude positions on control tokens, immediately
                         before a control token, or inside exploration branches
                         (stack depth > 0).  Matches the notebook's section-8b
                         context filter.  If False, all positions are included.

    Returns:
        unique_ns   — one entry per position with exactly one NS and next token
                      matching that NS placement (used for margin distribution).
        all_ns_pairs — one entry per (position, NS-cell) pair, subject to
                      require_unique and filter_branches.
    """
    unique_ns: list[dict] = []
    all_ns_pairs: list[dict] = []
    assert session.n_clues is not None

    for pi in tqdm(range(session.n_puzzles), desc="Scanning puzzles"):
        seq = session.sequences[pi]
        slen = _seq_len(seq)

        # Anchor: SEP position (= n_clues) if has_sep, else last-clue position
        nc = int(session.n_clues[pi])
        anchor = nc if session.has_sep else nc - 1

        # Pre-build board with all clue tokens (positions before anchor)
        board: dict[int, int] = {}
        for tok in seq[:anchor]:
            tok = int(tok)
            if 0 <= tok <= 728:
                r, c, d = decode_fill(tok)
                board[r * 9 + c] = d

        # Scan from anchor onwards; depth starts at 0 (no branches in clue phase)
        stack: list[dict] = []
        depth = 0
        for sp in range(anchor, slen - 1):
            tok = int(seq[sp])
            if 0 <= tok <= 728:
                r, c, d = decode_fill(tok)
                board[r * 9 + c] = d
            elif tok == PUSH_TOKEN:
                stack.append(dict(board))
                depth += 1
            elif tok == POP_TOKEN and stack:
                board = stack.pop()
                depth -= 1

            next_tok = int(seq[sp + 1])

            # Control-token positions always skipped: PUSH/POP don't produce a new
            # board state (PUSH saves a snapshot without filling; POP restores one),
            # so including them would double-count the NS cells of the surrounding
            # non-branch board state.
            if tok in _CONTROL:
                continue

            if filter_branches:
                if depth > 0:
                    continue
                if next_tok in _CONTROL:
                    continue

            cands = _candidates(board)
            ns = [(i, m) for i, m in enumerate(cands) if m and not (m & (m - 1))]
            if not ns:
                continue

            # All-NS-pairs, subject to require_unique
            if not require_unique or len(ns) == 1:
                for cell, mask in ns:
                    digit = mask.bit_length()
                    all_ns_pairs.append({"pi": pi, "sp": sp, "cell": cell, "digit": digit})

            # Unique-NS: always exactly one NS + model predicting it (for margin plot)
            if len(ns) == 1:
                cell, mask = ns[0]
                digit = mask.bit_length()
                if next_tok == cell * 9 + (digit - 1):
                    unique_ns.append({"pi": pi, "sp": sp, "cell": cell, "digit": digit})

    return unique_ns, all_ns_pairs


def _make_idx(states: list[dict]) -> tuple[ActivationIndex, np.ndarray, np.ndarray]:
    n = len(states)
    puzzle_idx = np.array([s["pi"]    for s in states], dtype=np.int32)
    seq_pos    = np.array([s["sp"]    for s in states], dtype=np.int32)
    cells      = np.array([s["cell"]  for s in states], dtype=np.int32)
    digits     = np.array([s["digit"] for s in states], dtype=np.int32)
    idx = ActivationIndex(
        puzzle_idx=puzzle_idx, seq_pos=seq_pos,
        step=np.zeros(n, dtype=np.int32),
        n_filled=np.zeros(n, dtype=np.int16),
        tokens=np.zeros(n, dtype=np.int16),
    )
    return idx, cells, digits


def compute_data(unique_ns: list[dict], all_ns_pairs: list[dict],
                 session: ProbeSession, session_attn: ProbeSession,
                 params) -> pd.DataFrame:
    last = session.n_layers - 1

    # ── Post-MLP margin (unique-NS states) ────────────────────────────────────
    n = len(unique_ns)
    idx, cells, digits = _make_idx(unique_ns)
    puzzle_idx = idx.puzzle_idx
    seq_pos    = idx.seq_pos

    acts_mlp   = session.acts(idx, layer=last)                      # (N, d_model)
    logits_mlp = _final_logits(acts_mlp, params)[:, :729]          # (N, 729)
    cell_log   = logits_mlp.reshape(n, 81, 9)[np.arange(n), cells] # (N, 9)
    forced     = cell_log[np.arange(n), digits - 1]
    rest       = cell_log.copy(); rest[np.arange(n), digits - 1] = -np.inf
    margin     = forced - rest.max(axis=1)

    # ── Pre-last-MLP within-cell accuracy (all NS pairs, matching notebook) ───
    m = len(all_ns_pairs)
    idx2, cells2, digits2 = _make_idx(all_ns_pairs)

    acts_attn   = session_attn.acts(idx2, layer=last)               # (M, d_model)
    logits_attn = _final_logits(acts_attn, params)[:, :729]        # (M, 729)
    cell_log2   = logits_attn.reshape(m, 81, 9)[np.arange(m), cells2]  # (M, 9)
    top1_pre    = cell_log2.argmax(axis=1) == (digits2 - 1)
    pct         = top1_pre.mean() * 100
    print(f"Pre-last-MLP within-cell top-1: {pct:.2f}%  ({top1_pre.sum():,} / {m:,} NS-cell pairs)")

    return pd.DataFrame({
        "margin": margin,
        "pi": puzzle_idx, "sp": seq_pos, "cell": cells, "digit": digits,
    })


# ── Logit lens ─────────────────────────────────────────────────────────────────

def _logit_lens(pi: int, sp: int, session: ProbeSession, session_attn: ProbeSession,
                params) -> tuple[list[str], np.ndarray]:
    """Return (labels, logits) where logits is (n_steps, 729).

    n_steps = 1 (emb) + 2 * n_layers (post_attn + post_mlp per layer).
    """
    seq = session.sequences[pi]
    n_layers = session.n_layers

    idx = ActivationIndex(
        puzzle_idx=np.array([pi], dtype=np.int32),
        seq_pos=np.array([sp], dtype=np.int32),
        step=np.zeros(1, dtype=np.int32),
        n_filled=np.zeros(1, dtype=np.int16),
        tokens=np.zeros(1, dtype=np.int16),
    )

    emb = (np.asarray(params["token_emb"]["embedding"][int(seq[sp])], dtype=np.float32)
         + np.asarray(params["pos_emb"]["embedding"][sp], dtype=np.float32))

    resids = [emb[None]]
    labels = ["Emb"]
    for l in range(n_layers):
        resids.append(session_attn.acts(idx, layer=l))
        labels.append(f"L{l}_attn")
        resids.append(session.acts(idx, layer=l))
        labels.append(f"L{l}_mlp")

    all_resids = np.vstack(resids)                              # (2*n_layers+1, d_model)
    return labels, _final_logits(all_resids, params)[:, :729]  # (n_steps, 729)


# ── Plotting ───────────────────────────────────────────────────────────────────

def plot(df: pd.DataFrame, pi: int, sp: int, cell: int, digit: int,
         session: ProbeSession, session_attn: ProbeSession, params):
    # Figure 1 — margin histogram
    fig1, ax_dist = plt.subplots(figsize=(4.5, 2.5))
    ax_dist.hist(np.asarray(df["margin"]), bins=60, color="steelblue", edgecolor="none", alpha=0.85)
    ax_dist.set_xlabel("Within-cell logit margin (forced − runner-up)")
    ax_dist.set_ylabel("Count")
    ax_dist.set_title(f"NS prediction margin  (n = {len(df):,})")
    sns.despine(fig1)
    fig1.tight_layout()
    fig1.savefig(OUTPUT_MARGIN, bbox_inches="tight")
    print(f"Saved {OUTPUT_MARGIN}")

    # Figure 2 — logit lens for forced cell
    labels, logits = _logit_lens(pi, sp, session, session_attn, params)
    x = np.arange(len(labels))
    r, c = divmod(cell, 9)
    cell_logits = logits[:, cell * 9: cell * 9 + 9]  # (n_steps, 9)

    fig2, ax_lens = plt.subplots(figsize=(4.5, 2.5))
    for d0 in range(9):
        is_forced = (d0 == digit - 1)
        ax_lens.plot(
            x, cell_logits[:, d0],
            color="tab:red" if is_forced else "#aaaaaa",
            lw=2.0 if is_forced else 0.8,
            alpha=1.0 if is_forced else 0.55,
            zorder=4 if is_forced else 2,
            label=f"digit {digit} (forced)" if is_forced else None,
        )

    ax_lens.set_xticks(x)
    ax_lens.set_xticklabels(labels, rotation=45, ha="right")
    ax_lens.set_ylabel("Logit score")
    ax_lens.set_title(
        f"Logit lens — R{r+1}C{c+1}, forced digit {digit}"
        f"  (puzzle {pi}, pos {sp})"
    )
    ax_lens.grid(axis="y", lw=0.4, alpha=0.4)
    ax_lens.legend(frameon=False)
    sns.despine(fig2)
    fig2.tight_layout()
    fig2.savefig(OUTPUT_LENS, bbox_inches="tight")
    print(f"Saved {OUTPUT_LENS}")


# ── Entry point ────────────────────────────────────────────────────────────────

def _forced_cell_digit(seq, sp: int) -> tuple[int, int]:
    board = _replay(seq, sp)
    cands = _candidates(board)
    ns = [(i, m) for i, m in enumerate(cands) if m and not (m & (m - 1))]
    cell, mask = ns[0]
    return cell, mask.bit_length()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache",  default=DEFAULT_CACHE)
    parser.add_argument("--data",   default=None,
                        help="precomputed CSV; skips scan + margin computation")
    parser.add_argument("--puzzle", type=int, default=None,
                        help="puzzle index for logit-lens panel")
    parser.add_argument("--pos",    type=int, default=None,
                        help="sequence position for logit-lens panel")
    parser.add_argument("--require-unique", action=argparse.BooleanOptionalAction,
                        default=False,
                        help="restrict NS-cell pairs to states with exactly one NS "
                             "(default: off — matches notebook section-8b)")
    parser.add_argument("--filter-branches", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="exclude control tokens and exploration-branch positions "
                             "(default: on — matches notebook section-8b)")
    args = parser.parse_args()

    ckpt_dir = args.cache.replace("activations.npz", "checkpoint")
    params, _ = load_checkpoint(ckpt_dir)

    session      = ProbeSession(args.cache, act_type="post_mlp")
    session_attn = ProbeSession(args.cache, act_type="post_attn")

    if args.data:
        df = pd.read_csv(args.data)
        if args.puzzle is None or args.pos is None:
            parser.error("--puzzle and --pos are required when using --data")
        pi, sp = args.puzzle, args.pos
        cell, digit = _forced_cell_digit(session.sequences[pi], sp)
    else:
        unique_ns, all_ns_pairs = _scan_ns(
            session,
            require_unique=args.require_unique,
            filter_branches=args.filter_branches,
        )
        print(f"\nFound {len(unique_ns):,} unique naked-single states")
        print(f"Found {len(all_ns_pairs):,} NS-cell pairs (any number of NS per state)")

        df = compute_data(unique_ns, all_ns_pairs, session, session_attn, params)
        df.to_csv(DATA_PATH, index=False)
        print(f"Saved data to {DATA_PATH}")

        if args.puzzle is not None and args.pos is not None:
            pi, sp = args.puzzle, args.pos
            cell, digit = _forced_cell_digit(session.sequences[pi], sp)
        else:
            pool = [s for s in unique_ns if s["pi"] == args.puzzle] \
                   if args.puzzle is not None else unique_ns
            s = (pool or unique_ns)[0]
            pi, sp, cell, digit = s["pi"], s["sp"], s["cell"], s["digit"]
            print(f"Logit lens: puzzle {pi}, pos {sp}  →  R{cell//9+1}C{cell%9+1}={digit}")

    plot(df, pi, sp, cell, digit, session, session_attn, params)


if __name__ == "__main__":
    main()
