"""Cosine similarity of per-cell candidate probes at adjacent steps.

For each step k=0..N-1, trains per-cell×digit probes at step k and k+1,
then reports cos(w_k, w_{k+1}). This reveals whether the probe directions
drift gradually (high adjacent cosine) or jump suddenly.

Usage:
    uv run python scripts/probe_stepwise_similarity.py \
        --cache_path results/3M-backtracking-packing/activations.npz
    uv run python scripts/probe_stepwise_similarity.py --max_step 50 --layer 4
"""

import argparse

import numpy as np
from sklearn.linear_model import LogisticRegression

from sudoku.data import SEP_TOKEN
from sudoku.activations import (
    load_probe_dataset, derive_n_clues, anchor_positions, sequences_to_traces,
)
from sudoku.probes.activations import build_grid_at_step, get_activations_at_positions
from sudoku.probes.targets import _cell_candidates
from sudoku.data_bt import PUSH_TOKEN, POP_TOKEN


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cache_path", default="results/3M-backtracking-packing/activations.npz")
    p.add_argument("--layer", type=int, default=4)
    p.add_argument("--max_step", type=int, default=50)
    p.add_argument("--step_size", type=int, default=1)
    p.add_argument("--no-bt", action="store_true",
                   help="Exclude puzzles whose sequences contain PUSH/POP tokens")
    return p.parse_args()


def extract_step_data(activations, puzzles, sequences, n_clues, step, layer,
                      _cache={}):
    """Extract per-puzzle activations and grids at a given step (cached)."""
    cache_key = (id(activations), step, layer)
    if cache_key in _cache:
        return _cache[cache_key]

    has_sep = any(SEP_TOKEN in seq for seq in sequences[:10])
    anchor = "sep" if has_sep else "last_clue"
    traces = sequences_to_traces(sequences, n_clues)
    ap = anchor_positions(n_clues, anchor)
    seq_len = activations.shape[2]

    valid = np.array([
        len(t) >= step and a + step >= 0
        for t, a in zip(traces, ap)
    ])
    valid_idx = np.where(valid)[0]

    positions_all = [min(ap[i] + step, seq_len - 1) for i in range(len(puzzles))]
    if step == 0 and anchor == "sep":
        grids_all = list(puzzles)
    else:
        grids_all = [
            build_grid_at_step(seq, positions_all[i]) if valid[i] else puzzles[i]
            for i, seq in enumerate(sequences)
        ]

    valid_positions = [positions_all[i] for i in valid_idx]
    acts = get_activations_at_positions(activations, valid_positions, layer, keep=valid_idx)
    grids = [grids_all[i] for i in valid_idx]

    result = (acts, grids, valid_idx)
    _cache[cache_key] = result
    return result


def fit_probes_at_step(activations, puzzles, sequences, n_clues, step, layer):
    """Fit per-cell×digit candidate probes. Returns dict (cell, d) -> (w, bias)."""
    acts, grids, valid_idx = extract_step_data(
        activations, puzzles, sequences, n_clues, step, layer
    )

    probes = {}
    for cell in range(81):
        empty = np.array([g[cell] not in "123456789" for g in grids])
        idx = np.where(empty)[0]
        if len(idx) < 10:
            continue

        Y = np.array([_cell_candidates(grids[i], cell) for i in idx], dtype=np.float32)
        X = acts[idx]

        for d in range(9):
            y_d = Y[:, d].astype(int)
            if len(np.unique(y_d)) < 2:
                continue
            clf = LogisticRegression(C=1.0, max_iter=1000)
            clf.fit(X, y_d)
            probes[(cell, d)] = clf.coef_[0].copy()

    return probes


def main():
    args = parse_args()

    print(f"Loading {args.cache_path} ...")
    activations, puzzles, sequences, n_clues, _ = load_probe_dataset(args.cache_path)
    if n_clues is None:
        n_clues = derive_n_clues(puzzles)

    D = activations.shape[3]
    print(f"  {len(puzzles)} puzzles, d_model={D}, layer={args.layer}")

    # Filter out backtracking traces if requested
    no_bt = getattr(args, "no_bt", False)
    if no_bt:
        bt_tokens = {PUSH_TOKEN, POP_TOKEN}
        keep = [i for i, seq in enumerate(sequences)
                if not any(t in bt_tokens for t in seq)]
        keep = np.array(keep)
        print(f"  --no-bt: keeping {len(keep)}/{len(puzzles)} puzzles without PUSH/POP")
        activations = activations[keep]
        puzzles = [puzzles[i] for i in keep]
        sequences = [sequences[i] for i in keep]
        n_clues = n_clues[keep]

    steps = list(range(0, args.max_step + 1, args.step_size))
    print(f"  Steps: {steps[0]}..{steps[-1]} (step_size={args.step_size}, {len(steps)} steps)")

    # --- Fit probes at each step ---
    all_probes = {}  # step -> {(cell, d) -> w}
    for step in steps:
        print(f"\nFitting probes at step {step}...", end="", flush=True)
        all_probes[step] = fit_probes_at_step(
            activations, puzzles, sequences, n_clues, step, args.layer
        )
        print(f" {len(all_probes[step])} probes")

    # --- Compute adjacent cosine similarities ---
    print(f"\n{'='*70}")
    print(f"Adjacent-step cosine similarity (layer {args.layer})")
    print(f"{'='*70}")

    print(f"\n  {'step_k':>6} {'step_k+1':>8}  {'n_common':>8}  {'cos_mean':>8}  "
          f"{'cos_std':>7}  {'cos_min':>7}  {'cos_max':>7}")
    print(f"  {'-'*6} {'-'*8}  {'-'*8}  {'-'*8}  {'-'*7}  {'-'*7}  {'-'*7}")

    step_pairs = list(zip(steps[:-1], steps[1:]))
    adj_means = []

    for k, k1 in step_pairs:
        common = set(all_probes[k].keys()) & set(all_probes[k1].keys())
        if not common:
            adj_means.append(float("nan"))
            print(f"  {k:>6} {k1:>8}  {0:>8}  {'nan':>8}")
            continue

        cos_vals = []
        for key in sorted(common):
            wa = all_probes[k][key]
            wb = all_probes[k1][key]
            cos_vals.append((wa @ wb) / (np.linalg.norm(wa) * np.linalg.norm(wb)))
        cos_vals = np.array(cos_vals)
        adj_means.append(np.mean(cos_vals))

        print(f"  {k:>6} {k1:>8}  {len(common):>8}  {np.mean(cos_vals):>+8.4f}  "
              f"{np.std(cos_vals):>7.4f}  {np.min(cos_vals):>+7.4f}  {np.max(cos_vals):>+7.4f}")

    # --- Cumulative: cos(w_0, w_k) for each step k ---
    print(f"\n{'='*70}")
    print(f"Cumulative drift from step 0")
    print(f"{'='*70}")

    print(f"\n  {'step':>5}  {'n_common':>8}  {'cos(w0,wk)':>11}  {'std':>7}  "
          f"{'min':>7}  {'max':>7}")
    print(f"  {'-'*5}  {'-'*8}  {'-'*11}  {'-'*7}  {'-'*7}  {'-'*7}")

    for k in steps:
        common = set(all_probes[0].keys()) & set(all_probes[k].keys())
        if not common:
            continue
        cos_vals = []
        for key in sorted(common):
            wa = all_probes[0][key]
            wb = all_probes[k][key]
            cos_vals.append((wa @ wb) / (np.linalg.norm(wa) * np.linalg.norm(wb)))
        cos_vals = np.array(cos_vals)
        print(f"  {k:>5}  {len(common):>8}  {np.mean(cos_vals):>+11.4f}  "
              f"{np.std(cos_vals):>7.4f}  {np.min(cos_vals):>+7.4f}  {np.max(cos_vals):>+7.4f}")

    # --- Product of adjacent cosines vs actual cumulative ---
    print(f"\n{'='*70}")
    print(f"Product of adjacent cosines vs actual cos(w0, wk)")
    print(f"{'='*70}")
    print(f"  If rotations compose, product of adjacent cos should approximate cumulative.")

    cum_product = 1.0
    print(f"\n  {'step':>5}  {'cos(w0,wk)':>11}  {'Π adj cos':>10}  {'ratio':>7}")
    print(f"  {'-'*5}  {'-'*11}  {'-'*10}  {'-'*7}")

    for i, k in enumerate(steps):
        common = set(all_probes[0].keys()) & set(all_probes[k].keys())
        if not common:
            continue
        cos_vals = []
        for key in sorted(common):
            wa = all_probes[0][key]
            wb = all_probes[k][key]
            cos_vals.append((wa @ wb) / (np.linalg.norm(wa) * np.linalg.norm(wb)))
        actual = np.mean(cos_vals)

        if i > 0:
            cum_product *= adj_means[i - 1]

        ratio = cum_product / actual if abs(actual) > 1e-8 else float("nan")
        print(f"  {k:>5}  {actual:>+11.4f}  {cum_product:>+10.4f}  {ratio:>7.3f}")


if __name__ == "__main__":
    main()
