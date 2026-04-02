"""Logit shift decomposition: how mean activation drift affects per-cell candidate probes.

For each cell, trains 9 binary logistic regression probes on layer-4 activations
at step 0 (filtering to puzzles where the cell is empty). Evaluates at step 0
and step 40. Decomposes Brier degradation via the mean activation shift.

Usage:
    uv run python scripts/logit_shift_decomposition.py \
        --cache_path results/3M-backtracking-packing/activations.npz
"""

import argparse

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from sudoku.data import SEP_TOKEN
from sudoku.activations import (
    load_probe_dataset, derive_n_clues, anchor_positions, sequences_to_traces,
)
from sudoku.probes.activations import build_grid_at_step, get_activations_at_positions
from sudoku.probes.targets import _cell_candidates


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cache_path", default="results/3M-backtracking-packing/activations.npz")
    p.add_argument("--layer", type=int, default=4)
    p.add_argument("--train_step", type=int, default=1)
    p.add_argument("--eval_step", type=int, default=40)
    return p.parse_args()


def extract_step_data(activations, puzzles, sequences, n_clues, step, layer):
    """Extract per-puzzle activations and grids at a given step.

    Returns (acts, grids, valid_idx) where:
      acts: (n_valid, D)
      grids: list of 81-char strings, length n_valid
      valid_idx: int array of puzzle indices that are valid at this step
    """
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

    return acts, grids, valid_idx


def main():
    args = parse_args()

    print(f"Loading {args.cache_path} ...")
    activations, puzzles, sequences, n_clues, _ = load_probe_dataset(args.cache_path)
    if n_clues is None:
        n_clues = derive_n_clues(puzzles)

    N = len(puzzles)
    D = activations.shape[3]
    print(f"  {N} puzzles, d_model={D}, layer={args.layer}")

    # --- Extract data at both steps ---
    print(f"\nExtracting step-{args.train_step} data...")
    X0, grids0, valid0 = extract_step_data(
        activations, puzzles, sequences, n_clues, args.train_step, args.layer
    )
    print(f"  {len(valid0)} valid puzzles, acts shape {X0.shape}")

    print(f"Extracting step-{args.eval_step} data...")
    X40, grids40, valid40 = extract_step_data(
        activations, puzzles, sequences, n_clues, args.eval_step, args.layer
    )
    print(f"  {len(valid40)} valid puzzles, acts shape {X40.shape}")

    # --- Global activation shift (over puzzles valid at both steps) ---
    common = np.intersect1d(valid0, valid40)
    # Map global puzzle ids to local indices in X0/X40
    v0_map = {g: i for i, g in enumerate(valid0)}
    v40_map = {g: i for i, g in enumerate(valid40)}
    common_local0 = np.array([v0_map[g] for g in common])
    common_local40 = np.array([v40_map[g] for g in common])

    print(f"\nMean activation at [clues_end]: {X0[common_local0].mean(axis=0)}")

    delta_global = X40[common_local40].mean(axis=0) - X0[common_local0].mean(axis=0)
    print(f"\nGlobal δ (over {len(common)} common puzzles): ||δ||={np.linalg.norm(delta_global):.3f}")
    print(f"  ||mean(X0)||={np.linalg.norm(X0[common_local0].mean(axis=0)):.3f}")
    print(f"  ||mean(X40)||={np.linalg.norm(X40[common_local40].mean(axis=0)):.3f}")
    print(f"  Mean per-sample ||x||: step0={np.mean(np.linalg.norm(X0[common_local0], axis=1)):.3f}, "
          f"step{args.eval_step}={np.mean(np.linalg.norm(X40[common_local40], axis=1)):.3f}")

    # --- Puzzle-level train/test split ---
    puz_train, puz_test = train_test_split(common, test_size=0.2, random_state=42)
    puz_test_set = set(puz_test)
    test_local0 = np.array([v0_map[g] for g in puz_test])
    test_local40 = np.array([v40_map[g] for g in puz_test])
    train_local0 = np.array([v0_map[g] for g in puz_train])

    X0_train = X0[train_local0]
    X0_test = X0[test_local0]
    X40_test = X40[test_local40]
    grids0_train = [grids0[i] for i in train_local0]
    grids0_test = [grids0[i] for i in test_local0]
    grids40_test = [grids40[i] for i in test_local40]

    print(f"\nTrain: {len(puz_train)} puzzles, Test: {len(puz_test)} puzzles")

    # --- Per-cell probing ---
    print(f"\nFitting per-cell probes (81 cells × 9 digits)...")

    # Accumulators: per-cell, per-digit
    brier_s0 = np.full((81, 9), np.nan)
    brier_s40 = np.full((81, 9), np.nan)
    auc_s0 = np.full((81, 9), np.nan)
    auc_s40 = np.full((81, 9), np.nan)
    shift_d = np.full((81, 9), np.nan)
    brier_pred = np.full((81, 9), np.nan)
    n_empty_s0 = np.zeros(81, dtype=int)
    n_empty_s40 = np.zeros(81, dtype=int)

    for cell in range(81):
        print(f"  Cell {cell:2d}/81", end="\r")

        # --- Build targets at step 0 ---
        empty_train = np.array([g[cell] not in "123456789" for g in grids0_train])
        empty_test0 = np.array([g[cell] not in "123456789" for g in grids0_test])
        empty_test40 = np.array([g[cell] not in "123456789" for g in grids40_test])

        n_empty_s0[cell] = empty_test0.sum()
        n_empty_s40[cell] = empty_test40.sum()

        if empty_train.sum() < 10 or empty_test0.sum() < 4:
            continue

        # Candidate targets
        Y_train = np.array([_cell_candidates(grids0_train[i], cell)
                            for i in np.where(empty_train)[0]], dtype=np.float32)
        Y_test0 = np.array([_cell_candidates(grids0_test[i], cell)
                            for i in np.where(empty_test0)[0]], dtype=np.float32)

        Xtr = X0_train[empty_train]
        Xte0 = X0_test[empty_test0]

        # Fit 9 binary probes
        probes = []
        for d in range(9):
            y_d = Y_train[:, d].astype(int)
            if len(np.unique(y_d)) < 2:
                probes.append(None)
                continue
            clf = LogisticRegression(C=1.0, max_iter=1000)
            clf.fit(Xtr, y_d)
            probes.append(clf)

        # Evaluate at step 0
        for d in range(9):
            if probes[d] is None:
                continue
            probas = probes[d].predict_proba(Xte0)[:, 1]
            brier_s0[cell, d] = np.mean((probas - Y_test0[:, d]) ** 2)
            if len(np.unique(Y_test0[:, d])) > 1:
                auc_s0[cell, d] = roc_auc_score(Y_test0[:, d], probas)

        # Evaluate at step 40
        if empty_test40.sum() >= 2:
            Y_test40 = np.array([_cell_candidates(grids40_test[i], cell)
                                 for i in np.where(empty_test40)[0]], dtype=np.float32)
            Xte40 = X40_test[empty_test40]

            for d in range(9):
                if probes[d] is None:
                    continue
                probas = probes[d].predict_proba(Xte40)[:, 1]
                brier_s40[cell, d] = np.mean((probas - Y_test40[:, d]) ** 2)
                if len(np.unique(Y_test40[:, d])) > 1:
                    auc_s40[cell, d] = roc_auc_score(Y_test40[:, d], probas)

        # Logit shift decomposition
        for d in range(9):
            if probes[d] is None:
                continue
            w = probes[d].coef_[0]
            shift_d[cell, d] = w @ delta_global

            # Predicted Brier: apply mean logit shift to step-0 test data
            logits_40 = Xte40 @ probes[d].coef_[0] + probes[d].intercept_[0]
            p40 = 1 / (1 + np.exp(-logits_40))
            p_shifted = 1 / (1 + np.exp(-(logits_40 - shift_d[cell, d])))
            y_true = Y_test40[:, d]
            brier_pred[cell, d] = np.mean((p_shifted - y_true) ** 2) - np.mean((p40 - y_true) ** 2)

    print()

    # --- Summary ---
    valid_cells = ~np.isnan(brier_s0[:, 0])
    n_valid = valid_cells.sum()
    print(f"\n{'='*70}")
    print(f"RESULTS: {n_valid} cells with valid probes, layer {args.layer}")
    print(f"{'='*70}")

    # Per-digit summary (averaged over cells)
    print(f"\n--- Per-digit summary (mean over {n_valid} cells) ---")
    print(f"  {'Digit':>5}  {'AUC_s0':>7}  {'AUC_s40':>8}  {'Brier_s0':>9}  {'Brier_s40':>10}  "
          f"{'ΔBrier_obs':>11}  {'ΔBrier_pred':>12}  {'shift':>7}")
    print(f"  {'-'*5}  {'-'*7}  {'-'*8}  {'-'*9}  {'-'*10}  {'-'*11}  {'-'*12}  {'-'*7}")
    for d in range(9):
        m_auc0 = np.nanmean(auc_s0[:, d])
        m_auc40 = np.nanmean(auc_s40[:, d])
        m_b0 = np.nanmean(brier_s0[:, d])
        m_b40 = np.nanmean(brier_s40[:, d])
        m_db = m_b40 - m_b0
        m_pred = np.nanmean(brier_pred[:, d])
        m_shift = np.nanmean(shift_d[:, d])
        print(f"  {d+1:>5}  {m_auc0:>7.4f}  {m_auc40:>8.4f}  {m_b0:>9.4f}  {m_b40:>10.4f}  "
              f"{m_db:>+11.4f}  {m_pred:>+12.4f}  {m_shift:>+7.3f}")

    # Overall summary
    mean_auc0 = np.nanmean(auc_s0)
    mean_auc40 = np.nanmean(auc_s40)
    mean_b0 = np.nanmean(brier_s0)
    mean_b40 = np.nanmean(brier_s40)
    mean_db_obs = mean_b40 - mean_b0
    mean_db_pred = np.nanmean(brier_pred)
    ratio = mean_db_pred / mean_db_obs if abs(mean_db_obs) > 1e-8 else float("nan")

    print(f"\n--- Overall ---")
    print(f"  Mean AUC:   step0={mean_auc0:.4f}  step{args.eval_step}={mean_auc40:.4f}")
    print(f"  Mean Brier: step0={mean_b0:.4f}  step{args.eval_step}={mean_b40:.4f}")
    print(f"  Mean ΔBrier observed:  {mean_db_obs:+.4f}")
    print(f"  Mean ΔBrier predicted: {mean_db_pred:+.4f}")
    print(f"  Ratio (predicted/observed): {ratio:.2f}")

    # Empty cell counts
    print(f"\n--- Empty cell counts (test set) ---")
    print(f"  Mean empty cells/puzzle at step 0:  {n_empty_s0.sum() / len(puz_test):.1f}")
    print(f"  Mean empty cells/puzzle at step {args.eval_step}: {n_empty_s40.sum() / len(puz_test):.1f}")


if __name__ == "__main__":
    main()
