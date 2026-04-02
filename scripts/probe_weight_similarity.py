"""Probe weight rotation analysis between steps.

For each cell×digit, constructs the minimal (planar) rotation mapping wb→wa,
then tests whether that same rotation also aligns probes of other digits in
the same cell, and probes of the same digit in other cells.

Usage:
    uv run python scripts/probe_weight_similarity.py \
        --cache_path results/3M-backtracking-packing/activations.npz
    uv run python scripts/probe_weight_similarity.py --step_a 0 --step_b 40 --layer 4
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


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cache_path", default="results/3M-backtracking-packing/activations.npz")
    p.add_argument("--layer", type=int, default=4)
    p.add_argument("--step_a", type=int, default=0)
    p.add_argument("--step_b", type=int, default=40)
    return p.parse_args()


def extract_step_data(activations, puzzles, sequences, n_clues, step, layer):
    """Extract per-puzzle activations and grids at a given step."""
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


def planar_rotation(u, v):
    """Build the minimal rotation in the (u, v) plane mapping direction u to direction v.

    Returns (e1, e2, cos_theta, sin_theta) where e1, e2 are the orthonormal
    basis of the rotation plane and theta is the rotation angle.

    To apply: R @ x = x + (cos_theta - 1)*(e1 (e1·x) + e2 (e2·x))
                        + sin_theta * (e2 (e1·x) - e1 (e2·x))
    """
    u_hat = u / np.linalg.norm(u)
    v_hat = v / np.linalg.norm(v)
    cos_theta = np.clip(u_hat @ v_hat, -1.0, 1.0)
    # Orthogonal component of v_hat w.r.t. u_hat
    v_perp = v_hat - cos_theta * u_hat
    v_perp_norm = np.linalg.norm(v_perp)
    if v_perp_norm < 1e-12:
        # Vectors are parallel (or anti-parallel), no unique plane
        return u_hat, u_hat, cos_theta, 0.0
    e2 = v_perp / v_perp_norm
    sin_theta = v_perp_norm  # = sin(angle) since u_hat and v_hat are unit
    # Fix sign: sin_theta should be positive for rotation from u to v
    return u_hat, e2, cos_theta, sin_theta


def apply_planar_rotation(e1, e2, cos_theta, sin_theta, x):
    """Apply the planar rotation to vector x. O(D), not O(D^2)."""
    proj1 = e1 @ x
    proj2 = e2 @ x
    return x + (cos_theta - 1) * (proj1 * e1 + proj2 * e2) \
             + sin_theta * (proj1 * e2 - proj2 * e1)


def main():
    args = parse_args()

    print(f"Loading {args.cache_path} ...")
    activations, puzzles, sequences, n_clues, _ = load_probe_dataset(args.cache_path)
    if n_clues is None:
        n_clues = derive_n_clues(puzzles)

    D = activations.shape[3]
    print(f"  {len(puzzles)} puzzles, d_model={D}, layer={args.layer}")

    print(f"\nExtracting step-{args.step_a} data...")
    Xa, grids_a, valid_a = extract_step_data(
        activations, puzzles, sequences, n_clues, args.step_a, args.layer
    )
    print(f"  {len(valid_a)} valid puzzles")

    print(f"Extracting step-{args.step_b} data...")
    Xb, grids_b, valid_b = extract_step_data(
        activations, puzzles, sequences, n_clues, args.step_b, args.layer
    )
    print(f"  {len(valid_b)} valid puzzles")

    # --- Fit per-cell, per-digit probes at both steps ---
    weights_a = {}  # (cell, d) -> wa
    weights_b = {}  # (cell, d) -> wb
    cos_raw = {}    # (cell, d) -> cos(wa, wb)

    print(f"\nFitting probes at step {args.step_a} and step {args.step_b}...")
    for cell in range(81):
        print(f"  Cell {cell:2d}/81", end="\r")

        empty_a = np.array([g[cell] not in "123456789" for g in grids_a])
        idx_a = np.where(empty_a)[0]
        empty_b = np.array([g[cell] not in "123456789" for g in grids_b])
        idx_b = np.where(empty_b)[0]

        if len(idx_a) < 10 or len(idx_b) < 10:
            continue

        Ya = np.array([_cell_candidates(grids_a[i], cell) for i in idx_a], dtype=np.float32)
        Yb = np.array([_cell_candidates(grids_b[i], cell) for i in idx_b], dtype=np.float32)

        for d in range(9):
            ya_d = Ya[:, d].astype(int)
            yb_d = Yb[:, d].astype(int)
            if len(np.unique(ya_d)) < 2 or len(np.unique(yb_d)) < 2:
                continue

            clf_a = LogisticRegression(C=1.0, max_iter=1000)
            clf_a.fit(Xa[idx_a], ya_d)
            clf_b = LogisticRegression(C=1.0, max_iter=1000)
            clf_b.fit(Xb[idx_b], yb_d)

            wa = clf_a.coef_[0]
            wb = clf_b.coef_[0]
            weights_a[(cell, d)] = wa.copy()
            weights_b[(cell, d)] = wb.copy()
            cos_raw[(cell, d)] = (wa @ wb) / (np.linalg.norm(wa) * np.linalg.norm(wb))

    print()
    keys = sorted(weights_a.keys())
    n_probes = len(keys)
    print(f"  {n_probes} valid probes")
    print(f"  Baseline cos(wa, wb): mean={np.mean(list(cos_raw.values())):+.4f}")

    # Build lookup structures
    cells_with_probes = sorted(set(c for c, d in keys))
    digits_with_probes = sorted(set(d for c, d in keys))
    probes_by_cell = {}
    probes_by_digit = {}
    for c, d in keys:
        probes_by_cell.setdefault(c, []).append(d)
        probes_by_digit.setdefault(d, []).append(c)

    # --- For each reference probe, build its planar rotation and test on others ---
    # Results: cos(R_ref @ wb', wa') for same-cell and same-digit probes
    same_cell_improvement = []   # (cos_before, cos_after) per test pair
    same_digit_improvement = []
    diff_both_improvement = []   # different cell AND different digit (control)

    # Per-cell and per-digit aggregates
    cell_improvements = {c: [] for c in cells_with_probes}
    digit_improvements = {d: [] for d in digits_with_probes}

    for ref_cell, ref_d in keys:
        wa_ref = weights_a[(ref_cell, ref_d)]
        wb_ref = weights_b[(ref_cell, ref_d)]

        # Rotation mapping wb_ref direction → wa_ref direction
        e1, e2, ct, st = planar_rotation(wb_ref, wa_ref)
        if abs(st) < 1e-12:
            continue  # degenerate — skip

        # Test on same-cell, different-digit probes
        for d2 in probes_by_cell[ref_cell]:
            if d2 == ref_d:
                continue
            wa2 = weights_a[(ref_cell, d2)]
            wb2 = weights_b[(ref_cell, d2)]
            cos_before = cos_raw[(ref_cell, d2)]
            # Apply rotation to wb2
            wb2_rot = apply_planar_rotation(e1, e2, ct, st, wb2)
            cos_after = (wa2 @ wb2_rot) / (np.linalg.norm(wa2) * np.linalg.norm(wb2_rot))
            same_cell_improvement.append((cos_before, cos_after))
            cell_improvements[ref_cell].append(cos_after - cos_before)

        # Test on same-digit, different-cell probes
        for c2 in probes_by_digit[ref_d]:
            if c2 == ref_cell:
                continue
            wa2 = weights_a[(c2, ref_d)]
            wb2 = weights_b[(c2, ref_d)]
            cos_before = cos_raw[(c2, ref_d)]
            wb2_rot = apply_planar_rotation(e1, e2, ct, st, wb2)
            cos_after = (wa2 @ wb2_rot) / (np.linalg.norm(wa2) * np.linalg.norm(wb2_rot))
            same_digit_improvement.append((cos_before, cos_after))
            digit_improvements[ref_d].append(cos_after - cos_before)

        # Control: different cell AND different digit (sample a few for speed)
        rng = np.random.default_rng(ref_cell * 9 + ref_d)
        control_keys = [(c, d) for c, d in keys if c != ref_cell and d != ref_d]
        if len(control_keys) > 10:
            control_keys = [control_keys[i] for i in rng.choice(len(control_keys), 10, replace=False)]
        for c2, d2 in control_keys:
            wa2 = weights_a[(c2, d2)]
            wb2 = weights_b[(c2, d2)]
            cos_before = cos_raw[(c2, d2)]
            wb2_rot = apply_planar_rotation(e1, e2, ct, st, wb2)
            cos_after = (wa2 @ wb2_rot) / (np.linalg.norm(wa2) * np.linalg.norm(wb2_rot))
            diff_both_improvement.append((cos_before, cos_after))

    # --- Report ---
    def summarize(pairs, label):
        before = np.array([b for b, a in pairs])
        after = np.array([a for b, a in pairs])
        delta = after - before
        print(f"\n  {label} ({len(pairs)} pairs):")
        print(f"    cos before: mean={np.mean(before):+.4f}")
        print(f"    cos after:  mean={np.mean(after):+.4f}")
        print(f"    Δcos:       mean={np.mean(delta):+.4f}  std={np.std(delta):.4f}  "
              f"min={np.min(delta):+.4f}  max={np.max(delta):+.4f}")
        print(f"    Fraction improved: {(delta > 0).mean():.1%}")

    print(f"\n{'='*70}")
    print(f"Planar rotation transfer test")
    print(f"For each probe, build the unique planar rotation R mapping wb→wa,")
    print(f"then apply R to other probes' wb and check alignment with their wa.")
    print(f"{'='*70}")

    summarize(same_cell_improvement, "Same cell, different digit")
    summarize(same_digit_improvement, "Same digit, different cell")
    summarize(diff_both_improvement, "Different cell AND digit (control)")

    # --- Per-digit breakdown ---
    print(f"\n--- Per-digit: mean Δcos when using same-digit rotation from another cell ---")
    print(f"  {'Digit':>5}  {'Δcos':>7}  {'std':>6}  {'n':>7}")
    print(f"  {'-'*5}  {'-'*7}  {'-'*6}  {'-'*7}")
    for d in digits_with_probes:
        vals = digit_improvements[d]
        if vals:
            print(f"  {d+1:>5}  {np.mean(vals):>+7.4f}  {np.std(vals):.4f}  {len(vals):>7}")

    # --- Per-cell breakdown (top/bottom 5) ---
    print(f"\n--- Per-cell: mean Δcos when using same-cell rotation from another digit ---")
    cell_means = {c: np.mean(v) for c, v in cell_improvements.items() if v}
    sorted_cells = sorted(cell_means.items(), key=lambda x: x[1])
    print(f"  {'Cell':>4}  {'row':>3}  {'col':>3}  {'Δcos':>7}")
    print(f"  {'-'*4}  {'-'*3}  {'-'*3}  {'-'*7}")
    for cell, v in sorted_cells[:5]:
        r, c = divmod(cell, 9)
        print(f"  {cell:>4}  {r:>3}  {c:>3}  {v:>+7.4f}")
    print(f"  ...")
    for cell, v in sorted_cells[-5:]:
        r, c = divmod(cell, 9)
        print(f"  {cell:>4}  {r:>3}  {c:>3}  {v:>+7.4f}")
    print(f"  Overall: mean={np.mean(list(cell_means.values())):+.4f}")


if __name__ == "__main__":
    main()
