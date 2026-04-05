"""Probe execution — runs probes over activations using mode strategies.

Public API:
    prepare_probe_inputs   — detect anchor, build grids, filter by step length
    filter_by_solve_status — keep only solved/unsolved puzzles
    probe_cell             — fit + eval one cell, one layer
    probe_structure        — fit + eval one row/col/box, one layer
    probe_layer            — probe all 81 cells for one layer
    run_probe_loop         — probe all layers (batch/CLI use)
    run_structure_probe_loop
    run_cross_step_probe_loop
    run_cell_temporal_probe
    compare_cell_probes
    stack_depth, cells_filled  — sequence statistics
    metric_name_for_mode   — backward-compat helper
"""

import numpy as np
from sklearn.model_selection import train_test_split

from sudoku.data import SEP_TOKEN
from sudoku.data_bt import PUSH_TOKEN, POP_TOKEN
from sudoku.activations import anchor_positions, sequences_to_traces
from sudoku.solver import solve
from sudoku.evaluate import evaluate_puzzle

from .activations import get_activations_at_positions, build_grid_at_step
from .modes import ProbeMode, MODES, STRUCTURE, CandidatesMode


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def prepare_probe_inputs(
    activations: np.ndarray,
    puzzles: list[str],
    sequences: list[list[int]],
    n_clues: np.ndarray,
    step: int,
) -> tuple[np.ndarray, list[str], list[int], np.ndarray | None]:
    """Detect anchor, filter by step length, compute probe grids and positions.

    Returns (activations, probe_grids, probe_positions, keep) where keep is an
    int array of puzzle indices into activations (None if no filtering occurred).
    activations is returned unchanged — callers must pass keep to probe functions.
    """
    has_sep = any(SEP_TOKEN in seq for seq in sequences[:10])
    anchor = "sep" if has_sep else "last_clue"

    traces = sequences_to_traces(sequences, n_clues)
    anchor_pos = anchor_positions(n_clues, anchor)

    keep = [i for i, (t, ap) in enumerate(zip(traces, anchor_pos))
            if len(t) >= step and ap + step >= 0]
    if len(keep) < len(puzzles):
        print(f"Filtered to {len(keep)}/{len(puzzles)} puzzles (need >= {step} trace steps)")
        keep_arr: np.ndarray | None = np.array(keep)
        puzzles = [puzzles[i] for i in keep]
        sequences = [sequences[i] for i in keep]
        anchor_pos = [anchor_pos[i] for i in keep]
    else:
        keep_arr = None

    probe_positions = [ap + step for ap in anchor_pos]

    if step == 0 and anchor == "sep":
        probe_grids = puzzles
    else:
        probe_grids = [build_grid_at_step(seq, pos) for seq, pos in zip(sequences, probe_positions)]

    return activations, probe_grids, probe_positions, keep_arr


def _compute_solve_mask(puzzles: list[str], traces: list) -> np.ndarray:
    """Return boolean array: True for puzzles the model solved correctly."""
    mask = np.zeros(len(puzzles), dtype=bool)
    for i, (puzzle, trace) in enumerate(zip(puzzles, traces)):
        result = solve(puzzle)
        if result is None:
            continue
        stats = evaluate_puzzle(trace, puzzle, result[0], verbose=False)
        mask[i] = stats["puzzle_solved"]
    return mask


def filter_by_solve_status(
    puzzles: list[str],
    sequences: list[list[int]],
    n_clues: np.ndarray,
    status: str,
) -> np.ndarray:
    """Return int index array of puzzles matching status ('solved' or 'unsolved')."""
    traces = sequences_to_traces(sequences, n_clues)
    mask = _compute_solve_mask(puzzles, traces)
    return np.where(mask if status == "solved" else ~mask)[0]


# ---------------------------------------------------------------------------
# Single-probe functions
# ---------------------------------------------------------------------------

def probe_cell(
    acts: np.ndarray,
    puzzles: list[str],
    cell_idx: int,
    mode: ProbeMode | str = "candidates",
) -> tuple[float, float, np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Fit and evaluate a probe for a single cell.

    Returns (auc, brier, y_true, per_digit_auc, per_digit_brier).
    """
    if isinstance(mode, str):
        mode = MODES[mode]

    targets, labels = mode.build_targets(puzzles, cell_idx)
    rel_idx, X, y = mode.prepare_samples(acts, targets, labels)

    nan_per_digit = np.full(9, float("nan")) if isinstance(mode, CandidatesMode) else None
    if len(X) < 4:
        return float("nan"), float("nan"), y, nan_per_digit, None

    idx = np.arange(len(rel_idx))
    idx_train, idx_test = train_test_split(idx, test_size=0.2, random_state=42)
    clf = mode.fit(X[idx_train], y[idx_train])
    return mode.evaluate(clf, X[idx_test], y[idx_test])


def probe_structure(
    acts: np.ndarray, puzzles: list[str], subtype: str, idx: int
) -> tuple[float, float]:
    """Fit and evaluate a structure probe for one row/col/box. Returns (auc, brier)."""
    targets = STRUCTURE.build_targets(puzzles, subtype, idx)
    indices = np.arange(len(puzzles))
    idx_train, idx_test = train_test_split(indices, test_size=0.2, random_state=42)
    clf = STRUCTURE.fit(acts[idx_train], targets[idx_train])
    return STRUCTURE.evaluate(clf, acts[idx_test], targets[idx_test])


# ---------------------------------------------------------------------------
# Layer-level helpers
# ---------------------------------------------------------------------------

def probe_layer(
    activations: np.ndarray,
    grids: list[str],
    positions: list[int],
    mode: ProbeMode | str,
    layer: int,
    *,
    keep: np.ndarray | None = None,
    use_deltas: bool = False,
) -> tuple[list[float], list[np.ndarray] | None, list[float]]:
    """Probe all 81 cells for one layer.

    Returns (accs, per_digit, briers) where per_digit is None if mode doesn't
    produce per-digit results.
    """
    if isinstance(mode, str):
        mode = MODES[mode]
    acts = get_activations_at_positions(activations, positions, layer, keep=keep, use_deltas=use_deltas)
    accs, briers, per_digit = [], [], []
    for cell in range(81):
        auc, brier, _, pda, _ = probe_cell(acts, grids, cell, mode)
        accs.append(auc)
        briers.append(brier)
        if pda is not None:
            per_digit.append(pda)
    return accs, (per_digit if per_digit else None), briers


def probe_structure_layer(
    activations: np.ndarray,
    grids: list[str],
    positions: list[int],
    layer: int,
    *,
    keep: np.ndarray | None = None,
    use_deltas: bool = False,
) -> tuple[dict[str, list[float]], dict[str, list[float]]]:
    """Probe all rows/cols/boxes for one layer. Returns (scores, briers)."""
    acts = get_activations_at_positions(activations, positions, layer, keep=keep, use_deltas=use_deltas)
    scores: dict[str, list[float]] = {"row": [], "col": [], "box": []}
    briers: dict[str, list[float]] = {"row": [], "col": [], "box": []}
    for subtype in ("row", "col", "box"):
        for idx in range(9):
            auc, brier = probe_structure(acts, grids, subtype, idx)
            scores[subtype].append(auc)
            briers[subtype].append(brier)
    return scores, briers


# ---------------------------------------------------------------------------
# Batch loops (CLI / batch runner use)
# ---------------------------------------------------------------------------

def run_probe_loop(
    activations: np.ndarray,
    probe_grids: list[str],
    probe_positions: list[int],
    mode: ProbeMode | str = "state_filled",
    keep: np.ndarray | None = None,
    use_deltas: bool = False,
) -> tuple[dict[int, list[float]], dict[int, np.ndarray], dict[int, list[float]]]:
    """Probe all layers and cells. Returns (all_auc, all_per_digit, all_brier)."""
    if isinstance(mode, str):
        mode = MODES[mode]
    n_layers = activations.shape[1]
    all_accuracies, all_per_digit, all_brier = {}, {}, {}

    for layer in range(n_layers):
        accs, per_digit, briers = probe_layer(
            activations, probe_grids, probe_positions, mode, layer,
            keep=keep, use_deltas=use_deltas,
        )
        avg_auc = np.nanmean(accs)
        avg_brier = np.nanmean(briers)
        print(f"  Layer {layer} | Mean {mode.metric_name.lower()} ({mode.name}): {avg_auc:.3f}  Brier: {avg_brier:.4f}")
        all_accuracies[layer] = accs
        all_brier[layer] = briers
        if per_digit is not None:
            all_per_digit[layer] = np.array(per_digit)

    return all_accuracies, all_per_digit, all_brier


def run_structure_probe_loop(
    activations: np.ndarray,
    probe_grids: list[str],
    probe_positions: list[int],
    keep: np.ndarray | None = None,
    use_deltas: bool = False,
) -> tuple[dict[int, dict[str, list[float]]], dict[int, dict[str, list[float]]]]:
    """Probe structure (rows/cols/boxes) across all layers. Returns (all_auc, all_brier)."""
    n_layers = activations.shape[1]
    all_scores, all_brier = {}, {}

    for layer in range(n_layers):
        scores, briers = probe_structure_layer(
            activations, probe_grids, probe_positions, layer,
            keep=keep, use_deltas=use_deltas,
        )
        for subtype in ("row", "col", "box"):
            print(f"  Layer {layer} | Mean AUC ({subtype}): {np.nanmean(scores[subtype]):.3f}  "
                  f"Brier: {np.nanmean(briers[subtype]):.4f}")
        all_scores[layer] = scores
        all_brier[layer] = briers

    return all_scores, all_brier


# ---------------------------------------------------------------------------
# Advanced analyses
# ---------------------------------------------------------------------------

def run_cross_step_probe_loop(
    activations: np.ndarray,
    steps_info: list[tuple[list[str], list[int], np.ndarray]],
    train_step: int,
    mode: ProbeMode | str,
    keep: np.ndarray | None = None,
) -> tuple[dict[int, dict[int, float]], dict[int, dict[int, float]]]:
    """Train probes at train_step, evaluate at every step in steps_info.

    steps_info: list of (probe_grids, probe_positions, valid_mask) per step.
    keep: int array of puzzle indices into activations (None = use all).
    Returns (auc_by_step, brier_by_step): dict[step_idx][layer] = mean over cells.
    """
    if isinstance(mode, str):
        mode = MODES[mode]

    n_layers = activations.shape[1]
    n_steps = len(steps_info)
    n = len(steps_info[0][0])
    n_valid = [int(vm.sum()) for _, _, vm in steps_info]
    print(f"Cross-step: train_step={train_step}, eval steps 0-{n_steps - 1}, mode={mode.name}, "
          f"{n_layers} layers, {n} puzzles (valid per step: {n_valid})")

    idx_train, idx_test = train_test_split(np.arange(n), test_size=0.2, random_state=42)
    train_grids = steps_info[train_step][0]

    auc_cells = [[[] for _ in range(n_layers)] for _ in range(n_steps)]
    brier_cells = [[[] for _ in range(n_layers)] for _ in range(n_steps)]

    for layer in range(n_layers):
        print(f"\nLayer {layer}/{n_layers - 1} -- extracting activations for {n_steps} steps...")
        acts_by_step = [
            get_activations_at_positions(activations, positions, layer, keep=keep)
            for _, positions, _ in steps_info
        ]

        for cell in range(81):
            print(f"  L{layer} | cell {cell:2d}/81", end="\r")

            targets_tr, labels_tr = mode.build_targets([train_grids[i] for i in idx_train], cell)
            _, X_fit, y_fit = mode.prepare_samples(acts_by_step[train_step][idx_train], targets_tr, labels_tr)

            if len(X_fit) < 4:
                for s in range(n_steps):
                    auc_cells[s][layer].append(float("nan"))
                    brier_cells[s][layer].append(float("nan"))
                continue

            clf = mode.fit(X_fit, y_fit)

            for s, (step_grids, _, valid_mask) in enumerate(steps_info):
                valid_test = idx_test[valid_mask[idx_test]]
                if len(valid_test) < 2:
                    auc_cells[s][layer].append(float("nan"))
                    brier_cells[s][layer].append(float("nan"))
                    continue

                targets_ev, labels_ev = mode.build_targets([step_grids[i] for i in valid_test], cell)
                _, X_ev, y_ev = mode.prepare_samples(acts_by_step[s][valid_test], targets_ev, labels_ev)

                if len(X_ev) < 2:
                    auc_cells[s][layer].append(float("nan"))
                    brier_cells[s][layer].append(float("nan"))
                    continue

                auc, brier, *_ = mode.evaluate(clf, X_ev, y_ev)
                auc_cells[s][layer].append(auc)
                brier_cells[s][layer].append(brier)

        del acts_by_step

        print(f"  L{layer} results:")
        for s in range(n_steps):
            n_test_valid = int(steps_info[s][2][idx_test].sum())
            avg_auc = np.nanmean(auc_cells[s][layer])
            avg_brier = np.nanmean(brier_cells[s][layer])
            marker = " <-- train" if s == train_step else ""
            print(f"    step {s:3d} | AUC {avg_auc:.3f}  Brier {avg_brier:.4f}  (n_test={n_test_valid}){marker}")

    auc_mean = {s: {l: float(np.nanmean(auc_cells[s][l])) for l in range(n_layers)} for s in range(n_steps)}
    brier_mean = {s: {l: float(np.nanmean(brier_cells[s][l])) for l in range(n_layers)} for s in range(n_steps)}
    return auc_mean, brier_mean


CellTemporalResult = dict[int, dict[str, list[float]]]


def run_cell_temporal_probe(
    activations: np.ndarray,
    puzzles: list[str],
    sequences: list[list[int]],
    n_clues: np.ndarray,
    cell_idx: int,
    max_step: int,
) -> tuple[CellTemporalResult, CellTemporalResult]:
    """For a single cell, probe candidate representations at every step from 0 to max_step.

    Returns (filtered_results, full_results):
    - filtered: only puzzles where the cell is still empty at each step
    - full: all puzzles (empty → candidates, filled → zeros)
    Both: {layer: {"auc": [...], "brier": [...]}}
    """
    mode = MODES["candidates"]

    has_sep = any(SEP_TOKEN in seq for seq in sequences[:10])
    anchor = "sep" if has_sep else "last_clue"
    traces = sequences_to_traces(sequences, n_clues)
    ap = anchor_positions(n_clues, anchor)

    n = len(puzzles)
    n_layers = activations.shape[1]
    seq_len = activations.shape[2]

    idx_train, idx_test = train_test_split(np.arange(n), test_size=0.2, random_state=42)

    steps_data = []
    for k in range(max_step + 1):
        valid = np.array([len(t) >= k and a + k >= 0 for t, a in zip(traces, ap)])
        positions = [min(a + k, seq_len - 1) for a in ap]
        if k == 0 and anchor == "sep":
            grids = list(puzzles)
        else:
            grids = [
                build_grid_at_step(seq, pos) if vm else puzzles[i]
                for i, (seq, pos, vm) in enumerate(zip(sequences, positions, valid))
            ]
        empty_at_step = np.array([g[cell_idx] not in "123456789" for g in grids])
        steps_data.append((grids, positions, valid, empty_at_step))

    print(f"Cell {cell_idx} (row {cell_idx // 9}, col {cell_idx % 9}), steps 0-{max_step}:")
    for k, (_, _, valid, empty_at_step) in enumerate(steps_data):
        n_filt = int((valid & empty_at_step).sum())
        n_full = int(valid.sum())
        print(f"  step {k:3d}: filtered n={n_filt:5d}  full n={n_full:5d}"
              + ("  << cell rarely empty here" if n_filt < 50 else ""))

    filtered_results: CellTemporalResult = {l: {"auc": [], "brier": []} for l in range(n_layers)}
    full_results: CellTemporalResult = {l: {"auc": [], "brier": []} for l in range(n_layers)}

    for layer in range(n_layers):
        print(f"\nLayer {layer}/{n_layers - 1}")

        for k, (grids, positions, valid, empty_at_step) in enumerate(steps_data):
            print(f"  step {k}/{max_step}", end="\r")
            acts = get_activations_at_positions(activations, positions, layer)

            # Filtered probe: valid AND cell empty at step k
            filt = valid & empty_at_step
            filt_train = idx_train[filt[idx_train]]
            filt_test = idx_test[filt[idx_test]]

            if len(filt_train) >= 4 and len(filt_test) >= 2:
                targets_tr, labels_tr = mode.build_targets([grids[i] for i in filt_train], cell_idx)
                targets_te, labels_te = mode.build_targets([grids[i] for i in filt_test], cell_idx)
                _, X_tr, y_tr = mode.prepare_samples(acts[filt_train], targets_tr, labels_tr)
                _, X_te, y_te = mode.prepare_samples(acts[filt_test], targets_te, labels_te)
                if len(X_tr) >= 4 and len(X_te) >= 2:
                    clf = mode.fit(X_tr, y_tr)
                    auc, brier, *_ = mode.evaluate(clf, X_te, y_te)
                else:
                    auc, brier = float("nan"), float("nan")
            else:
                auc, brier = float("nan"), float("nan")

            filtered_results[layer]["auc"].append(auc)
            filtered_results[layer]["brier"].append(brier)

            # Full probe: all valid puzzles
            full_train = idx_train[valid[idx_train]]
            full_test = idx_test[valid[idx_test]]

            if len(full_train) >= 4 and len(full_test) >= 2:
                targets_tr, _ = mode.build_targets([grids[i] for i in full_train], cell_idx)
                targets_te, _ = mode.build_targets([grids[i] for i in full_test], cell_idx)
                clf = mode.fit(acts[full_train], targets_tr)
                auc, brier, *_ = mode.evaluate(clf, acts[full_test], targets_te)
            else:
                auc, brier = float("nan"), float("nan")

            full_results[layer]["auc"].append(auc)
            full_results[layer]["brier"].append(brier)

        filt_auc0 = filtered_results[layer]["auc"][0]
        full_auc0 = full_results[layer]["auc"][0]
        filt_str = f"{filt_auc0:.3f}" if not np.isnan(filt_auc0) else "nan (too few/constant targets)"
        full_str = f"{full_auc0:.3f}" if not np.isnan(full_auc0) else "nan"
        print(f"  Layer {layer} done (step 0 filtered AUC: {filt_str}, full AUC: {full_str})")

    return filtered_results, full_results


def compare_cell_probes(
    activations: np.ndarray,
    puzzles: list[str],
    sequences: list[list[int]],
    n_clues: np.ndarray,
    cell_idx: int,
    step: int = 0,
) -> None:
    """Print a per-layer comparison of full vs filtered candidate probes at one step."""
    mode = MODES["candidates"]

    has_sep = any(SEP_TOKEN in seq for seq in sequences[:10])
    anchor = "sep" if has_sep else "last_clue"
    traces = sequences_to_traces(sequences, n_clues)
    ap = anchor_positions(n_clues, anchor)

    n = len(puzzles)
    seq_len = activations.shape[2]
    n_layers = activations.shape[1]

    valid = np.array([len(t) >= step and a + step >= 0 for t, a in zip(traces, ap)])
    positions = [min(a + step, seq_len - 1) for a in ap]
    if step == 0 and anchor == "sep":
        grids = list(puzzles)
    else:
        grids = [
            build_grid_at_step(seq, pos) if vm else puzzles[i]
            for i, (seq, pos, vm) in enumerate(zip(sequences, positions, valid))
        ]
    empty_at_step = np.array([g[cell_idx] not in "123456789" for g in grids])

    idx_train, idx_test = train_test_split(np.arange(n), test_size=0.2, random_state=42)
    full_train  = idx_train[valid[idx_train]]
    full_test   = idx_test[valid[idx_test]]
    filt_train  = idx_train[(valid & empty_at_step)[idx_train]]
    filt_test   = idx_test[(valid & empty_at_step)[idx_test]]

    row, col = divmod(cell_idx, 9)
    print(f"\nCell {cell_idx} (row {row}, col {col}), step {step}")
    print(f"  Full     — train: {len(full_train):5d}  test: {len(full_test):5d}")
    print(f"  Filtered — train: {len(filt_train):5d}  test: {len(filt_test):5d}")
    print(f"\n  {'Layer':<6}  {'Full AUC':>9}  {'Full Brier':>11}  {'Filt AUC':>9}  {'Filt Brier':>11}")
    print(f"  {'-'*6}  {'-'*9}  {'-'*11}  {'-'*9}  {'-'*11}")

    def fmt(v):
        return f"{v:9.4f}" if not np.isnan(v) else "      nan"

    for layer in range(n_layers):
        acts = get_activations_at_positions(activations, positions, layer)

        if len(full_train) >= 4 and len(full_test) >= 2:
            tgt_tr, _ = mode.build_targets([grids[i] for i in full_train], cell_idx)
            tgt_te, _ = mode.build_targets([grids[i] for i in full_test],  cell_idx)
            clf = mode.fit(acts[full_train], tgt_tr)
            full_auc, full_brier, *_ = mode.evaluate(clf, acts[full_test], tgt_te)
        else:
            full_auc, full_brier = float("nan"), float("nan")

        if len(filt_train) >= 4 and len(filt_test) >= 2:
            tgt_tr, lbl_tr = mode.build_targets([grids[i] for i in filt_train], cell_idx)
            tgt_te, lbl_te = mode.build_targets([grids[i] for i in filt_test],  cell_idx)
            _, X_tr, y_tr = mode.prepare_samples(acts[filt_train], tgt_tr, lbl_tr)
            _, X_te, y_te = mode.prepare_samples(acts[filt_test],  tgt_te, lbl_te)
            if len(X_tr) >= 4 and len(X_te) >= 2:
                clf = mode.fit(X_tr, y_tr)
                filt_auc, filt_brier, *_ = mode.evaluate(clf, X_te, y_te)
            else:
                filt_auc, filt_brier = float("nan"), float("nan")
        else:
            filt_auc, filt_brier = float("nan"), float("nan")

        print(f"  {layer:<6}  {fmt(full_auc)}  {fmt(full_brier)}  {fmt(filt_auc)}  {fmt(filt_brier)}")


# ---------------------------------------------------------------------------
# Sequence statistics
# ---------------------------------------------------------------------------

def stack_depth(sequences: list, n_clues: np.ndarray, n_max: int) -> list[float]:
    """Mean backtracking stack depth at each of the first n_max trace steps."""
    steps = [[] for _ in range(n_max)]
    for i, sequence in enumerate(sequences):
        depth = 0
        for index, token in enumerate(sequence[n_clues[i]:n_clues[i] + n_max]):
            if token == PUSH_TOKEN:
                depth += 1
            elif token == POP_TOKEN:
                depth -= 1
            steps[index].append(depth)
    return [sum(s) / len(s) for s in steps]


def cells_filled(sequences: list, n_clues: np.ndarray, n_max: int) -> list[float]:
    """Mean number of filled cells at each of the first n_max trace steps."""
    steps = [[] for _ in range(n_max)]
    for i, sequence in enumerate(sequences):
        for index, step in enumerate(range(n_clues[i], n_clues[i] + n_max)):
            grid = build_grid_at_step(sequence, step)
            steps[index].append(81 - grid.count("0"))
    return [sum(s) / len(s) for s in steps]


# ---------------------------------------------------------------------------
# Backward-compat helpers
# ---------------------------------------------------------------------------

def metric_name_for_mode(mode: ProbeMode | str) -> str:
    if isinstance(mode, str):
        return MODES[mode].metric_name
    return mode.metric_name
