"""High-level probe orchestration loops."""

import numpy as np
from sklearn.model_selection import train_test_split

from sudoku.data import SEP_TOKEN
from sudoku.activations import anchor_positions, sequences_to_traces

from .activations import get_activations_at_positions, build_grid_at_step
from .targets import build_probe_targets, filter_by_mode
from .fitting import fit_probe, eval_probe, probe_cell, probe_structure, metric_name_for_mode


def run_probe_loop(
    activations: np.ndarray,
    probe_grids: list[str],
    probe_positions: list[int],
    mode: str = "state_filled",
    keep: np.ndarray | None = None,
    use_deltas: bool = False,
) -> tuple[dict[int, list[float]], dict[int, np.ndarray], dict[int, list[float]]]:
    """Run probing loop across all layers and cells.

    Returns (all_auc, all_per_digit_auc, all_brier) dicts keyed by layer index.
    """
    n_layers = activations.shape[1]
    all_accuracies = {}
    all_per_digit = {}
    all_brier = {}

    for layer in range(n_layers):
        acts = get_activations_at_positions(activations, probe_positions, layer, keep=keep, use_deltas=use_deltas)
        print(f"\nLayer {layer}, activations shape: {acts.shape}")

        accuracies = []
        briers = []
        per_digit_layer = []
        for cell in range(81):
            print(f"  Layer {layer} | Cell {cell:2d}/81", end="\r")
            metric_val, brier_val, _, per_digit_auc, _ = probe_cell(acts, probe_grids, cell, mode=mode)
            accuracies.append(metric_val)
            briers.append(brier_val)
            if per_digit_auc is not None:
                per_digit_layer.append(per_digit_auc)

        avg_auc = np.nanmean(accuracies)
        avg_brier = np.nanmean(briers)
        print(f"  Layer {layer} | Mean {metric_name_for_mode(mode).lower()} ({mode}): {avg_auc:.3f}  Brier: {avg_brier:.4f}")
        all_accuracies[layer] = accuracies
        all_brier[layer] = briers
        if per_digit_layer:
            all_per_digit[layer] = np.array(per_digit_layer)

    return all_accuracies, all_per_digit, all_brier


def run_structure_probe_loop(
    activations: np.ndarray,
    probe_grids: list[str],
    probe_positions: list[int],
    keep: np.ndarray | None = None,
    use_deltas: bool = False,
) -> tuple[dict[int, dict[str, list[float]]], dict[int, dict[str, list[float]]]]:
    """Run structure probing: 27 probes per layer (9 rows, 9 cols, 9 boxes).

    Returns (all_auc, all_brier), each dict[layer, dict[subtype, list[float]]].
    """
    n_layers = activations.shape[1]
    all_scores = {}
    all_brier = {}
    for layer in range(n_layers):
        acts = get_activations_at_positions(activations, probe_positions, layer, keep=keep, use_deltas=use_deltas)
        print(f"\nLayer {layer}, activations shape: {acts.shape}")
        layer_scores: dict[str, list[float]] = {"row": [], "col": [], "box": []}
        layer_brier: dict[str, list[float]] = {"row": [], "col": [], "box": []}
        for subtype in ("row", "col", "box"):
            for idx in range(9):
                print(f"  Layer {layer} | {subtype} {idx}/9", end="\r")
                auc, brier = probe_structure(acts, probe_grids, subtype, idx)
                layer_scores[subtype].append(auc)
                layer_brier[subtype].append(brier)
        for subtype in ("row", "col", "box"):
            avg_auc = np.nanmean(layer_scores[subtype])
            avg_brier = np.nanmean(layer_brier[subtype])
            print(f"  Layer {layer} | Mean AUC ({subtype}): {avg_auc:.3f}  Brier: {avg_brier:.4f}")
        all_scores[layer] = layer_scores
        all_brier[layer] = layer_brier
    return all_scores, all_brier


def run_cross_step_probe_loop(
    activations: np.ndarray,
    steps_info: list[tuple[list[str], list[int], np.ndarray]],
    train_step: int,
    mode: str,
    keep: np.ndarray | None = None,
) -> tuple[dict[int, dict[int, float]], dict[int, dict[int, float]]]:
    """Train probes at train_step, evaluate at every step in steps_info.

    steps_info: list of (probe_grids, probe_positions, valid_mask) per step.
        All lists have length n_puzzles (the universe of puzzles valid at train_step).
        valid_mask is a boolean array of length n_puzzles; True where the puzzle has
        enough trace steps for that eval step.  Grids/positions for invalid puzzles
        are placeholders and must not be used for targets.
    keep: int array of puzzle indices into activations (None = use all).
    Returns (auc_by_step, brier_by_step): dict[step_idx][layer] = mean over cells.
    """
    n_layers = activations.shape[1]
    n_steps = len(steps_info)

    n = len(steps_info[0][0])
    n_valid = [int(vm.sum()) for _, _, vm in steps_info]
    print(f"Cross-step: train_step={train_step}, eval steps 0-{n_steps - 1}, mode={mode}, "
          f"{n_layers} layers, {n} puzzles (valid per step: {n_valid})")

    # Shared train/test split at puzzle level (over the full universe)
    idx_train, idx_test = train_test_split(np.arange(n), test_size=0.2, random_state=42)
    train_grids = steps_info[train_step][0]

    # Accumulators: [step][layer] -> list of per-cell metrics
    auc_cells = [[[] for _ in range(n_layers)] for _ in range(n_steps)]
    brier_cells = [[[] for _ in range(n_layers)] for _ in range(n_steps)]

    for layer in range(n_layers):
        print(f"\nLayer {layer}/{n_layers - 1} -- extracting activations for {n_steps} steps...")
        # Extract this layer for all steps with a direct fancy index
        acts_by_step = [
            get_activations_at_positions(activations, positions, layer, keep=keep)
            for _, positions, _ in steps_info
        ]

        for cell in range(81):
            print(f"  L{layer} | cell {cell:2d}/81", end="\r")

            # --- Fit on train puzzles at train_step ---
            targets_tr, labels_tr = build_probe_targets(
                [train_grids[i] for i in idx_train], cell, mode
            )
            _, X_fit, y_fit = filter_by_mode(
                acts_by_step[train_step][idx_train], targets_tr, labels_tr, mode
            )

            if len(X_fit) < 4:
                for s in range(n_steps):
                    auc_cells[s][layer].append(float("nan"))
                    brier_cells[s][layer].append(float("nan"))
                continue

            clf = fit_probe(X_fit, y_fit, mode)

            # --- Evaluate at each step on test puzzles ---
            for s, (step_grids, _, valid_mask) in enumerate(steps_info):
                valid_test = idx_test[valid_mask[idx_test]]
                if len(valid_test) < 2:
                    auc_cells[s][layer].append(float("nan"))
                    brier_cells[s][layer].append(float("nan"))
                    continue

                targets_ev, labels_ev = build_probe_targets(
                    [step_grids[i] for i in valid_test], cell, mode
                )
                rel_ev_idx, X_ev, y_ev = filter_by_mode(
                    acts_by_step[s][valid_test], targets_ev, labels_ev, mode
                )

                if len(rel_ev_idx) < 2:
                    auc_cells[s][layer].append(float("nan"))
                    brier_cells[s][layer].append(float("nan"))
                    continue

                auc, brier, _, _, _ = eval_probe(clf, X_ev, y_ev, mode)
                auc_cells[s][layer].append(auc)
                brier_cells[s][layer].append(brier)

        del acts_by_step  # release before next layer

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


# Result type alias for cell temporal probe
# {layer: {"auc": [auc_step0, ...], "brier": [brier_step0, ...]}}
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

    Runs two parallel probe series:
    - **filtered**: at each step k, only uses puzzles where the cell is still empty at step k.
      Ground truth = current candidates given the board state at step k.
    - **full**: at each step k, uses all puzzles with enough trace steps.
      For empty cells, ground truth = current candidates; for filled cells, targets = zeros.

    Both series use the same consistent train/test split over all puzzles.
    The filtered probe tracks "candidate knowledge for unsolved cells"; the full probe
    tracks "does the representation encode candidate info regardless of fill state".

    Returns (filtered_results, full_results), each:
        {layer: {"auc": [float, ...], "brier": [float, ...]}}
    where list indices correspond to steps 0..max_step.
    """
    has_sep = any(SEP_TOKEN in seq for seq in sequences[:10])
    anchor = "sep" if has_sep else "last_clue"

    traces = sequences_to_traces(sequences, n_clues)
    ap = anchor_positions(n_clues, anchor)

    n = len(puzzles)
    n_layers = activations.shape[1]
    seq_len = activations.shape[2]

    # Consistent train/test split across all steps
    idx_train, idx_test = train_test_split(np.arange(n), test_size=0.2, random_state=42)

    # Precompute per-step data: grids, positions, valid mask, empty-at-step mask
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

    # Print per-step sample counts once (before the layer loop)
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

            # --- Filtered probe: valid AND cell empty at step k ---
            filt = valid & empty_at_step
            filt_train = idx_train[filt[idx_train]]
            filt_test = idx_test[filt[idx_test]]

            if len(filt_train) >= 4 and len(filt_test) >= 2:
                targets_tr, labels_tr = build_probe_targets(
                    [grids[i] for i in filt_train], cell_idx, "candidates"
                )
                targets_te, labels_te = build_probe_targets(
                    [grids[i] for i in filt_test], cell_idx, "candidates"
                )
                _, X_tr, y_tr = filter_by_mode(acts[filt_train], targets_tr, labels_tr, "candidates")
                _, X_te, y_te = filter_by_mode(acts[filt_test], targets_te, labels_te, "candidates")
                if len(X_tr) >= 4 and len(X_te) >= 2:
                    clf = fit_probe(X_tr, y_tr, "candidates")
                    auc, brier, _, _, _ = eval_probe(clf, X_te, y_te, "candidates")
                else:
                    auc, brier = float("nan"), float("nan")
            else:
                auc, brier = float("nan"), float("nan")

            filtered_results[layer]["auc"].append(auc)
            filtered_results[layer]["brier"].append(brier)

            # --- Full probe: all valid puzzles (empty → candidates, filled → zeros) ---
            full_train = idx_train[valid[idx_train]]
            full_test = idx_test[valid[idx_test]]

            if len(full_train) >= 4 and len(full_test) >= 2:
                targets_tr, _ = build_probe_targets(
                    [grids[i] for i in full_train], cell_idx, "candidates"
                )
                targets_te, _ = build_probe_targets(
                    [grids[i] for i in full_test], cell_idx, "candidates"
                )
                clf = fit_probe(acts[full_train], targets_tr, "candidates")
                auc, brier, _, _, _ = eval_probe(clf, acts[full_test], targets_te, "candidates")
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
    """Print a per-layer comparison table of full vs filtered candidate probes at one step.

    Full probe   — trained on all puzzles (empty cells get candidate targets, filled get zeros).
    Filtered probe — trained only on puzzles where the cell is still empty at the given step.
    """
    has_sep = any(SEP_TOKEN in seq for seq in sequences[:10])
    anchor = "sep" if has_sep else "last_clue"

    traces = sequences_to_traces(sequences, n_clues)
    ap = anchor_positions(n_clues, anchor)

    n = len(puzzles)
    seq_len = activations.shape[2]
    n_layers = activations.shape[1]

    # Build grids and positions at the requested step
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
    print()
    print(f"  {'Layer':<6}  {'Full AUC':>9}  {'Full Brier':>11}  {'Filt AUC':>9}  {'Filt Brier':>11}")
    print(f"  {'-'*6}  {'-'*9}  {'-'*11}  {'-'*9}  {'-'*11}")

    for layer in range(n_layers):
        acts = get_activations_at_positions(activations, positions, layer)

        # Full probe
        if len(full_train) >= 4 and len(full_test) >= 2:
            tgt_tr, _ = build_probe_targets([grids[i] for i in full_train], cell_idx, "candidates")
            tgt_te, _ = build_probe_targets([grids[i] for i in full_test],  cell_idx, "candidates")
            clf = fit_probe(acts[full_train], tgt_tr, "candidates")
            full_auc, full_brier, _, _, _ = eval_probe(clf, acts[full_test], tgt_te, "candidates")
        else:
            full_auc, full_brier = float("nan"), float("nan")

        # Filtered probe
        if len(filt_train) >= 4 and len(filt_test) >= 2:
            tgt_tr, lbl_tr = build_probe_targets([grids[i] for i in filt_train], cell_idx, "candidates")
            tgt_te, lbl_te = build_probe_targets([grids[i] for i in filt_test],  cell_idx, "candidates")
            _, X_tr, y_tr = filter_by_mode(acts[filt_train], tgt_tr, lbl_tr, "candidates")
            _, X_te, y_te = filter_by_mode(acts[filt_test],  tgt_te, lbl_te, "candidates")
            if len(X_tr) >= 4 and len(X_te) >= 2:
                clf = fit_probe(X_tr, y_tr, "candidates")
                filt_auc, filt_brier, _, _, _ = eval_probe(clf, X_te, y_te, "candidates")
            else:
                filt_auc, filt_brier = float("nan"), float("nan")
        else:
            filt_auc, filt_brier = float("nan"), float("nan")

        def fmt(v):
            return f"{v:9.4f}" if not np.isnan(v) else "      nan"

        print(f"  {layer:<6}  {fmt(full_auc)}  {fmt(full_brier)}  {fmt(filt_auc)}  {fmt(filt_brier)}")
