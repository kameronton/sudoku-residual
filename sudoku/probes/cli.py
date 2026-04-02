"""CLI entry point for standalone probe runs."""

import argparse
import os

import numpy as np

from sudoku.data import SEP_TOKEN
from sudoku.activations import (
    load_probe_dataset, derive_n_clues, anchor_positions,
    generate_probe_dataset, sequences_to_traces,
)

from .activations import build_grid_at_step
from .filters import prepare_probe_inputs, _compute_solve_mask
from .loops import run_probe_loop, run_structure_probe_loop, run_cross_step_probe_loop, run_cell_temporal_probe, compare_cell_probes
from .plotting import plot_all_layers, plot_all_layers_per_digit, plot_structure, plot_cross_step, plot_cell_temporal
from .fitting import metric_name_for_mode


def main():
    parser = argparse.ArgumentParser(description="Linear probes on residual stream")
    parser.add_argument("--ckpt_dir", default="checkpoints")
    parser.add_argument("--ckpt_step", type=int)
    parser.add_argument("--traces_path", default=None, help="NPZ file with test split puzzles")
    parser.add_argument("--cache_path", default="probe_acts.npz", help="Path to cache activations + puzzles")
    parser.add_argument("--output", default="probe_accuracies.png")
    parser.add_argument("--n_puzzles", type=int, default=6400)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--no-compress", action="store_true", help="Skip compression when saving probe cache")
    parser.add_argument("--mode", default="state_filled",
                        choices=["filled", "state_filled", "candidates", "structure", "cell_temporal", "cell_compare"])
    parser.add_argument("--cell-idx", type=int, default=None,
                        help="Cell index 0-80 for cell_temporal mode (row*9+col)")
    parser.add_argument("--per-digit", action="store_true", help="Per-digit F1 heatmap (candidates mode only)")
    parser.add_argument("--use-deltas", action="store_true", help="Probe layer-wise deltas (acts[i] - acts[i-1]) instead of cumulative activations")
    parser.add_argument("--step", type=int, default=0, help="Trace step to probe at (0 = SEP/initial board, 1 = after first fill, ...)")
    parser.add_argument("--filter", choices=["all", "solved", "unsolved"], default="all",
                        help="Filter puzzles for both train and eval")
    parser.add_argument("--cross-steps", type=int, default=0,
                        help="Train probe at --step, then evaluate at steps 0..N (0 = disabled)")
    args = parser.parse_args()

    # --- Load data ---
    if os.path.exists(args.cache_path):
        activations, puzzles, sequences, n_clues, _ = load_probe_dataset(args.cache_path)
    else:
        if not args.traces_path:
            raise ValueError("--traces_path required when no cache exists")
        activations, puzzles, sequences, n_clues, _ = generate_probe_dataset(
            ckpt_dir=args.ckpt_dir, ckpt_step=args.ckpt_step,
            traces_path=args.traces_path,
            n_puzzles=args.n_puzzles, batch_size=args.batch_size,
            cache_path=args.cache_path, compress=not args.no_compress,
        )

    if n_clues is None:
        n_clues = derive_n_clues(puzzles)

    # --- Filter puzzles by solve status ---
    filter_keep: np.ndarray | None = None
    if args.filter != "all":
        print(f"Filtering puzzles by solve status: {args.filter}")
        traces = sequences_to_traces(sequences, n_clues)
        solve_mask = _compute_solve_mask(puzzles, traces)
        filter_keep = np.where(solve_mask if args.filter == "solved" else ~solve_mask)[0]
        puzzles = [puzzles[i] for i in filter_keep]
        sequences = [sequences[i] for i in filter_keep]
        n_clues = n_clues[filter_keep]
        print(f"  Kept {len(puzzles)} {args.filter} puzzles")

    # Keep the mmap'd activations for cross-step eval
    activations_for_cross = activations

    # --- Prepare probe inputs ---
    activations, probe_grids, probe_positions, step_keep = prepare_probe_inputs(
        activations, puzzles, sequences, n_clues, args.step,
    )

    if not probe_grids:
        print("No puzzles remaining after filtering.")
        return

    # Compose solve-status filter and step-length filter into a single keep array
    if step_keep is not None and filter_keep is not None:
        final_keep = filter_keep[step_keep]
    elif step_keep is not None:
        final_keep = step_keep
    else:
        final_keep = filter_keep  # may be None

    output = args.output
    if output == "probe_accuracies.png" and args.step > 0:
        output = f"probe_step{args.step}.png"
    if args.use_deltas:
        output = output.replace(".png", "_deltas.png")

    # --- Cell compare probe ---
    if args.mode == "cell_compare":
        if args.cell_idx is None:
            raise ValueError("--cell-idx required for cell_compare mode")
        compare_cell_probes(activations, puzzles, sequences, n_clues,
                            cell_idx=args.cell_idx, step=args.step)
        return

    # --- Cell temporal probe ---
    if args.mode == "cell_temporal":
        if args.cell_idx is None:
            raise ValueError("--cell-idx required for cell_temporal mode")
        cell_idx = args.cell_idx
        if not (0 <= cell_idx <= 80):
            raise ValueError(f"--cell-idx must be 0-80, got {cell_idx}")
        print(f"Running cell temporal probe: cell {cell_idx} (row {cell_idx//9}, col {cell_idx%9}), steps 0-{args.step}")
        filtered_res, full_res = run_cell_temporal_probe(
            activations, puzzles, sequences, n_clues,
            cell_idx=cell_idx, max_step=args.step,
        )
        cell_output = output.replace(".png", f"_cell{cell_idx}_temporal.png")
        if cell_output == output:
            cell_output = f"probe_cell{cell_idx}_temporal.png"
        plot_cell_temporal(filtered_res, full_res, cell_idx=cell_idx, output_path=cell_output)
        return

    # --- Probing loop + plot ---
    if args.mode == "structure":
        all_scores, all_brier_struct = run_structure_probe_loop(
            activations, probe_grids, probe_positions, keep=final_keep, use_deltas=args.use_deltas,
        )
        plot_structure(all_scores, output)
        plot_structure(all_brier_struct, output.replace(".png", "_brier.png"),
                       vmin=0.0, vmax=0.25, cmap="RdYlGn_r")
    else:
        all_accuracies, all_per_digit, all_brier = run_probe_loop(
            activations, probe_grids, probe_positions, args.mode,
            keep=final_keep, use_deltas=args.use_deltas,
        )
        metric = metric_name_for_mode(args.mode)
        if args.per_digit and all_per_digit:
            plot_all_layers_per_digit(all_per_digit, output.replace(".png", "_per_digit.png"))
        else:
            plot_all_layers(all_accuracies, output, metric_name=metric)
        plot_all_layers(all_brier, output.replace(".png", "_brier.png"),
                        metric_name="Brier", vmin=0.0, vmax=0.25, cmap="RdYlGn_r")

    # --- Cross-step evaluation ---
    if args.cross_steps > 0:
        has_sep = any(SEP_TOKEN in seq for seq in sequences[:10])
        anchor = "sep" if has_sep else "last_clue"
        traces_all = sequences_to_traces(sequences, n_clues)
        ap_all = anchor_positions(n_clues, anchor)
        cross_keep_local = [
            i for i, (t, a) in enumerate(zip(traces_all, ap_all))
            if len(t) >= args.step and a + args.step >= 0
        ]
        if len(cross_keep_local) < len(puzzles):
            print(f"Cross-step: filtered to {len(cross_keep_local)}/{len(puzzles)} puzzles with >= {args.step} trace steps (train_step)")

        cross_keep_local_arr = np.array(cross_keep_local)
        cross_keep_abs = filter_keep[cross_keep_local_arr] if filter_keep is not None else cross_keep_local_arr
        puzzles_cs = [puzzles[i] for i in cross_keep_local]
        sequences_cs = [sequences[i] for i in cross_keep_local]
        n_clues_cs = n_clues[cross_keep_local_arr]
        traces_cs = [traces_all[i] for i in cross_keep_local]
        ap_cs = [ap_all[i] for i in cross_keep_local]
        seq_len = activations_for_cross.shape[2]

        steps_info = []
        for k in range(args.cross_steps + 1):
            valid_mask = np.array([len(t) >= k and a + k >= 0
                                   for t, a in zip(traces_cs, ap_cs)])
            positions_k = [min(a + k, seq_len - 1) for a in ap_cs]
            if k == 0 and anchor == "sep":
                grids_k = list(puzzles_cs)
            else:
                grids_k = [
                    build_grid_at_step(seq, pos) if vm else puz
                    for seq, pos, vm, puz in zip(sequences_cs, positions_k, valid_mask, puzzles_cs)
                ]
            steps_info.append((grids_k, positions_k, valid_mask))

        cross_output = output.replace(".png", f"_cross{args.cross_steps}.png")
        auc_cs, brier_cs = run_cross_step_probe_loop(
            activations_for_cross, steps_info, train_step=args.step, mode=args.mode,
            keep=cross_keep_abs,
        )
        plot_cross_step(auc_cs, brier_cs, train_step=args.step, output_path=cross_output)
