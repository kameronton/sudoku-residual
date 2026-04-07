"""CLI entry point for standalone probe runs."""

import argparse
import os

import numpy as np

from sudoku.activations import generate_probe_dataset

from .activations import build_grid_at_step
from .modes import MODES
from .plotting import plot_all_layers, plot_all_layers_per_digit, plot_structure, plot_cross_step, plot_cell_temporal
from .probing import (
    filter_by_solve_status,
    run_probe_loop, run_structure_probe_loop, run_cross_step_probe_loop,
    run_cell_temporal_probe, compare_cell_probes,
)
from .session import ProbeSession


def main():
    parser = argparse.ArgumentParser(description="Linear probes on residual stream")
    parser.add_argument("--ckpt_dir", default="checkpoints")
    parser.add_argument("--ckpt_step", type=int)
    parser.add_argument("--traces_path", default=None, help="NPZ file with test split puzzles")
    parser.add_argument("--cache_path", default="probe_acts.npz", help="Path to activations cache")
    parser.add_argument("--output", default="probe_accuracies.png")
    parser.add_argument("--n_puzzles", type=int, default=6400)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--no-compress", action="store_true")
    parser.add_argument("--mode", default="state_filled",
                        choices=["filled", "state_filled", "candidates", "structure",
                                 "cell_temporal", "cell_compare"])
    parser.add_argument("--cell-idx", type=int, default=None,
                        help="Cell index 0-80 for cell_temporal/cell_compare (row*9+col)")
    parser.add_argument("--per-digit", action="store_true",
                        help="Per-digit F1 heatmap (candidates mode only)")
    parser.add_argument("--use-deltas", action="store_true",
                        help="Probe layer-wise deltas instead of cumulative activations")
    parser.add_argument("--act-type", default="post_mlp",
                        choices=["post_mlp", "post_attn"],
                        help="Which activation type to probe (default: post_mlp)")
    parser.add_argument("--step", type=int, default=0,
                        help="Probe at anchor+step (0 = SEP/initial board)")
    parser.add_argument("--filter", choices=["all", "solved", "unsolved"], default="all")
    parser.add_argument("--cross-steps", type=int, default=0,
                        help="Train at --step, evaluate at steps 0..N (0 = disabled)")
    args = parser.parse_args()

    # --- Generate activations if cache is missing ---
    if not os.path.exists(args.cache_path):
        if not args.traces_path:
            raise ValueError("--traces_path required when no cache exists")
        generate_probe_dataset(
            ckpt_dir=args.ckpt_dir, ckpt_step=args.ckpt_step,
            traces_path=args.traces_path,
            n_puzzles=args.n_puzzles, batch_size=args.batch_size,
            cache_path=args.cache_path, compress=not args.no_compress,
        )

    session = ProbeSession(args.cache_path, act_type=args.act_type)

    # --- Solve status filter: narrow to a subset of puzzle indices ---
    puzzle_subset = None
    if args.filter != "all":
        puzzle_subset = filter_by_solve_status(
            session.puzzles, session.sequences, session.n_clues, args.filter,
        )
        print(f"  Kept {len(puzzle_subset)} {args.filter} puzzles")

    # --- Output path ---
    output = args.output
    if output == "probe_accuracies.png" and args.step > 0:
        output = f"probe_step{args.step}.png"
    if args.use_deltas:
        output = output.replace(".png", "_deltas.png")

    # --- Modes that consume raw session data (not the index) ---
    if args.mode in ("cell_compare", "cell_temporal"):
        if args.cell_idx is None:
            raise ValueError(f"--cell-idx required for {args.mode} mode")
        if not (0 <= args.cell_idx <= 80):
            raise ValueError(f"--cell-idx must be 0-80, got {args.cell_idx}")
        acts, puzzles, sequences, n_clues = _filtered_data(session, puzzle_subset)
        if args.mode == "cell_compare":
            compare_cell_probes(acts, puzzles, sequences, n_clues,
                                cell_idx=args.cell_idx, step=args.step)
        else:
            print(f"Cell temporal: cell {args.cell_idx} "
                  f"(row {args.cell_idx // 9}, col {args.cell_idx % 9}), steps 0-{args.step}")
            filtered_res, full_res = run_cell_temporal_probe(
                acts, puzzles, sequences, n_clues,
                cell_idx=args.cell_idx, max_step=args.step,
            )
            cell_output = output.replace(".png", f"_cell{args.cell_idx}_temporal.png")
            if cell_output == output:
                cell_output = f"probe_cell{args.cell_idx}_temporal.png"
            plot_cell_temporal(filtered_res, full_res, cell_idx=args.cell_idx, output_path=cell_output)
        return

    # --- Build index for standard probe modes ---
    idx = session.index
    if puzzle_subset is not None:
        idx = idx.filter(np.isin(idx.puzzle_idx, puzzle_subset))
    idx = idx.at_step(args.step)

    if len(idx) == 0:
        print("No puzzles remaining after filtering.")
        return

    grids = session.grids(idx)
    positions = idx.seq_pos.tolist()
    keep = idx.puzzle_idx

    # --- Probe loop ---
    if args.mode == "structure":
        all_scores, all_brier = run_structure_probe_loop(
            session.activations, grids, positions,
            keep=keep, use_deltas=args.use_deltas,
        )
        plot_structure(all_scores, output)
        plot_structure(all_brier, output.replace(".png", "_brier.png"),
                       vmin=0.0, vmax=0.25, cmap="RdYlGn_r")
    else:
        mode = MODES[args.mode]
        all_accuracies, all_per_digit, all_brier = run_probe_loop(
            session.activations, grids, positions, mode,
            keep=keep, use_deltas=args.use_deltas,
        )
        if args.per_digit and all_per_digit:
            plot_all_layers_per_digit(all_per_digit, output.replace(".png", "_per_digit.png"))
        else:
            plot_all_layers(all_accuracies, output, metric_name=mode.metric_name)
        plot_all_layers(all_brier, output.replace(".png", "_brier.png"),
                        metric_name="Brier", vmin=0.0, vmax=0.25, cmap="RdYlGn_r")

    # --- Cross-step evaluation ---
    if args.cross_steps > 0:
        _run_cross_steps(session, puzzle_subset, args, output)


def _filtered_data(session: ProbeSession, puzzle_subset):
    """Return (activations, puzzles, sequences, n_clues) filtered to puzzle_subset.

    Slices the activations array so callers that don't accept a `keep` argument
    (run_cell_temporal_probe, compare_cell_probes) work correctly.
    """
    if puzzle_subset is None:
        return session.activations, session.puzzles, session.sequences, session.n_clues
    return (
        session.activations[puzzle_subset],
        [session.puzzles[i] for i in puzzle_subset],
        [session.sequences[i] for i in puzzle_subset],
        session.n_clues[puzzle_subset],
    )


def _run_cross_steps(session: ProbeSession, puzzle_subset, args, output):
    """Train probes at --step, evaluate at every step 0..--cross-steps."""
    base_idx = session.index
    if puzzle_subset is not None:
        base_idx = base_idx.filter(np.isin(base_idx.puzzle_idx, puzzle_subset))

    # Retain only puzzles valid at every step from 0 to max(train_step, cross_steps).
    max_step_needed = max(args.step, args.cross_steps)
    valid_at_all = set(base_idx.at_step(0).puzzle_idx.tolist())
    for k in range(1, max_step_needed + 1):
        valid_at_all &= set(base_idx.at_step(k).puzzle_idx.tolist())

    cs_puzzles = np.array(sorted(valid_at_all), dtype=np.int32)
    if len(cs_puzzles) < 10:
        print(f"Cross-step: only {len(cs_puzzles)} puzzles valid at all steps "
              f"0..{max_step_needed}, skipping.")
        return

    print(f"Cross-step: {len(cs_puzzles)} puzzles valid at all steps 0..{max_step_needed} "
          f"(train_step={args.step})")

    # Build the steps_info list expected by run_cross_step_probe_loop.
    steps_info = []
    for k in range(args.cross_steps + 1):
        idx_k = base_idx.at_step(k)
        pos_map = {int(pi): int(sp) for pi, sp in zip(idx_k.puzzle_idx, idx_k.seq_pos)}
        positions_k = [pos_map[int(pi)] for pi in cs_puzzles]
        grids_k = [build_grid_at_step(session.sequences[int(pi)], pos_map[int(pi)])
                   for pi in cs_puzzles]
        valid_mask = np.ones(len(cs_puzzles), dtype=bool)  # all valid by construction
        steps_info.append((grids_k, positions_k, valid_mask))

    mode = MODES.get(args.mode, MODES["state_filled"])
    auc_cs, brier_cs = run_cross_step_probe_loop(
        session.activations, steps_info, train_step=args.step, mode=mode,
        keep=cs_puzzles,
    )
    cross_output = output.replace(".png", f"_cross{args.cross_steps}.png")
    plot_cross_step(auc_cs, brier_cs, train_step=args.step, output_path=cross_output)
