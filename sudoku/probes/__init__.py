"""Linear probes on residual stream activations."""

from sudoku.probes.modes import (
    ProbeMode,
    StructureMode,
    FilledMode,
    StateFilledMode,
    CandidatesMode,
    MODES,
    STRUCTURE,
    cell_candidates,
)
from sudoku.probes.activations import (
    get_activations_at_positions,
    compute_deltas,
    build_grid_at_step,
)
from sudoku.probes.probing import (
    prepare_probe_inputs,
    filter_by_solve_status,
    probe_cell,
    probe_structure,
    probe_layer,
    probe_structure_layer,
    run_probe_loop,
    run_structure_probe_loop,
    run_cross_step_probe_loop,
    run_cell_temporal_probe,
    compare_cell_probes,
    stack_depth,
    cells_filled,
    metric_name_for_mode,
)
from sudoku.probes.plotting import (
    plot_all_layers,
    plot_all_layers_per_digit,
    plot_structure,
    plot_structure_single_layer,
    plot_cross_step,
    plot_cell_temporal,
)
from sudoku.probes.session import ActivationIndex, ProbeSession
from sudoku.probes.cli import main

__all__ = [
    # session
    "ActivationIndex", "ProbeSession",
    # modes
    "ProbeMode", "StructureMode", "FilledMode", "StateFilledMode", "CandidatesMode",
    "MODES", "STRUCTURE", "cell_candidates",
    # activations
    "get_activations_at_positions", "compute_deltas", "build_grid_at_step",
    # probing
    "prepare_probe_inputs", "filter_by_solve_status",
    "probe_cell", "probe_structure", "probe_layer", "probe_structure_layer",
    "run_probe_loop", "run_structure_probe_loop", "run_cross_step_probe_loop",
    "run_cell_temporal_probe", "compare_cell_probes",
    "stack_depth", "cells_filled", "metric_name_for_mode",
    # plotting
    "plot_all_layers", "plot_all_layers_per_digit", "plot_structure",
    "plot_structure_single_layer", "plot_cross_step", "plot_cell_temporal",
    # cli
    "main",
]
