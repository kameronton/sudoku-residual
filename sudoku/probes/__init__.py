"""Linear probes on residual stream activations."""

from sudoku.probes.targets import (
    build_probe_targets,
    build_structure_targets,
    filter_by_mode,
)
from sudoku.probes.fitting import (
    fit_probe,
    eval_probe,
    probe_cell,
    probe_structure,
    metric_name_for_mode,
)
from sudoku.probes.activations import (
    get_activations_at_positions,
    compute_deltas,
    build_grid_at_step,
)
from sudoku.probes.filters import prepare_probe_inputs
from sudoku.probes.loops import (
    run_probe_loop,
    run_structure_probe_loop,
    run_cross_step_probe_loop,
    run_cell_temporal_probe,
    compare_cell_probes,
)
from sudoku.probes.plotting import (
    plot_all_layers,
    plot_all_layers_per_digit,
    plot_structure,
    plot_cross_step,
    plot_cell_temporal,
)
from sudoku.probes.cli import main

__all__ = [
    "build_probe_targets",
    "build_structure_targets",
    "filter_by_mode",
    "fit_probe",
    "eval_probe",
    "probe_cell",
    "probe_structure",
    "metric_name_for_mode",
    "get_activations_at_positions",
    "compute_deltas",
    "build_grid_at_step",
    "prepare_probe_inputs",
    "run_probe_loop",
    "run_structure_probe_loop",
    "run_cross_step_probe_loop",
    "run_cell_temporal_probe",
    "compare_cell_probes",
    "plot_all_layers",
    "plot_all_layers_per_digit",
    "plot_structure",
    "plot_cross_step",
    "plot_cell_temporal",
    "main",
]
