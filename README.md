# Sudoku Residual

This repository contains research code for studying whether transformers trained
on Sudoku solving traces learn explicit board-state and candidate-set structure
in their residual streams.

The project uses small GPT-style causal transformers trained to predict Sudoku
solver traces. The analysis then asks questions such as:

- Can linear probes recover which cells are filled?
- Can probes recover candidate digits for empty cells?
- Do row, column, and box constraints appear as linearly readable features?
- Which attention heads and MLP components write useful Sudoku information into
  the residual stream?


## What Is In This Repo

- A Flax/JAX transformer implementation for Sudoku traces.
- Training, evaluation, activation collection, and probe-fitting scripts.
- A probe framework for cell, candidate, and structure-level linear probes.
- A set of notebooks for the current analysis workflow.

Large experiment artifacts are not included. Checkpoints, activation caches,
generated arrays, plots, datasets, and scratch analyses live locally under paths
such as `results/`, `data/`, and `playground/`. The trace dataset is not included neither, for all of this -- contact me.

## Setup

The project uses Python `>=3.13` and `uv`.

```bash
uv sync --extra probes
```

For TPU environments:

```bash
uv sync --extra probes --extra tpu
```

Run Python commands through `uv run`, for example:

```bash
uv run python -m compileall sudoku scripts
```

## Data

The main current experiments use backtracking traces. Placement tokens encode a
cell and digit as:

```text
token = row * 81 + col * 9 + (digit - 1)
```

Backtracking vocabulary:

```text
0..728  placement tokens
729     END_CLUES_TOKEN
730     PAD_TOKEN
731     PUSH_TOKEN
732     POP_TOKEN
733     SUCCESS_TOKEN
```

Raw trace files are not included in the repository. Once you have
`data/all_traces.bin`, prepare an NPZ dataset with:

```bash
uv run python -m sudoku.data_bt prepare \
  --bin_path data/all_traces.bin \
  --output bt_traces.npz
```

## Running Experiments

Experiment defaults live in `sudoku/default_experiments.py`. Local overrides can
be placed in `experiments_local.py`, which is ignored by git.

Typical batch workflow:

```bash
uv run python scripts/run_training.py --dry-run
uv run python scripts/run_training.py --filter <substring>
uv run python scripts/collect_activations.py --filter <substring>
uv run python scripts/run_probes.py --filter <substring> --mode state_filled --step 0
uv run python scripts/run_eval.py --filter <substring>
```

The batch scripts discover completed runs from `results/*/config.json`.
Activation collection writes a small metadata NPZ plus separate large `.npy`
activation arrays:

```text
results/{name}/
  config.json
  checkpoint/
  train_log.json
  activations.npz
  activations_acts_post_mlp.npy
  activations_acts_post_attn.npy
  eval.txt
```

## Model And Probes

The model in `sudoku/model.py` is a pre-norm GPT-2-style causal transformer
implemented in Flax/JAX. Training is token-budget based and supports packed
training for long backtracking traces.

The probe framework in `sudoku/probes/` includes:

- `filled`: whether a cell is filled.
- `state_filled`: which digit is in a filled cell.
- `candidates`: candidate digits for empty cells.
- `structure`: row/column/box digit-presence features.
- `cell_temporal` and `cell_compare`: cell-specific temporal/comparison probes.

For notebook work, `ProbeSession` is the main entry point:

```python
from sudoku.probes.session import ProbeSession

session = ProbeSession("results/baseline/activations.npz", act_type="post_mlp")
idx = session.index.at_step(0)
acts = session.acts(idx, layer=4)
grids = session.grids(idx)
train_mask, test_mask = session.split(idx)
```

## Notebooks

Tracked notebooks are intended to be the current, cleaned analysis surface. They
expect local activation caches and checkpoints under `results/`.

- `notebooks/mse_cell_candidate_degradation.ipynb`  
  Trains cell candidate probes at the initial board state and measures how
  predictions degrade as the board fills.

- `notebooks/mse_structure_degradation.ipynb`  
  Repeats the degradation analysis for row, column, and box structure probes.

- `notebooks/probe_prediction_viewers.ipynb`  
  Interactive viewer for cell candidate probe predictions, including cosine
  similarity maps between cell-probe weight vectors.

- `notebooks/structure_probe_viewer.ipynb`  
  Interactive viewer for row/column/box candidate-probe predictions.

- `notebooks/activation_patching_substructure.ipynb`  
  Uses substructure probe directions as residual-stream interventions.

- `notebooks/logit_lens_unembedding_viewer.ipynb`  
  Browses layerwise unembedding/logit-lens behavior across puzzle positions.

- `notebooks/head_dla.ipynb`  
  Computes direct logit attribution for attention heads and includes a
  selected-head follow-up workflow for signed DLA, conditional structure splits,
  and head-output probes.