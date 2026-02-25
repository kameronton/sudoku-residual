# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Research project investigating whether transformers trained on Sudoku solving traces learn internal representations of board state and candidate sets in their residual streams. The pipeline is: raw puzzles (CSV) → NPZ with train/val/test splits → GPT-2-style model (JAX/Flax) → linear probes on residual stream activations.

## Commands

All commands use `uv run` (project uses `uv` package manager, Python 3.13+).

```bash
uv sync                          # Install deps (add --extra tpu for Colab TPU)

# 1. Data preparation — generates tokenized traces with train/val/test splits
uv run python data.py --prepare --data_path sudoku-3m.csv --trace_mode constraint --output traces.npz

# 2. Batch pipeline — all experiments defined in experiment_config.py
uv run python run_training.py --dry-run          # preview training commands
uv run python run_training.py                    # train all experiments
uv run python collect_activations.py --dry-run   # preview activation collection
uv run python collect_activations.py             # collect activations for all
uv run python run_probes.py --dry-run            # preview probe runs
uv run python run_probes.py                      # run probes for all
uv run python run_eval.py --dry-run              # preview evaluation
uv run python run_eval.py                        # evaluate all experiments

# All batch runners support: --filter <substr> to select experiments by name

# 3. Standalone (single experiment)
uv run python training.py --traces_path traces.npz --batch_size 64 --num_tokens 100000000
uv run python evaluate.py --cache_path results/baseline/activations.npz --quiet
uv run python probes.py --cache_path results/baseline/activations.npz --step 0 --mode state_filled
uv run python probes.py --cache_path results/baseline/activations.npz --step 10 --mode candidates

# Mistake analysis from cached traces
uv run python evaluate.py --mistake-map --cache_path results/baseline/activations.npz
uv run python evaluate.py --mistake-position --cache_path results/baseline/activations.npz

# Utilities
uv run python visualize.py --data_path sudoku-3m.csv --index 0 --mode random --step
uv run python plot.py results/baseline/train_log.json
uv run python -c "from data import sanity_check; sanity_check()"
```

## File Structure

| File | Purpose |
|---|---|
| `solver.py` | Norvig-style constraint propagation + backtracking solver |
| `data.py` | Trace generation, tokenization, dataset class, train/val/test splitting |
| `model.py` | GPT-2 transformer (Flax) |
| `training.py` | Training loop, config, logger |
| `evaluate.py` | Puzzle evaluation, mistake analysis, statistics |
| `activations.py` | Checkpoint loading, trace generation, activation collection, dataset I/O |
| `probes.py` | Linear probing experiments on residual stream activations |
| `experiment_config.py` | Shared experiment definitions (COMMON defaults, EXPERIMENTS list) |
| `run_training.py` | Batch training runner (subprocess per experiment) |
| `collect_activations.py` | Batch activation collection |
| `run_probes.py` | Batch probe fitting + plotting |
| `run_eval.py` | Batch evaluation from cached activations |
| `visualize.py` | Terminal-based trace step-through visualization |
| `plot.py` | Loss curve plotting from train_log.json |

## Output Layout

All batch runner scripts write to `results/{experiment_name}/`:

```
results/{name}/
  checkpoint/                      # Orbax checkpoint
  train_log.json                   # training log
  activations.npz                  # residual stream activations + sequences
  eval.txt                         # evaluation summary
  probe_{mode}_step{step}.png      # probe accuracy plots
```

Input data (traces NPZ, CSV) stays in project root. `experiment_config.experiment_dir(name)` returns the path.

## Architecture

### Experiment Config (`experiment_config.py`)

Central experiment definitions shared by all batch runners:
- `COMMON` — default training hyperparameters
- `EXPERIMENTS` — list of `(name, overrides)` tuples
- `experiment_dir(name)` — returns `results/{name}`
- `parse_batch_args()` — common CLI flags: `--dry-run`, `--filter`
- `filter_experiments(filt)` — select experiments by name substring

### Data format (NPZ)

The NPZ file produced by `data.py --prepare` contains deterministic train/val/test splits:
- `sequences_train`, `sequences_val`, `sequences_test` — tokenized sequences (int16)
- `puzzles_train`, `puzzles_val`, `puzzles_test` — 81-char puzzle strings (U81)
- `n_clues_train`, `n_clues_val`, `n_clues_test` — number of clues per puzzle

Split proportions are controlled by `--train_frac`, `--val_frac`, `--test_frac` (default 90/5/5). Shuffle is seeded via `--seed`.

### Tokenization (`data.py`)

Tokens 0-728 encode fill actions as `row*81 + col*9 + (digit-1)`. Token 729 = `<sep>`, 730 = `<pad>`. Vocab size 731, max sequence length 82.

A sequence is: `[clue_tokens...] <sep> [trace_tokens...]`, padded to 82 with `<pad>`.

Two trace modes control the order empty cells are filled:
- **random**: shuffled order
- **constraint**: causal propagation order from Norvig-style solver (naked singles resolved as they propagate)

Clue token order is randomized by default (`randomize_clues=True` in `tokenize_trace`).

`SudokuDataset(path, split="train")` loads the appropriate split from the NPZ.

### Model (`model.py`)

GPT-2 architecture with pre-norm (LayerNorm before attention/FFN). Causal masking. Default config: 6 layers, 4 heads, d_model=128, d_ff=512 (~1.4M params). Configurable via `TransformerConfig` dataclass. Supports `return_intermediates=True` for probing and `use_pos_emb=False` to disable positional embeddings.

### Training (`training.py`)

- Token-based budget via `--num_tokens`, epoch-based sequential iteration with reshuffling
- Loads `sequences_train` and `sequences_val` directly from NPZ
- LR schedules: `linear` (1cycle) or `cosine` (warmup + cosine decay), selected via `--schedule_type`
- Loss masking: `after_clues` (default, only trace tokens) or `all` (all non-pad tokens)
- Dataset preloaded to device at startup for zero-copy batch indexing on TPU
- Fused eval: `--eval_every` logs both train and val loss together; loss readback only at eval points
- `--full_val` evaluates on the entire val set; default samples 10 random batches
- Checkpointing via Orbax with schedule metadata for correct LR resume
- `TrainConfig` dataclass holds all hyperparameters (model, optimizer, schedule, paths)

### Evaluation (`evaluate.py`)

Evaluates cached traces against ground-truth solutions, reporting per-puzzle and aggregate statistics: cell accuracy, solve rate, consistency errors, clue/fill overwrites. Requires `--cache_path` (activations NPZ). The `--mistake-map` flag plots a 9x9 heatmap of where first inconsistencies occur.

### Probing (`probes.py`)

Extracts residual stream activations at each layer, trains linear classifiers (Ridge regression) to predict per-cell targets:
- **Filled mask** (`--mode filled`): binary filled/empty classification
- **Digit values** (`--mode state_filled`): 9-class digit classification on filled cells
- **Candidate sets** (`--mode candidates`): 9-dim binary candidate vectors on empty cells

The `--step` flag controls which token position to probe and what ground truth to use:
- `--step 0` (default): probe at the SEP token, ground truth = initial board (clues only)
- `--step N` (N >= 1): probe at sep+N, ground truth = board state after N trace fills

Anchor mode (sep vs last_clue) is auto-detected from whether sequences contain the SEP token.

Core functions: `generate_probe_dataset()` (activation collection), `run_probe_loop()` (probe fitting), `build_probe_targets()` (data prep), `fit_probe()` (Ridge fit), `eval_probe()` (evaluation).

## Key Design Decisions

- **Deterministic splits**: train/val/test splits are baked into the NPZ at preparation time with a seeded shuffle. Training, evaluation, and probing all use these pre-computed splits — no runtime splitting.
- **Token budget over epochs**: training measures progress in tokens processed, not passes over data. This makes runs comparable across dataset sizes.
- **Trace ordering matters**: constraint-guided traces encode causal structure (cell X was filled because cell Y eliminated candidates). This is the primary training mode.
- **Loss masking**: clue tokens are "free" context — the model only needs to predict the solving trace. This focuses capacity on the interesting part.
- **bfloat16 on TPU**: the model supports mixed precision via `--dtype bfloat16` for ~2x throughput on TPU.
- **Uncompressed activations**: `collect_activations.py` saves with `compress=False` because `np.savez_compressed` is extremely slow on multi-GB float32 activation arrays.
