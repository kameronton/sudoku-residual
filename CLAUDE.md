# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Research project investigating whether transformers trained on Sudoku solving traces learn internal representations of board state and candidate sets in their residual streams. The pipeline is: raw puzzles → solving traces → tokenized sequences → GPT-2-style model (JAX/Flax) → linear probes on residual stream activations.

## Commands

All commands use `uv run` (project uses `uv` package manager, Python 3.13+).

```bash
uv sync                          # Install deps (add --extra tpu for Colab TPU)

# Data preparation — generates tokenized trace sequences from raw CSV
uv run python data.py --prepare --data_path sudoku-3m.csv --trace_mode constraint --output traces_constraint.npz

# Training
uv run python training.py --traces_path traces_constraint.npz --batch_size 64 --num_tokens 100000000

# Training (TPU-optimized)
uv run python training.py --traces_path traces_constraint.npz --batch_size 512 --dtype bfloat16 --num_tokens 100000000

# Evaluation — autoregressive generation on held-out puzzles
uv run python evaluate.py --ckpt_dir checkpoints --data_path sudoku-3m.csv --n 100

# Probing — linear probes on residual stream activations
uv run python probes.py --ckpt_dir checkpoints --data_path sudoku-3m.csv --n_puzzles 1000
uv run python probes.py --single-cell --ckpt_dir checkpoints   # single cell filled/empty probe
uv run python probes.py --multi-cell --ckpt_dir checkpoints    # multi-cell probe across grid

# Visualization (step-through mode)
uv run python visualize.py --data_path sudoku-3m.csv --index 0 --mode random --step

# Plot training curves
uv run python plot.py train_log.json --tokens_per_step 5184

# Sanity check
uv run python -c "from data import sanity_check; sanity_check()"
```

## File Structure

| File | Purpose |
|---|---|
| `data.py` | Solver, trace generation, tokenization, dataset class |
| `model.py` | GPT-2 transformer (Flax) |
| `training.py` | Training loop, config, logger, evaluation helpers |
| `evaluate.py` | Checkpoint loading, autoregressive generation, puzzle evaluation |
| `probes.py` | Linear probing experiments on residual stream activations |
| `visualize.py` | Terminal-based trace step-through visualization |
| `plot.py` | Loss curve plotting from `train_log.json` |

## Architecture

### Tokenization (`data.py`)

Tokens 0–728 encode fill actions as `row*81 + col*9 + (digit-1)`. Token 729 = `<sep>`, 730 = `<pad>`. Vocab size 731, max sequence length 82.

A sequence is: `[clue_tokens...] <sep> [trace_tokens...]`, padded to 82 with `<pad>`.

Three trace modes control the order empty cells are filled:
- **random**: shuffled order
- **constraint**: causal propagation order from Norvig-style solver (naked singles resolved as they propagate)
- **human**: naked singles → hidden singles → fallback to constraint order

### Model (`model.py`)

GPT-2 architecture with pre-norm (LayerNorm before attention/FFN). Causal masking. Default config: 6 layers, 4 heads, d_model=128, d_ff=512 (~1.4M params). Configurable via `TransformerConfig` dataclass. Supports `return_intermediates=True` for probing.

### Training (`training.py`)

- Token-based budget (not epochs) via `--num_tokens`
- LR schedules: `linear` (1cycle) or `cosine` (warmup + cosine decay), selected via `--schedule_type`
- Loss masking: only tokens after `<sep>` contribute to loss (trace portion, not clues)
- Dataset preloaded to device at startup for zero-copy batch indexing on TPU
- Loss readback deferred to `log_every` intervals to avoid TPU pipeline stalls
- Checkpointing via Orbax with schedule metadata for correct LR resume
- `TrainConfig` dataclass holds all hyperparameters (model, optimizer, schedule, paths)
- Also contains evaluation helpers: `encode_clues`, `evaluate_puzzle`, `_is_consistent`

### Probing (`probes.py`)

Extracts residual stream activations at each layer, trains linear classifiers to predict:
- **Filled mask**: which cells have clues (81-dim binary)
- **Digit values**: what digit is in each cell (81-dim regression or per-cell classification)
- **Candidate sets**: which digits are legal for each empty cell (81×9 binary)

Activation aggregation strategies: SEP token, mean over clues, concat(mean, max, SEP).

Also includes single-cell and multi-cell binary probe experiments that test whether specific cell fill status is linearly decodable from activations at each layer.

### Evaluation (`evaluate.py`)

Loads a checkpoint, autoregressively generates solving traces for puzzles from CSV, and reports per-puzzle and aggregate statistics: cell accuracy, solve rate, consistency errors, clue/fill overwrites.

## Key Design Decisions

- **Token budget over epochs**: training measures progress in tokens processed, not passes over data. This makes runs comparable across dataset sizes.
- **Trace ordering matters**: constraint-guided traces encode causal structure (cell X was filled because cell Y eliminated candidates). This is the primary training mode.
- **Loss masking**: clue tokens are "free" context — the model only needs to predict the solving trace. This focuses capacity on the interesting part.
- **bfloat16 on TPU**: the model supports mixed precision via `--dtype bfloat16` for ~2× throughput on TPU.
