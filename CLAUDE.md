# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Research project investigating whether transformers trained on Sudoku solving traces learn representations of board state and candidate sets in their residual streams. Trains a GPT-2-style model (JAX/Flax) on tokenized Sudoku solving traces, then probes residual stream activations with linear classifiers.

## Commands

All commands use `uv run` (project uses `uv` package manager, Python 3.13+).

```bash
uv sync                          # Install deps

# Data preparation
uv run python data.py --prepare --data_path sudoku-3m.csv --trace_mode constraint --output traces_constraint.npz

# Training
uv run python training.py --traces_path traces_constraint.npz --batch_size 64 --num_tokens 100000000

# Evaluation
uv run python evaluate.py --ckpt_dir checkpoints --data_path sudoku-3m.csv --n 100

# Visualization (step-through mode)
uv run python visualize.py --data_path sudoku-3m.csv --index 0 --mode random --step

# Sanity check
uv run python -c "from data import sanity_check; sanity_check()"
```

## Architecture

**Tokenization** (`data.py`): Tokens 0-728 encode fill actions as `row*81 + col*9 + (digit-1)`. Token 729 = `<sep>`, 730 = `<pad>`. Vocab size 731, max sequence length 82. Three trace modes: random, constraint (causal propagation order), human-like.

**Model** (`model.py`): GPT-2 architecture — 6 layers, 4 heads, d_model=128, d_ff=512. Pre-norm with causal masking. Config via `TransformerConfig` dataclass.

**Training** (`training.py`): Token-based budget (not epochs). Warmup + cosine decay LR schedule. Validation loss computed only on tokens after `<sep>` (trace portion, not clues). `TrainConfig` dataclass holds all hyperparameters. Checkpointing uses Orbax.

**Probing** (`probes.py`): Extracts residual stream activations at each layer, trains linear classifiers to predict cell filled status, digit values, and candidate sets. Uses `SudokuFeatures` dataclass for ground-truth feature extraction.
