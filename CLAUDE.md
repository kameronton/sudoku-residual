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
uv run python data.py --prepare --data_path sudoku-3m.csv --trace_mode constraint --output traces.npz --train_frac 0.90 --val_frac 0.05 --test_frac 0.05

# 2. Training — loads train/val splits from NPZ
uv run python training.py --traces_path traces.npz --batch_size 64 --num_tokens 100000000
uv run python training.py --traces_path traces.npz --batch_size 512 --dtype bfloat16 --num_tokens 100000000  # TPU
uv run python training.py --traces_path traces.npz --batch_size 64 --num_tokens 100000000 --full_val  # eval on full val set

# 3. Evaluation — loads test puzzles from NPZ
uv run python evaluate.py --ckpt_dir checkpoints --traces_path traces.npz --n 100
uv run python evaluate.py --ckpt_dir checkpoints --data_path sudoku-3m.csv --n 100  # legacy CSV fallback
uv run python evaluate.py --cache_path probe_acts.npz --quiet  # from cached probe dataset

# Mistake analysis from cached traces
uv run python evaluate.py --mistake-map --cache_path probe_acts.npz
uv run python evaluate.py --mistake-position --cache_path probe_acts.npz

# 4. Activation collection + probing — loads test puzzles from NPZ
uv run python probes.py --ckpt_dir checkpoints --traces_path traces.npz --n_puzzles 1000
uv run python probes.py --ckpt_dir checkpoints --data_path sudoku-3m.csv --n_puzzles 1000  # legacy CSV fallback
uv run python probes.py --step 0 --cache_path probe_acts.npz   # probe at SEP (initial board)
uv run python probes.py --step 10 --cache_path probe_acts.npz  # probe after 10 fills
uv run python probes.py --eval-filter solved --cache_path probe_acts.npz   # train on all, eval on solved
uv run python probes.py --eval-filter unsolved --cache_path probe_acts.npz # train on all, eval on unsolved
uv run python probes.py --step 5 --filter solved --cache_path probe_acts.npz  # only solved puzzles, step 5

# Utilities
uv run python visualize.py --data_path sudoku-3m.csv --index 0 --mode random --step
uv run python plot.py train_log.json --tokens_per_step 5184
uv run python -c "from data import sanity_check; sanity_check()"
```

## File Structure

| File | Purpose |
|---|---|
| `data.py` | Solver, trace generation, tokenization, dataset class, train/val/test splitting |
| `model.py` | GPT-2 transformer (Flax) |
| `training.py` | Training loop, config, logger |
| `evaluate.py` | Checkpoint loading, autoregressive generation, puzzle evaluation |
| `probes.py` | Linear probing experiments on residual stream activations |
| `visualize.py` | Terminal-based trace step-through visualization |
| `plot.py` | Loss curve plotting from `train_log.json` |

## Architecture

### Data format (NPZ)

The NPZ file produced by `data.py --prepare` contains deterministic train/val/test splits:
- `sequences_train`, `sequences_val`, `sequences_test` — tokenized sequences (int16)
- `puzzles_train`, `puzzles_val`, `puzzles_test` — 81-char puzzle strings (U81)
- `sequences` — all sequences concatenated (backward compat)

Split proportions are controlled by `--train_frac`, `--val_frac`, `--test_frac` (default 90/5/5). Shuffle is seeded via `--seed`.

### Tokenization (`data.py`)

Tokens 0-728 encode fill actions as `row*81 + col*9 + (digit-1)`. Token 729 = `<sep>`, 730 = `<pad>`. Vocab size 731, max sequence length 82.

A sequence is: `[clue_tokens...] <sep> [trace_tokens...]`, padded to 82 with `<pad>`.

Two trace modes control the order empty cells are filled:
- **random**: shuffled order
- **constraint**: causal propagation order from Norvig-style solver (naked singles resolved as they propagate)

`SudokuDataset(path, split="train")` loads the appropriate split from the NPZ. Falls back to `sequences` for old files without splits.

### Model (`model.py`)

GPT-2 architecture with pre-norm (LayerNorm before attention/FFN). Causal masking. Default config: 6 layers, 4 heads, d_model=128, d_ff=512 (~1.4M params). Configurable via `TransformerConfig` dataclass. Supports `return_intermediates=True` for probing.

### Training (`training.py`)

- Token-based budget (not epochs) via `--num_tokens`
- Loads `sequences_train` and `sequences_val` directly from NPZ (no runtime splitting)
- LR schedules: `linear` (1cycle) or `cosine` (warmup + cosine decay), selected via `--schedule_type`
- Loss masking: only tokens after `<sep>` contribute to loss (trace portion, not clues)
- Dataset preloaded to device at startup for zero-copy batch indexing on TPU
- Loss readback deferred to `log_every` intervals to avoid TPU pipeline stalls
- `--full_val` evaluates on the entire val set; default samples 10 random batches
- Checkpointing via Orbax with schedule metadata for correct LR resume
- `TrainConfig` dataclass holds all hyperparameters (model, optimizer, schedule, paths)

### Evaluation (`evaluate.py`)

Loads a checkpoint, autoregressively generates solving traces for test puzzles, and reports per-puzzle and aggregate statistics: cell accuracy, solve rate, consistency errors, clue/fill overwrites.

Puzzle source priority: `--traces_path` NPZ (`puzzles_test`) > `--data_path` CSV fallback. Also supports evaluation from cached probe datasets (`--cache_path`) without needing a checkpoint. The `--mistake-map` flag plots a 9x9 heatmap of where first inconsistencies occur across puzzles.

### Probing (`probes.py`)

Extracts residual stream activations at each layer, trains linear classifiers (Ridge regression) to predict per-cell targets:
- **Filled mask** (`--mode filled`): binary filled/empty classification
- **Digit values** (`--mode state_filled`): 9-class digit classification on filled cells
- **Candidate sets** (`--mode candidates`): 9-dim binary candidate vectors on empty cells

Puzzle source priority (when no cache exists): `--traces_path` NPZ (`puzzles_test`) > `--data_path` CSV fallback.

The `--step` flag controls which token position to probe and what ground truth to use:
- `--step 0` (default): probe at the SEP token, ground truth = initial board (clues only)
- `--step N` (N >= 1): probe at sep+N, ground truth = board state after N trace fills

Filtering options:
- `--filter {solved,unsolved}`: restrict both training and evaluation to a puzzle subset
- `--eval-filter {solved,unsolved}`: train on all puzzles, evaluate only on the specified subset

Core functions are split for reuse: `build_probe_targets` (data prep), `fit_probe` (Ridge fit), `eval_probe` (evaluation). `probe_cell` wraps all three with a train/test split.

## Key Design Decisions

- **Deterministic splits**: train/val/test splits are baked into the NPZ at preparation time with a seeded shuffle. Training, evaluation, and probing all use these pre-computed splits — no runtime splitting.
- **Token budget over epochs**: training measures progress in tokens processed, not passes over data. This makes runs comparable across dataset sizes.
- **Trace ordering matters**: constraint-guided traces encode causal structure (cell X was filled because cell Y eliminated candidates). This is the primary training mode.
- **Loss masking**: clue tokens are "free" context — the model only needs to predict the solving trace. This focuses capacity on the interesting part.
- **bfloat16 on TPU**: the model supports mixed precision via `--dtype bfloat16` for ~2x throughput on TPU.
