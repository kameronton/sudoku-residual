# CLAUDE.md

> **Always use `uv run` to execute any Python command in this project.** The project uses `uv` as its package manager (Python 3.13+). Install deps with `uv sync --extra probes` (add `--extra tpu` for TPU).

## Project Overview

Research project: do transformers trained on Sudoku solving traces learn board state / candidate sets in their residual streams? Pipeline: raw puzzles (CSV) → NPZ (tokenized, pre-split) → GPT-2 model (JAX/Flax) → linear probes on activations.

## Pipeline

### 1. Data Preparation

```bash
# Standard traces (constraint or random order)
uv run python -m sudoku.data --prepare --data_path sudoku-3m.csv --trace_mode constraint --solver_bin ./solver --output traces.npz
# Backtracking traces (from C solver binary)
uv run python -m sudoku.data_bt prepare --bin_path all_traces.bin --output bt_traces.npz
```

Key flags: `--trace_mode {constraint,random}`, `--sep_token`, `--randomize_clues`, `--max_puzzles`, `--max_seq_len` (BT only).

### 2. Batch Pipeline

```bash
uv run python scripts/run_training.py          # train all experiments
uv run python scripts/collect_activations.py    # collect activations
uv run python scripts/run_probes.py             # fit probes
uv run python scripts/run_eval.py               # evaluate
```

All support `--dry-run`, `--filter <substr>`, `--name <exact>`. Training reads from `EXPERIMENTS` list; all downstream runners discover from `results/*/config.json` on disk.

Extra flags: `collect_activations` has `--all-steps`, `--traces-only`, `--n_puzzles N`; `run_probes` has `--mode`, `--step N`, `--all-steps`, `--per-digit`, `--use-deltas`, `--act-type`; `run_eval` has `--all-steps`.

### 3. Standalone Commands

```bash
uv run python scripts/training.py --traces_path traces.npz --batch_size 64 --num_tokens 100000000
uv run python -m sudoku.evaluate --cache_path results/baseline/activations.npz --quiet
uv run python -m sudoku.probes --cache_path results/baseline/activations.npz --step 0 --mode state_filled
uv run python -m sudoku.evaluate --mistake-map --cache_path results/baseline/activations.npz
```

## File Structure

| Path | Purpose |
|---|---|
| `sudoku/model.py` | GPT-2 transformer (Flax), KV cache, `return_intermediates` |
| `sudoku/data.py` | Trace generation, tokenization, `SudokuDataset`, splits |
| `sudoku/data_bt.py` | Backtracking trace adapter (C solver binary → NPZ) |
| `sudoku/activations.py` | Checkpoint loading, activation collection, dataset I/O |
| `sudoku/evaluate.py` | Puzzle evaluation, mistake analysis |
| `sudoku/probes/` | Linear probing subpackage (session, modes, plotting, CLI) |
| `sudoku/solver.py` | Norvig constraint propagation + backtracking solver |
| `sudoku/experiment_config.py` | Experiment definitions, discovery, batch runner helpers |
| `sudoku/default_experiments.py` | Default `COMMON` / `EXPERIMENTS` (committed); override via `experiments_local.py` (gitignored) |
| `scripts/training.py` | Training loop with `TrainConfig` |
| `scripts/run_training.py` | Batch training runner |
| `scripts/collect_activations.py` | Batch activation collection |
| `scripts/run_probes.py` | Batch probe fitting + plotting |
| `scripts/run_eval.py` | Batch evaluation |

## Output Layout

```
results/{name}/
  config.json                       # experiment config (discovery key)
  checkpoint/                       # Orbax checkpoint
  train_log.json
  activations.npz                   # metadata: puzzles, sequences, n_clues
  activations_acts_post_mlp.npy     # (n_puzzles, n_layers, seq_len, d_model)
  activations_acts_post_attn.npy    # same shape, after attention before MLP
  eval.txt
  probe_{mode}_step{step}.png
  steps/{step}/...                  # per-checkpoint outputs (--all-steps)
```

## Tokenization

**Standard vocab (731):** Tokens 0-728 = fill actions `row*81 + col*9 + (digit-1)`. 729 = `<sep>`, 730 = `<pad>`. Sequence: `[clues] [<sep>?] [trace] [<pad>...]`, max length 82.

**Backtracking vocab (734):** Same base + 731=`PUSH`, 732=`POP`, 733=`SUCCESS`. Variable length (controlled by `--max_seq_len`).

Two trace modes: **constraint** (causal propagation order) and **random** (shuffled).

## Model

GPT-2 pre-norm, causal masking. Default: 6 layers, 4 heads, d_model=128, d_ff=512 (~1.4M params). `TransformerConfig` dataclass. `return_intermediates=True` yields `layer_{i}_post_attn` and `layer_{i}_post_mlp` activations.

## Training

Token-based budget (`--num_tokens`). LR schedules: `linear` (1cycle) or `cosine`. Loss masking: `after_clues` (default) or `all`. Supports `--dtype bfloat16` for TPU.

## Probing

Probe modes: `state_filled` (digit classification on filled cells), `candidates` (binary candidate vectors on empty cells), `filled` (filled/empty binary), `structure` (row/col/box digit presence), `cell_temporal`, `cell_compare`.

`--step N` controls probe position (0=SEP token). `--act-type {post_mlp,post_attn}`. `--use-deltas` for layer differences.

### ProbeSession (notebook API)

```python
from sudoku.probes.session import ProbeSession
session = ProbeSession("results/baseline/activations.npz")  # or act_type="post_attn"
idx = session.index.at_step(0)
acts = session.acts(idx, layer=4)       # (N, d_model)
grids = session.grids(idx)              # list of 81-char strings
train_mask, test_mask = session.split(idx)  # puzzle-level split
```

`ActivationIndex` chainable filters: `.at_step(n)`, `.where_filled(n)`, `.first_per_puzzle()`, `.last_per_puzzle()`, `.filter(mask)`.

## Key Design Decisions

- **Deterministic splits** baked into NPZ at preparation time (seeded shuffle)
- **Token budget** over epochs for cross-dataset comparability
- **Loss masking** on trace tokens only (clues are free context)
- **Uncompressed activations** (np.savez_compressed is too slow on multi-GB arrays)
