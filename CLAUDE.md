# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Research project investigating whether transformers trained on Sudoku solving traces learn internal representations of board state and candidate sets in their residual streams. The pipeline is: raw puzzles (CSV) → NPZ with train/val/test splits → GPT-2-style model (JAX/Flax) → linear probes on residual stream activations.

## Commands

All commands use `uv run` (project uses `uv` package manager, Python 3.13+).

```bash
uv sync --extra probes           # Install deps (add --extra tpu for Colab TPU)

# 1a. Standard data preparation (no backtracking)
# --solver_bin is required; for constraint mode it uses the binary if it exists, else falls back to Python solver
uv run python -m sudoku.data --prepare --data_path sudoku-3m.csv --trace_mode constraint --solver_bin ./solver --output traces.npz
uv run python -m sudoku.data --prepare --data_path sudoku-3m.csv --trace_mode random --solver_bin ./solver --sep_token --randomize_clues --output traces_random.npz

# 1b. Backtracking trace preparation (from C solver binary output)
uv run python -m sudoku.data_bt prepare --bin_path all_traces.bin --output bt_traces.npz
uv run python -m sudoku.data_bt prepare --bin_path all_traces.bin --output bt_traces.npz --max_seq_len 300 --max_puzzles 100000
uv run python -m sudoku.data_bt check --bin_path all_traces.bin --n 5   # inspect first 5 traces

# 2. Batch pipeline
# scripts/run_training.py reads from EXPERIMENTS list and writes results/*/config.json
# collect_activations, run_probes, run_eval discover experiments from results/*/config.json on disk
uv run python scripts/run_training.py --dry-run          # preview training commands
uv run python scripts/run_training.py                    # train all experiments
uv run python scripts/collect_activations.py --dry-run   # preview activation collection
uv run python scripts/collect_activations.py             # collect activations for all discovered experiments
uv run python scripts/run_probes.py --dry-run            # preview probe runs
uv run python scripts/run_probes.py                      # run probes for all
uv run python scripts/run_eval.py --dry-run              # preview evaluation
uv run python scripts/run_eval.py                        # evaluate all experiments

# All batch runners support:
#   --filter <substr>     select experiments by name substring
#   --name <exact>        select a single experiment by exact name
#   --dry-run             print what would run without executing

# collect_activations.py extra flags:
#   --all-steps           collect activations at every checkpoint step (requires --name)
#   --traces-only         skip activation collection, only generate traces
#   --n_puzzles N         number of puzzles (default 6400)

# run_probes.py extra flags:
#   --mode <mode>         state_filled (default), candidates, filled, structure
#   --step N              probe at sep+N (default 0)
#   --all-steps           run probes at every checkpoint step (requires --name)
#   --per-digit           per-digit F1 heatmap (candidates mode only)
#   --use-deltas          use delta activations (layer differences) instead of raw
#   --act-type <type>     post_mlp (default) or post_attn

# run_eval.py extra flags:
#   --all-steps           evaluate at every checkpoint step (requires --name)

# 3. Standalone (single experiment)
uv run python scripts/training.py --traces_path traces.npz --batch_size 64 --num_tokens 100000000
uv run python -m sudoku.evaluate --cache_path results/baseline/activations.npz --quiet
uv run python -m sudoku.probes --cache_path results/baseline/activations.npz --step 0 --mode state_filled
uv run python -m sudoku.probes --cache_path results/baseline/activations.npz --step 10 --mode candidates
uv run python -m sudoku.probes --cache_path results/baseline/activations.npz --mode state_filled --act-type post_attn

# Mistake analysis from cached traces
uv run python -m sudoku.evaluate --mistake-map --cache_path results/baseline/activations.npz
uv run python -m sudoku.evaluate --mistake-position --cache_path results/baseline/activations.npz

# Utilities
uv run python -m sudoku.visualize --data_path sudoku-3m.csv --index 0 --mode random --step
uv run python scripts/plot.py results/baseline/train_log.json
uv run python -c "from sudoku.data import sanity_check; sanity_check()"
```

## File Structure

**`sudoku/` — Python package (library)**

| File | Purpose |
|---|---|
| `sudoku/solver.py` | Norvig-style constraint propagation + backtracking solver |
| `sudoku/model.py` | GPT-2 transformer (Flax) with optional KV cache for inference |
| `sudoku/data.py` | Trace generation, tokenization, dataset class, train/val/test splitting |
| `sudoku/data_bt.py` | Backtracking trace adapter: reads binary C solver format, produces NPZ |
| `sudoku/activations.py` | Checkpoint loading, trace generation, activation collection, dataset I/O |
| `sudoku/evaluate.py` | Puzzle evaluation, mistake analysis, statistics |
| `sudoku/probes/` | Linear probing subpackage (see below) |
| `sudoku/visualize.py` | Terminal-based trace step-through visualization |
| `sudoku/experiment_config.py` | Experiment definitions, disk discovery, batch runner helpers |
| `sudoku/default_experiments.py` | Default COMMON and EXPERIMENTS definitions (git-committed fallback) |

**`scripts/` — entry-point scripts**

| File | Purpose |
|---|---|
| `scripts/training.py` | Training loop, config, logger |
| `scripts/run_training.py` | Batch training runner (subprocess per experiment) |
| `scripts/collect_activations.py` | Batch activation collection |
| `scripts/run_probes.py` | Batch probe fitting + plotting |
| `scripts/run_eval.py` | Batch evaluation from cached activations |
| `scripts/plot.py` | Loss curve plotting from train_log.json |
| `scripts/analyze_steps.py` | Solve-step position histogram analysis |
| `scripts/analyze_activation_shift.py` | Mean activation drift vs token position |

## Output Layout

All batch runner scripts write to `results/{experiment_name}/`:

```
results/{name}/
  config.json                          # experiment config (written by run_training.py; used for discovery)
  checkpoint/                          # Orbax checkpoint
  train_log.json                       # training log
  activations.npz                      # metadata: puzzles, sequences, n_clues
  activations_acts_post_mlp.npy        # residual stream after each block (n_puzzles, n_layers, seq_len, d_model)
  activations_acts_post_attn.npy       # residual stream after attention, before MLP (same shape)
  eval.txt                             # evaluation summary
  probe_{mode}_step{step}.png          # probe accuracy plots
  steps/{step}/                        # per-checkpoint outputs when using --all-steps
    activations.npz
    activations_acts_post_mlp.npy
    activations_acts_post_attn.npy
    eval.txt
    probe_{mode}_step{step}.png
```

Input data (traces NPZ, CSV) stays in project root. `experiment_config.experiment_dir(name)` returns the path.

## Architecture

### Experiment Config (`experiment_config.py`)

Shared helpers for all batch runners. Experiment definitions live in `default_experiments.py` (committed) or `experiments_local.py` (gitignored, for per-machine overrides):
- `COMMON` — default training hyperparameters (imported from defaults or local override)
- `EXPERIMENTS` — list of `(name, overrides)` tuples
- `experiment_dir(name)` — returns `results/{name}`
- `parse_batch_args()` — common CLI flags: `--dry-run`, `--filter`, `--name`, `--all-steps`, `--traces-only`
- `filter_experiments(filt, name)` — select from EXPERIMENTS list (used by `run_training.py` only)
- `discover_experiments(filt, name)` — discover experiments from `results/*/config.json` on disk (used by downstream runners)
- `list_checkpoint_steps(name)` — return sorted list of all saved checkpoint steps for an experiment
- `resolve_runs(opts)` — resolve `(name, config, ckpt_step, output_dir)` for each run; handles both default (latest checkpoint) and `--all-steps` modes

**Important**: `run_training.py` reads from the `EXPERIMENTS` list. All downstream runners (`collect_activations`, `run_probes`, `run_eval`) discover experiments from `results/*/config.json` on disk via `resolve_runs()` — they do not read the `EXPERIMENTS` list.

### Data format (NPZ)

The NPZ file produced by `data.py --prepare` contains deterministic train/val/test splits:
- `sequences_train`, `sequences_val`, `sequences_test` — tokenized sequences (int16)
- `puzzles_train`, `puzzles_val`, `puzzles_test` — 81-char puzzle strings (U81)
- `n_clues_train`, `n_clues_val`, `n_clues_test` — number of clues per puzzle

Split proportions are controlled by `--train_frac`, `--val_frac`, `--test_frac` (default 90/5/5). Shuffle is seeded via `--seed`.

### Tokenization (`data.py`)

Tokens 0-728 encode fill actions as `row*81 + col*9 + (digit-1)`. Token 729 = `<sep>`, 730 = `<pad>`. Vocab size 731, max sequence length 82.

A sequence is `[clue_tokens...] [<sep>] [trace_tokens...]`, padded to 82 with `<pad>`. The SEP token is optional — include it with `--sep_token` when calling `data.py --prepare` (default: no SEP).

Two trace modes control the order empty cells are filled:
- **random**: shuffled order
- **constraint**: causal propagation order from Norvig-style solver (naked singles resolved as they propagate); uses C solver binary if available, else falls back to Python solver

`data.py --prepare` flags: `--solver_bin` (required), `--trace_mode` (default: `random`), `--sep_token`, `--randomize_clues`, `--max_puzzles`.

### Backtracking tokenization (`data_bt.py`)

Reads the binary format produced by the C solver (one record per puzzle: `uint32 n` followed by `n` little-endian `uint16` events). Placement events use a decimal encoding `100*r + 10*c + d`; the adapter converts these to `r*81 + c*9 + (d-1)` tokens (same as `encode_fill()`).

Extended vocabulary (729 + 4, plus PAD):

| Token | Value | Meaning |
|-------|-------|---------|
| fill action | 0–728 | same as standard vocab |
| `END_CLUES_TOKEN` | 729 | end of given clues (same index as `SEP_TOKEN` — compatible with probe code) |
| `PAD_TOKEN_BT` | 730 | padding (same index as `PAD_TOKEN` — unified padding across formats) |
| `PUSH_TOKEN` | 731 | entering a new search branch |
| `POP_TOKEN` | 732 | backtracking from a failed branch |
| `SUCCESS_TOKEN` | 733 | solution found, always last in trace |
| `VOCAB_SIZE_BT` | 734 | — |

Sequences can be longer than the standard 82 tokens (backtracking adds PUSH/POP/re-fills). `--max_seq_len` defaults to the 99th percentile rounded up to the nearest 50. Traces exceeding the limit are truncated. The NPZ stores `vocab_size` and `max_seq_len` as metadata scalars.

Clue token order is randomized by default in `tokenize_trace` (`randomize_clues=True`), but `data.py --prepare` defaults to unrandomized — pass `--randomize_clues` explicitly.

`SudokuDataset(path, split="train")` loads the appropriate split from the NPZ.

### Model (`model.py`)

GPT-2 architecture with pre-norm (LayerNorm before attention/FFN). Causal masking. Default config: 6 layers, 4 heads, d_model=128, d_ff=512 (~1.4M params). Configurable via `TransformerConfig` dataclass. Supports `return_intermediates=True` for probing and `use_pos_emb=False` to disable positional embeddings.

When `return_intermediates=True`, the model returns `(logits, intermediates_dict)` where `intermediates_dict` maps named keys to `(batch, seq_len, d_model)` arrays:

| Key | Meaning |
|-----|---------|
| `"layer_{i}_post_attn"` | Residual stream after the attention sublayer of block `i` (before MLP) |
| `"layer_{i}_post_mlp"` | Residual stream after the full block `i` (after MLP) |

`ACTIVATION_DESCRIPTORS = ("post_attn", "post_mlp")` in `activations.py` lists the descriptors that are collected and saved during activation collection.

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

### Probing (`sudoku/probes/`)

Subpackage for linear probing experiments on residual stream activations. Split into focused modules:

| Module | Purpose |
|---|---|
| `session.py` | `ProbeSession` + `ActivationIndex` — persistent dataset access with flexible filtering (see below). |
| `modes.py` | `ProbeMode` strategy classes (`FilledMode`, `StateFilledMode`, `CandidatesMode`, `StructureMode`) — each encapsulates target building, sample selection, fitting, and evaluation. |
| `probing.py` | Probe execution: `prepare_probe_inputs`, `filter_by_solve_status`, `probe_cell`, `probe_structure`, `run_probe_loop`, `run_structure_probe_loop`, `run_cross_step_probe_loop`, `run_cell_temporal_probe`, `compare_cell_probes`. |
| `activations.py` | Activation extraction (`get_activations_at_positions`), grid replay (`build_grid_at_step`), deltas. |
| `plotting.py` | All matplotlib visualization (heatmaps, structure plots, cross-step charts, cell temporal). |
| `cli.py` | Standalone CLI entry point (`main()`). |

All public symbols are re-exported from `sudoku.probes.__init__` for backward compatibility.

#### `ProbeSession` and `ActivationIndex` (`session.py`)

`ProbeSession` loads an activations NPZ once and exposes a query interface for ad-hoc probing without re-reading the file. Activations are memory-mapped via the companion `_acts_{act_type}.npy` file. Typical usage (e.g. in a notebook):

```python
from sudoku.probes.session import ProbeSession

session = ProbeSession("results/baseline/activations.npz")                  # default: post_mlp
session_attn = ProbeSession("results/baseline/activations.npz", act_type="post_attn")

# Classic: one sample per puzzle at step 0 (SEP token)
idx = session.index.at_step(0)
acts = session.acts(idx, layer=4)       # (N, d_model) float32
grids = session.grids(idx)              # list of N 81-char board-state strings

# By cell-fill count (useful for BT traces with backtracking)
idx = session.index.where_filled(35).first_per_puzzle()
acts = session.acts(idx, layer=4)

# Arbitrary predicate on board state
acts, grids, puzzle_idx = session.query(
    layer=4, predicate=lambda g: g[40] == "0"
)

# Puzzle-level train/test split (prevents data leakage across positions)
train_mask, test_mask = session.split(idx)
```

`ActivationIndex` is a flat array of `(puzzle_idx, seq_pos, step, n_filled)` tuples covering all trace positions from the anchor onward. Chainable filter methods:
- `.at_step(n)` — keep samples at exactly step n from the anchor
- `.where_filled(n)` — keep samples where exactly n cells are filled (including clues)
- `.first_per_puzzle()` / `.last_per_puzzle()` — deduplicate to one sample per puzzle
- `.filter(mask)` — arbitrary boolean/integer mask

`session.query(layer, step=, n_filled=, predicate=, per_puzzle=, use_deltas=)` chains all filters and returns `(acts, grids, puzzle_idx)` in one call.

`session.split(idx, test_size=0.2)` splits on unique puzzle IDs so that all positions from the same puzzle land in the same partition — preventing leakage when multiple positions per puzzle are present (e.g. after `where_filled()` on BT traces).

Probe modes:
- **Filled mask** (`--mode filled`): binary filled/empty classification
- **Digit values** (`--mode state_filled`): 9-class digit classification on filled cells
- **Candidate sets** (`--mode candidates`): 9-dim binary candidate vectors on empty cells
- **Structure** (`--mode structure`): 27 probes per layer (9 rows × 9 cols × 9 boxes), each predicting which digits are present; reports F1 per substructure
- **Cell temporal** (`--mode cell_temporal`): track a single cell's probe accuracy across trace steps (requires `--cell-idx`)
- **Cell compare** (`--mode cell_compare`): compare probe accuracy for a cell across experiments (requires `--cell-idx`)

The `--step` flag controls which token position to probe and what ground truth to use:
- `--step 0` (default): probe at the SEP token, ground truth = initial board (clues only)
- `--step N` (N >= 1): probe at sep+N, ground truth = board state after N trace fills

The `--act-type` flag selects which activation descriptor to probe (default: `post_mlp`; also `post_attn`).

Anchor mode (sep vs last_clue) is auto-detected from whether sequences contain the SEP token.

`--use-deltas` computes delta activations (layer[i] − layer[i−1]) before probing, to isolate what each layer adds.

To add a new probe mode: subclass `ProbeMode` in `modes.py` (implement `build_targets`, `prepare_samples`, `fit`, `evaluate`) and register it in the `MODES` dict.

## Key Design Decisions

- **Deterministic splits**: train/val/test splits are baked into the NPZ at preparation time with a seeded shuffle. Training, evaluation, and probing all use these pre-computed splits — no runtime splitting.
- **Token budget over epochs**: training measures progress in tokens processed, not passes over data. This makes runs comparable across dataset sizes.
- **Trace ordering matters**: constraint-guided traces encode causal structure (cell X was filled because cell Y eliminated candidates). This is the primary training mode.
- **Loss masking**: clue tokens are "free" context — the model only needs to predict the solving trace. This focuses capacity on the interesting part.
- **bfloat16 on TPU**: the model supports mixed precision via `--dtype bfloat16` for ~2x throughput on TPU.
- **Uncompressed activations**: `collect_activations.py` saves with `compress=False` because `np.savez_compressed` is extremely slow on multi-GB float32 activation arrays.
