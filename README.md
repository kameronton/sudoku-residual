# Sudoku-residual

This is a repo that tests whether a transformer trained to solve sudokus has:
1) a representation of the current state of the board in the residual stream
2) a representation of possible values for each cell in the residual stream

## Results

For a ~12M parameters transformer trained on 8B tokens generated from 3M traces, we can see that it indeed does represent the state of the board and possible cell candidates linearly(!) in the residual stream.
Even though the performance is not excellent (for now, on 1k grids it gets 74.6% cell accuracy, 42.2% puzzle accuracy) there is a strong signal on the separation token:

1. (Sanity check) whether a cell is filled is represented in the residual stream:
![](/imgs/probes_filled.png)

2. The digit a cell is filled with is represented in the residual stream:
![](/imgs/probes_sep.png)

3. The candidates for each cell are represented in the residual stream (average F1-score over 9 digits):
![](/imgs/probes_candidates.png)

### Observations

1. There is a drop in accuracy in the last rows in the grid, consistently across tasks and layers. I cannot explain why that happens for now, but here are a couple of hypotheses:
- This is a data artefact: the clues are always given in the cell order, so for some reason the model doesn't learn to compress the last cells into the `[SEP]` token. If we probe the token right before `[SEP]`, it represents the last cells very well.
- This is because of causal attention, and for this use-case it would be more reasonable to use bi-directional attention on the clues. Anyway, it needs more investigating.

2. The accuracy of representation drops significantly from the middle layers to the last layer. It means that the "world model" is stored in layers 3 to 5, and the last layer is specialized in using this information for the next token generation. This is consistent with the literature, and it's worth looking at what exactly the last layer is doing.

## Setup

```bash
uv sync
```

### Google Colab (TPU)

```bash
!pip install uv
!git clone https://github.com/kameronton/sudoku-residual.git
%cd sudoku-residual
!uv sync --extra tpu
```

Check TPU is visible:

```python
!uv run python -c "import jax; print(jax.devices())"
```

## Pipeline

The full pipeline goes: **CSV** → **NPZ (with train/val/test splits)** → **Training** → **Evaluation / Activation collection / Probes**.

All experiment artifacts are organized under `results/{experiment_name}/`:

```
results/{name}/
  checkpoint/                      # Orbax model checkpoint
  train_log.json                   # training loss log
  activations.npz                  # residual stream activations + sequences
  eval.txt                         # evaluation summary
  probe_{mode}_step{step}.png      # probe accuracy plots
```

### 1. Prepare traces

Takes a CSV of puzzles (columns: `puzzle`, `solution`), generates solving traces, tokenizes them, splits into train/val/test, and saves everything to a single NPZ file.

```bash
uv run python data.py --prepare --data_path sudoku-3m.csv --trace_mode constraint --output traces.npz
```

The NPZ contains:
- `sequences_train`, `sequences_val`, `sequences_test` — tokenized sequences for each split
- `puzzles_train`, `puzzles_val`, `puzzles_test` — 81-char puzzle strings for each split
- `n_clues_train`, `n_clues_val`, `n_clues_test` — number of clues per puzzle

Key flags:
- `--trace_mode {random,constraint}` — order in which empty cells are filled. `constraint` encodes causal propagation order from a Norvig-style solver.
- `--train_frac 0.90 --val_frac 0.05 --test_frac 0.05` — split proportions (default 90/5/5)
- `--seed 42` — seed for the shuffle before splitting
- `--max_puzzles N` — limit number of puzzles to process

### 2. Run experiments

Experiments are defined in `experiment_config.py` — a `COMMON` dict of defaults and an `EXPERIMENTS` list of `(name, overrides)` tuples. All batch runner scripts share this config.

```bash
# Preview what each step will do
uv run python run_training.py --dry-run
uv run python collect_activations.py --dry-run
uv run python run_probes.py --dry-run
uv run python run_eval.py --dry-run

# Run the full pipeline
uv run python run_training.py              # train all experiments
uv run python collect_activations.py       # collect activations for all
uv run python run_probes.py                # fit probes for all
uv run python run_eval.py                  # evaluate all from cached traces

# Filter to a subset of experiments
uv run python run_training.py --filter no_sep
uv run python run_probes.py --filter baseline --mode candidates --step 5
```

### 3. Standalone commands (single experiment)

#### Train

```bash
uv run python training.py --traces_path traces.npz --batch_size 512 --num_tokens 8_000_000_000
```

Training uses a token-based budget (`--num_tokens`). Use `--resume` to continue from the latest checkpoint.

Key flags:
- `--num_tokens`, `--warmup_tokens`, `--lr`, `--schedule_type {linear,cosine}`
- `--eval_every`, `--num_checkpoints`
- `--full_val` — evaluate on the entire val set (default: 10 random batches)
- `--dtype bfloat16` — mixed precision for ~2x throughput on TPU
- `--loss_mask {after_clues,all}` — what tokens contribute to loss
- `--no_pos_emb` — disable positional embeddings

#### Evaluate

```bash
# From checkpoint + NPZ test split
uv run python evaluate.py --ckpt_dir results/baseline/checkpoint --traces_path traces.npz --n 100

# From cached activations (no checkpoint needed)
uv run python evaluate.py --cache_path results/baseline/activations.npz --quiet
```

Key flags: `--n` (number of puzzles), `--temperature`, `--quiet` (summary only), `--batch_size`.

#### Mistake analysis

```bash
uv run python evaluate.py --mistake-map --cache_path results/baseline/activations.npz
uv run python evaluate.py --mistake-position --cache_path results/baseline/activations.npz
```

#### Collect activations

```bash
uv run python probes.py --ckpt_dir results/baseline/checkpoint --traces_path traces.npz --n_puzzles 1000
```

Saves activations NPZ containing per-layer residual stream activations, puzzle strings, and token sequences.

#### Probe residual stream

```bash
uv run python probes.py --cache_path results/baseline/activations.npz --mode state_filled --step 0
uv run python probes.py --cache_path results/baseline/activations.npz --mode candidates --step 10
```

Probe modes (`--mode`):
- `filled` — binary: is each cell filled or empty?
- `state_filled` — what digit is in each filled cell? (default)
- `candidates` — which digits are legal for each empty cell? (reports F1)

The `--step` flag controls which point in the solving trace to probe:
- `--step 0` (default) — probe at the `<sep>` token; ground truth = initial board (clues only)
- `--step N` (N >= 1) — probe at sep+N; ground truth = board state after N trace fills

Filtering: `--filter {solved,unsolved}` restricts both training and evaluation to a subset.

### Utilities

```bash
# Visualization (step-through mode)
uv run python visualize.py --data_path sudoku-3m.csv --index 0 --mode random --step

# Plot training curves
uv run python plot.py results/baseline/train_log.json

# Sanity check
uv run python -c "from data import sanity_check; sanity_check()"
```
