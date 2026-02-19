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

### 1. Prepare traces

Takes a CSV of puzzles (columns: `puzzle`, `solution`), generates solving traces, tokenizes them, splits into train/val/test, and saves everything to a single NPZ file.

```bash
uv run python data.py --prepare --data_path sudoku-3m.csv --trace_mode constraint --output traces.npz
```

The NPZ contains:
- `sequences_train`, `sequences_val`, `sequences_test` — tokenized sequences for each split
- `puzzles_train`, `puzzles_val`, `puzzles_test` — 81-char puzzle strings for each split
- `sequences` — all sequences concatenated (backward compat)

Key flags:
- `--trace_mode {random,constraint}` — order in which empty cells are filled. `constraint` encodes causal propagation order from a Norvig-style solver.
- `--train_frac 0.90 --val_frac 0.05 --test_frac 0.05` — split proportions (default 90/5/5)
- `--seed 42` — seed for the shuffle before splitting
- `--max_puzzles N` — limit number of puzzles to process
- `--randomize_clues` — randomize the order of clue tokens within each sequence

### 2. Train

Trains a GPT-2 model on the train split. Validates on the val split.

```bash
uv run python training.py --traces_path traces.npz --batch_size 512 --num_tokens 8_000_000_000
```

Training uses a token-based budget (`--num_tokens`) with a tqdm progress bar showing throughput and loss. Train and val splits are loaded directly from the NPZ — no runtime splitting. Use `--resume` to continue from the latest checkpoint in `checkpoints/`.

Key flags:
- `--num_tokens`, `--warmup_tokens`, `--lr`, `--schedule_type {linear,cosine}`
- `--val_every`, `--log_every`, `--num_checkpoints`
- `--full_val` — evaluate on the entire val set (default: 10 random batches)
- `--dtype bfloat16` — mixed precision for ~2x throughput on TPU

### 3. Evaluate

Autoregressively generates solving traces for test puzzles and reports accuracy.

```bash
# From checkpoint + NPZ test split
uv run python evaluate.py --ckpt_dir checkpoints --traces_path traces.npz --n 100

# From checkpoint + CSV (legacy)
uv run python evaluate.py --ckpt_dir checkpoints --data_path sudoku-3m.csv --n 100

# From cached probe dataset (no checkpoint needed)
uv run python evaluate.py --cache_path probe_acts.npz --quiet
```

When `--traces_path` is provided, puzzles are loaded from the `puzzles_test` array in the NPZ. Falls back to `--data_path` (CSV) if no NPZ is given.

Key flags: `--n` (number of puzzles), `--random_sample`, `--temperature`, `--quiet` (summary only), `--batch_size`.

#### Mistake analysis

```bash
# First-mistake heatmap (9x9 grid showing where errors occur)
uv run python evaluate.py --mistake-map --cache_path probe_acts.npz

# First-mistake position distribution (how many steps from end)
uv run python evaluate.py --mistake-position --cache_path probe_acts.npz
```

### 4. Collect activations

Before probing, you need a cache of residual stream activations. This step generates traces for test puzzles, runs them through the model, and saves per-layer activations.

```bash
uv run python probes.py --ckpt_dir checkpoints --traces_path traces.npz --n_puzzles 1000
```

This saves `probe_acts.npz` containing activations, puzzle strings, and token sequences. All subsequent probe commands reuse this cache.

### 5. Probe residual stream

Train linear probes (Ridge regression) on cached activations to test what the model represents internally at each layer.

```bash
# Reuse cached activations (no checkpoint needed)
uv run python probes.py --cache_path probe_acts.npz
```

Probe modes (`--mode`):
- `filled` — binary: is each cell filled or empty?
- `state_filled` — what digit is in each filled cell? (default)
- `candidates` — which digits are legal for each empty cell? (reports F1)

The `--step` flag controls which point in the solving trace to probe:
- `--step 0` (default) — probe at the `<sep>` token; ground truth = initial board (clues only)
- `--step N` (N >= 1) — probe at sep+N; ground truth = board state after N trace fills

```bash
# Probe the initial board state (default)
uv run python probes.py --cache_path probe_acts.npz

# Probe after 10 fills — does the model track the evolving board?
uv run python probes.py --step 10 --cache_path probe_acts.npz
```

Filtering flags:
- `--filter {solved,unsolved}` — restrict both training and evaluation to a subset of puzzles
- `--eval-filter {solved,unsolved}` — train on all puzzles (80/20 split), evaluate only on the specified subset of held-out data

```bash
# Train on all, report accuracy separately for solved vs unsolved
uv run python probes.py --eval-filter solved --cache_path probe_acts.npz
uv run python probes.py --eval-filter unsolved --cache_path probe_acts.npz
```

Other useful flags:
- `--per-digit` — produce a per-digit F1 heatmap (only with `--mode candidates`)
- `--output probe_accuracies.png` — path for the output plot

### Utilities

```bash
# Visualization (step-through mode)
uv run python visualize.py --data_path sudoku-3m.csv --index 0 --mode random --step

# Plot training curves
uv run python plot.py train_log.json --tokens_per_step 5184

# Sanity check
uv run python -c "from data import sanity_check; sanity_check()"
```
