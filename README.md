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

## Usage

### 1. Prepare traces (one-time)

```bash
uv run python data.py --prepare --data_path sudoku-3m.csv --trace_mode constraint --output traces.npz
```

Trace modes: `random`, `constraint`.

### 2. Train

```bash
uv run python training.py --traces_path traces.npz --batch_size 512 --num_tokens 8_000_000_000
```

Training uses a token-based budget (`--num_tokens`) with a tqdm progress bar showing throughput and loss. Use `--resume` to continue from the latest checkpoint in `checkpoints/`.

Key flags: `--num_tokens`, `--warmup_tokens`, `--lr`, `--val_every`, `--log_every`, `--num_checkpoints`.

### 3. Evaluate

```bash
uv run python evaluate.py --ckpt_dir checkpoints --data_path sudoku-3m.csv --n 100
```

Key flags: `--n` (number of puzzles), `--random_sample`, `--temperature`, `--quiet` (summary only).

You can also evaluate from a cached probe dataset (no checkpoint needed):

```bash
uv run python evaluate.py --cache_path probe_acts.npz --data_path sudoku-3m.csv --quiet
```

### 4. Probe residual stream

Train linear probes on residual stream activations to test what the model represents internally at each layer.

```bash
# Run probes (generates traces, collects activations, caches to probe_acts.npz)
uv run python probes.py --ckpt_dir checkpoints --data_path sudoku-3m.csv --n_puzzles 1000

# Reuse cached activations (no checkpoint needed)
uv run python probes.py --cache_path probe_acts.npz
```

Probe modes (`--mode`):
- `filled` — binary: is each cell filled or empty?
- `state_filled` — what digit is in each filled cell? (default)
- `candidates` — which digits are legal for each empty cell? (reports F1)

The `--step` flag controls which point in the solving trace to probe:
- `--step 0` (default) — probe at the `<sep>` token; ground truth = initial board (clues only)
- `--step N` (N ≥ 1) — probe at sep+N; ground truth = board state after N trace fills

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
