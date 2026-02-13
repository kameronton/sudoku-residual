This is a repo that tests whether a transformer trained to solve sudokus has:
1) a representation of the current state of the board in the residual stream
2) a representation of possible values for each cell in the residual stream

## Setup

```bash
uv sync
```

### Google Colab (TPU)

```bash
!pip install uv
!git clone https://github.com/<your-username>/sudoku-residual.git
%cd sudoku-residual
!uv sync --extra tpu
```

Then prefix all commands with `uv run`, same as below. Verify TPU is visible:

```python
!uv run python -c "import jax; print(jax.devices())"
```

## Usage

### 1. Prepare traces (one-time)

```bash
uv run python data.py --prepare --data_path sudoku-3m.csv --trace_mode random --output traces_random.npz
```

Trace modes: `random`, `constraint`, `human`.

### 2. Train

```bash
uv run python training.py --traces_path traces_random.npz --batch_size 64 --num_tokens 100000000
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

To plot a heatmap of where the model's first mistakes occur:

```bash
uv run python evaluate.py --mistake-map --cache_path probe_acts.npz
```

### 4. Visualize traces

Step through a solving trace to see how cells are filled:

```bash
uv run python visualize.py --data_path sudoku-3m.csv --index 0 --mode random --step
```

Modes: `random`, `constraint`, `human`. Use `--puzzle` to pass an 81-char puzzle string directly.

### 5. Sanity check

```bash
uv run python -c "from data import sanity_check; sanity_check()"
```
