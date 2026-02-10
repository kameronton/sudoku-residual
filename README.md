This is a repo that tests whether a transformer trained to solve sudokus has:
1) a representation of the current state of the board in the residual stream
2) a representation of possible values for each cell in the residual stream

## Setup

```bash
uv sync
```

## Usage

### 1. Prepare traces (one-time)

```bash
uv run python data.py --prepare --data_path sudoku-3m.csv --trace_mode random --output traces_random.npz
```

Trace modes: `random`, `constraint`, `human`.

### 2. Train

```bash
uv run python training.py --traces_path traces_random.npz --batch_size 64 --num_steps 100000
```

Training auto-resumes from the latest checkpoint in `checkpoints/`.

### 3. Sanity check

```bash
uv run python -c "from data import sanity_check; sanity_check()"
```