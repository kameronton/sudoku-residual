"""Shared training utilities for JAX and PyTorch backends."""

import argparse
import json
import os
import time
from dataclasses import dataclass

from data import encode_fill, SEP_TOKEN, PAD_TOKEN
from visualize import print_grid


@dataclass
class TrainConfig:
    traces_path: str = "traces_constraint.npz"
    resume: bool = False
    batch_size: int = 64
    num_tokens: int = 100_000_000
    lr: float = 3e-4
    warmup_tokens: int = 1_000_000
    weight_decay: float = 0.1
    val_fraction: float = 0.05
    log_every: int = 50
    val_every: int = 500
    num_checkpoints: int = 10
    ckpt_dir: str = "checkpoints"
    log_path: str = "train_log.json"
    seed: int = 42
    # Model config
    n_layers: int = 6
    n_heads: int = 4
    d_model: int = 128
    d_ff: int = 512
    dtype: str = "float32"
    backend: str = "jax"


class TrainLogger:
    """Accumulates training statistics and periodically saves to disk."""

    def __init__(self, log_path: str = "train_log.json"):
        self.log_path = log_path
        self.step_losses: list[float] = []
        self.step_tokens: list[int] = []
        self.step_times: list[float] = []
        self.val_losses: list[tuple[int, float]] = []  # (step, loss)
        self.checkpoints: list[int] = []
        self.total_tokens: int = 0
        self._t0: float = time.time()

    def log_step(self, step: int, loss: float, n_tokens: int):
        self.step_losses.append(loss)
        self.step_tokens.append(n_tokens)
        self.step_times.append(time.time())
        self.total_tokens += n_tokens

    def log_val(self, step: int, val_loss: float):
        self.val_losses.append((step, val_loss))

    def log_checkpoint(self, step: int):
        self.checkpoints.append(step)

    def recent_loss(self, n: int = 50) -> float:
        if not self.step_losses:
            return float("nan")
        window = self.step_losses[-n:]
        return sum(window) / len(window)

    @property
    def tokens_per_sec(self) -> float:
        elapsed = time.time() - self._t0
        if elapsed <= 0:
            return 0.0
        return self.total_tokens / elapsed

    def summary(self) -> dict:
        elapsed = time.time() - self._t0
        return {
            "total_steps": len(self.step_losses),
            "total_tokens": self.total_tokens,
            "wall_time_s": elapsed,
            "tokens_per_sec": self.tokens_per_sec,
            "final_loss": self.step_losses[-1] if self.step_losses else None,
            "final_val_loss": self.val_losses[-1][1] if self.val_losses else None,
        }

    def save(self):
        data = {
            "step_losses": self.step_losses,
            "step_tokens": self.step_tokens,
            "step_times": self.step_times,
            "val_losses": self.val_losses,
            "checkpoints": self.checkpoints,
            "total_tokens": self.total_tokens,
            "summary": self.summary(),
        }
        tmp = self.log_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(data, f)
        os.replace(tmp, self.log_path)


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser()
    for name, fld in TrainConfig.__dataclass_fields__.items():
        t = fld.type
        if t is bool or t == "bool":
            parser.add_argument(f"--{name}", action="store_true", default=fld.default)
        else:
            type_map = {"str": str, "int": int, "float": float, str: str, int: int, float: float}
            parser.add_argument(f"--{name}", type=type_map.get(t, type(fld.default)), default=fld.default)
    args = parser.parse_args()
    return TrainConfig(**vars(args))


# ---------------------------------------------------------------------------
# Evaluation helpers (framework-agnostic)
# ---------------------------------------------------------------------------

def encode_clues(puzzle: str) -> list[int]:
    """Encode puzzle clues + <sep> as token list."""
    tokens = []
    for i in range(81):
        if puzzle[i] in "123456789":
            r, c = divmod(i, 9)
            tokens.append(encode_fill(r, c, int(puzzle[i])))
    tokens.append(SEP_TOKEN)
    return tokens


def _is_consistent(grid: list[str], r: int, c: int, d: int) -> bool:
    """Check if placing digit d at (r,c) is consistent with current grid state."""
    ds = str(d)
    for j in range(9):
        if j != c and grid[r * 9 + j] == ds:
            return False
    for i in range(9):
        if i != r and grid[i * 9 + c] == ds:
            return False
    br, bc = (r // 3) * 3, (c // 3) * 3
    for i in range(br, br + 3):
        for j in range(bc, bc + 3):
            if (i, j) != (r, c) and grid[i * 9 + j] == ds:
                return False
    return True


def evaluate_puzzle(trace: list[tuple[int, int, int]], puzzle: str, solution: str, verbose: bool = True) -> dict:
    """Evaluate a generated trace against a puzzle. Returns stats dict."""
    grid = list(puzzle)
    n_empties = sum(1 for ch in puzzle if ch not in "123456789")

    correct = 0
    overwrites_clue = 0
    overwrites_fill = 0
    inconsistent = 0
    wrong_consistent = 0
    filled_positions = set()

    details = []
    for r, c, d in trace:
        pos = r * 9 + c
        if puzzle[pos] in "123456789":
            overwrites_clue += 1
            details.append((r, c, d, "OVERWRITES_CLUE"))
            continue
        if pos in filled_positions:
            overwrites_fill += 1
            details.append((r, c, d, "OVERWRITES_FILL"))
            continue
        filled_positions.add(pos)
        consistent = _is_consistent(grid, r, c, d)
        grid[pos] = str(d)
        if str(d) == solution[pos]:
            correct += 1
            details.append((r, c, d, "CORRECT"))
        elif not consistent:
            inconsistent += 1
            details.append((r, c, d, "INCONSISTENT"))
        else:
            wrong_consistent += 1
            details.append((r, c, d, "WRONG_CONSISTENT"))

    missing = n_empties - len(filled_positions)

    if verbose:
        print(f"  Empties: {n_empties} | Correct: {correct} | "
              f"Wrong(consistent): {wrong_consistent} | Inconsistent: {inconsistent} | "
              f"Clue overwrite: {overwrites_clue} | Fill overwrite: {overwrites_fill} | "
              f"Missing: {missing}")
        errors = [d for d in details if d[3] != "CORRECT"]
        if errors:
            print("  Model output:")
            print_grid(grid)
            for r, c, d, kind in errors:
                print(f"    ({r},{c})={d}: {kind} (solution={solution[r*9+c]})")

    return {
        "n_empties": n_empties,
        "correct": correct,
        "wrong_consistent": wrong_consistent,
        "inconsistent": inconsistent,
        "overwrites_clue": overwrites_clue,
        "overwrites_fill": overwrites_fill,
        "missing": missing,
        "cell_accuracy": correct / n_empties if n_empties > 0 else 1.0,
        "puzzle_solved": correct == n_empties,
    }
