"""Training loop and utilities for GPT-2 Sudoku model."""

import argparse
import json
import os
import random
import time
from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
from flax.training import train_state
from tqdm import tqdm

from data import PAD_TOKEN, SEP_TOKEN, encode_fill, MAX_SEQ_LEN
from model import GPT2Model, TransformerConfig
from visualize import print_grid


# ---------------------------------------------------------------------------
# Config and logging (formerly common.py)
# ---------------------------------------------------------------------------

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
    schedule_type: str = "linear"  # or "cosine"
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
# Evaluation helpers
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
        print("  Trace:")
        for i, (r, c, d, kind) in enumerate(details):
            expected = solution[r * 9 + c]
            if kind == "CORRECT":
                print(f"    {i+1:3d}. ({r},{c})={d}  CORRECT")
            else:
                print(f"    {i+1:3d}. ({r},{c})={d}  {kind} (expected={expected})")
        print("  Model output:")
        print_grid(grid)

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


# ---------------------------------------------------------------------------
# Training functions
# ---------------------------------------------------------------------------

def make_schedule(cfg: TrainConfig, total_steps: int, warmup_steps: int, schedule_type: str = "linear") -> optax.Schedule:
    if schedule_type == "linear":
        return optax.linear_onecycle_schedule(
            transition_steps=total_steps,
            peak_value=cfg.lr,
            pct_start=warmup_steps / max(total_steps, 1),
            div_factor=10.0,
            final_div_factor=100.0,
            )
    elif schedule_type == "cosine":
        return optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=cfg.lr,
            warmup_steps=warmup_steps,
            decay_steps=max(total_steps, warmup_steps + 1),
            end_value=cfg.lr * 0.1,
        )
    else:
        raise ValueError(f"Unsupported schedule type: {schedule_type}")


def create_train_state(rng, cfg: TrainConfig, model_cfg: TransformerConfig, total_steps: int, warmup_steps: int):
    model = GPT2Model(model_cfg)
    dummy = jnp.ones((1, 82), dtype=jnp.int32)
    params = model.init(rng, dummy)["params"]
    schedule = make_schedule(cfg, total_steps, warmup_steps, schedule_type=cfg.schedule_type)
    tx = optax.adamw(learning_rate=schedule, weight_decay=cfg.weight_decay)
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)


@partial(jax.jit, static_argnames=("vocab_size",), donate_argnums=(0,))
def train_step(state, batch, vocab_size):
    """Single training step. batch is (B, T) int32 tokens."""
    inputs = batch[:, :-1]
    targets = batch[:, 1:]

    # Mask: only compute loss for tokens after <sep>, excluding pad
    sep_pos = jnp.argmax(inputs == SEP_TOKEN, axis=1, keepdims=True)  # (B, 1)
    positions = jnp.arange(targets.shape[1])[None, :]  # (1, T-1)
    mask = ((positions >= sep_pos) & (targets != PAD_TOKEN)).astype(jnp.float32)

    def loss_fn(params):
        logits = state.apply_fn({"params": params}, inputs)
        log_probs = jax.nn.log_softmax(logits.astype(jnp.float32), axis=-1)
        one_hot = jax.nn.one_hot(targets, vocab_size)
        per_token_loss = -jnp.sum(one_hot * log_probs, axis=-1)  # (B, T-1)
        loss = jnp.sum(per_token_loss * mask) / jnp.maximum(jnp.sum(mask), 1.0)
        return loss

    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss


@partial(jax.jit, static_argnames=("vocab_size",))
def eval_step(state, batch, vocab_size):
    inputs = batch[:, :-1]
    targets = batch[:, 1:]
    sep_pos = jnp.argmax(inputs == SEP_TOKEN, axis=1, keepdims=True)
    positions = jnp.arange(targets.shape[1])[None, :]
    mask = ((positions >= sep_pos) & (targets != PAD_TOKEN)).astype(jnp.float32)
    logits = state.apply_fn({"params": state.params}, inputs)
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    one_hot = jax.nn.one_hot(targets, vocab_size)
    per_token_loss = -jnp.sum(one_hot * log_probs, axis=-1)
    loss = jnp.sum(per_token_loss * mask) / jnp.maximum(jnp.sum(mask), 1.0)
    return loss


def train(cfg: TrainConfig):
    # Seed everything
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    rng = jax.random.PRNGKey(cfg.seed)

    # Load dataset — preload entire array onto device for zero-copy batch indexing
    print("Loading dataset...", flush=True)
    raw = np.load(cfg.traces_path)["sequences"]  # (N, MAX_SEQ_LEN) int16
    n = raw.shape[0]
    max_seq_len = raw.shape[1]
    n_val = max(1, int(n * cfg.val_fraction))
    n_train = n - n_val
    all_indices = np.arange(n)
    np.random.shuffle(all_indices)
    train_indices = all_indices[:n_train]
    val_indices = all_indices[n_train:]

    # Upload full dataset to device — avoids CPU→device transfer every step
    device_sequences = jax.device_put(jnp.array(raw, dtype=jnp.int32))
    del raw
    print(f"Dataset: {n} total, {n_train} train, {n_val} val (preloaded to {jax.devices()[0].platform})", flush=True)
    print(f"Max sequence length: {max_seq_len}", flush=True)

    # Compute token budget and steps
    tokens_per_step = cfg.batch_size * (max_seq_len - 1)
    total_steps = cfg.num_tokens // tokens_per_step
    warmup_steps = cfg.warmup_tokens // tokens_per_step
    ckpt_every = total_steps // cfg.num_checkpoints if cfg.num_checkpoints > 0 else 0
    print(f"Token budget: {cfg.num_tokens:,} tokens -> {total_steps:,} steps ({tokens_per_step} tok/step)", flush=True)
    if ckpt_every > 0:
        print(f"Checkpointing every {ckpt_every} steps ({cfg.num_checkpoints} checkpoints)", flush=True)

    # Model config
    model_cfg = TransformerConfig(
        n_layers=cfg.n_layers,
        n_heads=cfg.n_heads,
        d_model=cfg.d_model,
        d_ff=cfg.d_ff,
        max_seq_len=max_seq_len,
        dtype=cfg.dtype,
    )

    # Create train state
    rng, init_rng = jax.random.split(rng)
    state = create_train_state(init_rng, cfg, model_cfg, total_steps, warmup_steps)
    param_count = sum(x.size for x in jax.tree.leaves(state.params))
    print(f"Model parameters: {param_count:,}", flush=True)

    # Checkpoint manager
    os.makedirs(cfg.ckpt_dir, exist_ok=True)
    ckpt_mgr = ocp.CheckpointManager(
        os.path.abspath(cfg.ckpt_dir),
        options=ocp.CheckpointManagerOptions(max_to_keep=None),
    )

    # Resume from checkpoint if a flag is set *and* a checkpoint exists
    start_step = 0
    schedule_meta_path = os.path.join(cfg.ckpt_dir, "schedule_meta.json")
    if ckpt_mgr.latest_step() is not None and cfg.resume:
        # Restore original schedule parameters so the LR curve is unchanged
        if os.path.exists(schedule_meta_path):
            with open(schedule_meta_path) as f:
                meta = json.load(f)
            orig_total_steps = meta["total_steps"]
            orig_warmup_steps = meta["warmup_steps"]
            print(f"Restoring schedule from original run: {orig_total_steps} total, {orig_warmup_steps} warmup steps")
            # Rebuild state with the original schedule before restoring weights
            rng, reinit_rng = jax.random.split(rng)
            state = create_train_state(reinit_rng, cfg, model_cfg, orig_total_steps, orig_warmup_steps)
        step = ckpt_mgr.latest_step()
        restored = ckpt_mgr.restore(step, args=ocp.args.StandardRestore(state))
        state = restored
        start_step = int(state.step) # type: ignore
        print(f"Resumed from checkpoint at step {start_step}")

    # Save schedule metadata (so resume can reconstruct the same LR curve)
    if not os.path.exists(schedule_meta_path):
        with open(schedule_meta_path, "w") as f:
            json.dump({"total_steps": total_steps, "warmup_steps": warmup_steps}, f)

    # Logger
    logger = TrainLogger(log_path=cfg.log_path)
    logger.total_tokens = start_step * tokens_per_step

    # Training loop with tqdm
    print("Compiling train_step...", flush=True)
    pbar = tqdm(
        total=cfg.num_tokens,
        initial=logger.total_tokens,
        unit="tok",
        unit_scale=True,
        desc="Training",
        smoothing=0.1,
    )

    for step in range(start_step, total_steps):
        # Sample batch — device-side gather, no CPU→device transfer
        batch_idx = np.random.choice(train_indices, size=cfg.batch_size, replace=False)
        batch = device_sequences[batch_idx]

        state, loss = train_step(state, batch, model_cfg.vocab_size)

        # Only read loss from device when we need to display it — float()
        # forces a blocking device→host sync that stalls the TPU pipeline.
        logger.total_tokens += tokens_per_step
        pbar.update(tokens_per_step)
        should_log = (step + 1) % cfg.log_every == 0
        if should_log:
            loss_val = float(loss)
            logger.step_losses.append(loss_val)

        if should_log:
            tok_s = logger.tokens_per_sec
            pbar.set_postfix_str(
                f"loss={loss_val:.4f} | tok/s={tok_s / 1000:.1f}K | step={step + 1}"
            )

        if (step + 1) % cfg.val_every == 0:
            val_losses = []
            for _ in range(min(10, max(1, n_val // cfg.batch_size))):
                vi = np.random.choice(val_indices, size=min(cfg.batch_size, n_val), replace=False)
                vb = device_sequences[vi]
                vl = eval_step(state, vb, model_cfg.vocab_size)
                val_losses.append(float(vl))
            avg_val = np.mean(val_losses)
            logger.log_val(step + 1, float(avg_val))
            logger.save()
            tqdm.write(f"  step {step + 1:>6d} | val_loss {avg_val:.4f}")

        if ckpt_every > 0 and (step + 1) % ckpt_every == 0:
            ckpt_mgr.save(step + 1, args=ocp.args.StandardSave(state))
            ckpt_mgr.wait_until_finished()
            logger.log_checkpoint(step + 1)
            tqdm.write(f"  checkpoint saved at step {step + 1}")

    pbar.close()

    # Final checkpoint
    ckpt_mgr.save(total_steps, args=ocp.args.StandardSave(state))
    ckpt_mgr.wait_until_finished()

    logger.save()

    # Print summary
    summary = logger.summary()
    print(f"\nTraining complete. Summary:")
    print(f"  Steps: {summary['total_steps']:,}")
    print(f"  Tokens: {summary['total_tokens']:,}")
    print(f"  Wall time: {summary['wall_time_s']:.1f}s")
    print(f"  Throughput: {summary['tokens_per_sec'] / 1000:.1f}K tok/s")
    if summary['final_loss'] is not None:
        print(f"  Final train loss: {summary['final_loss']:.4f}")
    if summary['final_val_loss'] is not None:
        print(f"  Final val loss: {summary['final_val_loss']:.4f}")


if __name__ == "__main__":
    cfg = parse_args()
    train(cfg)
