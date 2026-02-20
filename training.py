"""Training loop and utilities for GPT-2 Sudoku model."""

import argparse
import dataclasses
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
    traces_path: str = "traces.npz"
    resume: bool = False
    batch_size: int = 64
    num_tokens: int = 100_000_000
    lr: float = 3e-4
    warmup_tokens: int = 1_000_000
    weight_decay: float = 0.1
    eval_every: int = 500
    full_val: bool = False
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

    def __init__(self, log_path: str = "train_log.json", tokens_per_step: int = 0):
        self.log_path = log_path
        self.tokens_per_step = tokens_per_step
        self.entries: list[dict] = []
        self.checkpoints: list[int] = []
        self.total_tokens: int = 0
        self._t0: float = time.time()

    def log_eval(self, step: int, epoch: float, train_loss: float, val_loss: float):
        self.entries.append({
            "step": step,
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
        })

    def log_checkpoint(self, step: int):
        self.checkpoints.append(step)

    @property
    def tokens_per_sec(self) -> float:
        elapsed = time.time() - self._t0
        if elapsed <= 0:
            return 0.0
        return self.total_tokens / elapsed

    def summary(self) -> dict:
        elapsed = time.time() - self._t0
        last = self.entries[-1] if self.entries else {}
        return {
            "total_steps": last.get("step", 0),
            "total_tokens": self.total_tokens,
            "wall_time_s": elapsed,
            "tokens_per_sec": self.tokens_per_sec,
            "final_train_loss": last.get("train_loss"),
            "final_val_loss": last.get("val_loss"),
        }

    def save(self):
        data = {
            "tokens_per_step": self.tokens_per_step,
            "entries": self.entries,
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
        per_token_loss = optax.softmax_cross_entropy_with_integer_labels(
            logits=logits, 
            labels=targets
        )
        masked_loss = per_token_loss * mask
        loss = jnp.sum(masked_loss) / (jnp.sum(mask) + 1e-9)

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
    per_token_loss = optax.softmax_cross_entropy_with_integer_labels(logits, targets)
    masked_loss = per_token_loss * mask
    loss = jnp.sum(masked_loss) / (jnp.sum(mask) + 1e-9)
    return loss


def train(cfg: TrainConfig):
    # Seed everything
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    rng = jax.random.PRNGKey(cfg.seed)

    # Load dataset — preload entire array onto device for zero-copy batch indexing
    print("Loading dataset...", flush=True)
    npz = np.load(cfg.traces_path)
    if "sequences_train" in npz:
        raw_train = npz["sequences_train"]
        raw_val = npz["sequences_val"]
    else:
        # Backward compat: old NPZ without splits
        raw = npz["sequences"]
        n = raw.shape[0]
        n_val = max(1, int(n * 0.05))
        indices = np.arange(n)
        np.random.shuffle(indices)
        raw_train = raw[indices[:-n_val]]
        raw_val = raw[indices[-n_val:]]
    n_train = raw_train.shape[0]
    n_val = raw_val.shape[0]
    max_seq_len = raw_train.shape[1]
    val_indices = np.arange(n_val)

    # Upload to device — avoids CPU→device transfer every step
    device_train = jax.device_put(jnp.array(raw_train, dtype=jnp.int32))
    device_val = jax.device_put(jnp.array(raw_val, dtype=jnp.int32))
    del raw_train, raw_val
    print(f"Dataset: {n_train} train, {n_val} val (preloaded to {jax.devices()[0].platform})", flush=True)
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
    if ckpt_mgr.latest_step() is not None and cfg.resume:
        step = ckpt_mgr.latest_step()
        # First restore train_config + step to get original schedule params
        meta = ckpt_mgr.restore(step, args=ocp.args.Composite(
            step=ocp.args.JsonRestore(),
            train_config=ocp.args.JsonRestore(),
        ))
        orig_cfg = meta.train_config
        orig_total_steps = orig_cfg["num_tokens"] // (cfg.batch_size * (max_seq_len - 1))
        orig_warmup_steps = orig_cfg["warmup_tokens"] // (cfg.batch_size * (max_seq_len - 1))
        print(f"Restoring schedule from original run: {orig_total_steps} total, {orig_warmup_steps} warmup steps")
        # Rebuild state with the original schedule before restoring weights
        rng, reinit_rng = jax.random.split(rng)
        state = create_train_state(reinit_rng, cfg, model_cfg, orig_total_steps, orig_warmup_steps)
        # Now restore params + opt_state
        restored = ckpt_mgr.restore(step, args=ocp.args.Composite(
            params=ocp.args.StandardRestore(state.params),
            opt_state=ocp.args.StandardRestore(state.opt_state),
            step=ocp.args.JsonRestore(),
        ))
        state = state.replace(params=restored.params, opt_state=restored.opt_state, step=restored.step)
        start_step = int(restored.step)
        print(f"Resumed from checkpoint at step {start_step}")

    # Epoch-based iteration
    steps_per_epoch = n_train // cfg.batch_size
    total_epochs = total_steps / steps_per_epoch
    print(f"Epochs: {total_epochs:.2f} ({steps_per_epoch} steps/epoch)", flush=True)

    # Logger
    logger = TrainLogger(log_path=cfg.log_path, tokens_per_step=tokens_per_step)
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

    def compute_val_loss():
        val_losses = []
        if cfg.full_val:
            for vstart in range(0, n_val, cfg.batch_size):
                vb = device_val[vstart:min(vstart + cfg.batch_size, n_val)]
                vl = eval_step(state, vb, model_cfg.vocab_size)
                val_losses.append(float(vl))
        else:
            for _ in range(min(10, max(1, n_val // cfg.batch_size))):
                vi = np.random.choice(val_indices, size=min(cfg.batch_size, n_val), replace=False)
                vb = device_val[vi]
                vl = eval_step(state, vb, model_cfg.vocab_size)
                val_losses.append(float(vl))
        return float(np.mean(val_losses))

    step = start_step
    epoch = 0
    while step < total_steps:
        perm = np.random.permutation(n_train)
        for batch_start in range(0, steps_per_epoch * cfg.batch_size, cfg.batch_size):
            if step >= total_steps:
                break
            batch_idx = perm[batch_start:batch_start + cfg.batch_size]
            batch = device_train[batch_idx]

            state, loss = train_step(state, batch, model_cfg.vocab_size)
            step += 1
            logger.total_tokens += tokens_per_step
            pbar.update(tokens_per_step)

            if step % cfg.eval_every == 0:
                train_loss = float(loss)
                val_loss = compute_val_loss()
                epoch_float = epoch + (batch_start // cfg.batch_size + 1) / steps_per_epoch
                logger.log_eval(step, epoch_float, train_loss, val_loss)
                logger.save()
                tok_s = logger.tokens_per_sec
                pbar.set_postfix_str(
                    f"train={train_loss:.4f} | val={val_loss:.4f} | epoch={epoch_float:.2f} | tok/s={tok_s / 1000:.1f}K"
                )
                tqdm.write(f"  step {step:>6d} | train={train_loss:.4f} | val={val_loss:.4f} | epoch={epoch_float:.2f}")

            if ckpt_every > 0 and step % ckpt_every == 0:
                ckpt_mgr.save(step, args=ocp.args.Composite(
                    params=ocp.args.StandardSave(state.params),
                    opt_state=ocp.args.StandardSave(state.opt_state),
                    step=ocp.args.JsonSave(int(state.step)),
                    model_config=ocp.args.JsonSave(dataclasses.asdict(model_cfg)),
                    train_config=ocp.args.JsonSave(dataclasses.asdict(cfg)),
                ))
                ckpt_mgr.wait_until_finished()
                logger.log_checkpoint(step)
                tqdm.write(f"  checkpoint saved at step {step}")

        epoch += 1
        tqdm.write(f"  epoch {epoch} complete at step {step}")

    pbar.close()

    # Final checkpoint
    ckpt_mgr.save(total_steps, args=ocp.args.Composite(
                params=ocp.args.StandardSave(state.params),
                opt_state=ocp.args.StandardSave(state.opt_state),
                step=ocp.args.JsonSave(int(state.step)),
                model_config=ocp.args.JsonSave(dataclasses.asdict(model_cfg)),
                train_config=ocp.args.JsonSave(dataclasses.asdict(cfg)),
            ))
    ckpt_mgr.wait_until_finished()

    logger.save()

    # Print summary
    summary = logger.summary()
    print(f"\nTraining complete. Summary:")
    print(f"  Steps: {summary['total_steps']:,}")
    print(f"  Tokens: {summary['total_tokens']:,}")
    print(f"  Wall time: {summary['wall_time_s']:.1f}s")
    print(f"  Throughput: {summary['tokens_per_sec'] / 1000:.1f}K tok/s")
    if summary['final_train_loss'] is not None:
        print(f"  Final train loss: {summary['final_train_loss']:.4f}")
    if summary['final_val_loss'] is not None:
        print(f"  Final val loss: {summary['final_val_loss']:.4f}")


if __name__ == "__main__":
    cfg = parse_args()
    train(cfg)
