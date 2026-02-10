"""Training loop for GPT-2 Sudoku model."""

import argparse
import os
import random
import time
from dataclasses import dataclass, field
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
from flax.training import train_state

from data import SudokuDataset, PAD_TOKEN, SEP_TOKEN, collate_batch
from transformer import GPT2Model, TransformerConfig


@dataclass
class TrainConfig:
    traces_path: str = "traces_constraint.npz"
    resume: bool = False
    batch_size: int = 64
    num_steps: int = 100_000
    lr: float = 3e-4
    warmup_steps: int = 1000
    weight_decay: float = 0.1
    val_fraction: float = 0.05
    log_every: int = 50
    val_every: int = 500
    ckpt_every: int = 5000
    ckpt_dir: str = "checkpoints"
    seed: int = 42
    # Model config
    n_layers: int = 6
    n_heads: int = 4
    d_model: int = 128
    d_ff: int = 512


def make_schedule(cfg: TrainConfig) -> optax.Schedule:
    return optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=cfg.lr,
        warmup_steps=cfg.warmup_steps,
        decay_steps=max(cfg.num_steps, cfg.warmup_steps + 1),
        end_value=cfg.lr * 0.1,
    )


def create_train_state(rng, cfg: TrainConfig, model_cfg: TransformerConfig):
    model = GPT2Model(model_cfg)
    dummy = jnp.ones((1, 256), dtype=jnp.int32)
    params = model.init(rng, dummy)["params"]
    schedule = make_schedule(cfg)
    tx = optax.adamw(learning_rate=schedule, weight_decay=cfg.weight_decay)
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)


@partial(jax.jit, static_argnames=("vocab_size",))
def train_step(state, batch, vocab_size):
    """Single training step. batch is (B, T) int32 tokens."""
    inputs = batch[:, :-1]
    targets = batch[:, 1:]

    # Mask: only compute loss for tokens after <sep>, excluding pad
    # <sep> appears in inputs; the target at that position is the first solve token
    sep_pos = jnp.argmax(inputs == SEP_TOKEN, axis=1, keepdims=True)  # (B, 1)
    positions = jnp.arange(targets.shape[1])[None, :]  # (1, T-1)
    mask = ((positions >= sep_pos) & (targets != PAD_TOKEN)).astype(jnp.float32)

    def loss_fn(params):
        logits = state.apply_fn({"params": params}, inputs)
        log_probs = jax.nn.log_softmax(logits, axis=-1)
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

    # Load dataset
    dataset = SudokuDataset(cfg.traces_path)
    n = len(dataset)
    n_val = max(1, int(n * cfg.val_fraction))
    n_train = n - n_val
    all_indices = np.arange(n)
    np.random.shuffle(all_indices)
    train_indices = all_indices[:n_train]
    val_indices = all_indices[n_train:]
    print(f"Dataset: {n} total, {n_train} train, {n_val} val", flush=True)

    # Find the longest sequence in the dataset to determine max_seq_len for the model
    max_seq_len = max(len(dataset[i]) for i in range(n))
    print(f"Max sequence length in dataset: {max_seq_len}", flush=True)
    assert max_seq_len <= 128, "Sequence length exceeds model max_seq_len"

    # Model config
    model_cfg = TransformerConfig(
        n_layers=cfg.n_layers,
        n_heads=cfg.n_heads,
        d_model=cfg.d_model,
        d_ff=cfg.d_ff,
        max_seq_len=max_seq_len + 1 , # account for the sep token
    )

    # Create train state
    rng, init_rng = jax.random.split(rng)
    state = create_train_state(init_rng, cfg, model_cfg)
    param_count = sum(x.size for x in jax.tree.leaves(state.params))
    print(f"Model parameters: {param_count:,}", flush=True)

    # Checkpoint manager
    os.makedirs(cfg.ckpt_dir, exist_ok=True)
    ckpt_mgr = ocp.CheckpointManager(
        os.path.abspath(cfg.ckpt_dir),
        options=ocp.CheckpointManagerOptions(max_to_keep=3),
    )

    # Resume from checkpoint if a flag is set *and* a checkpoint exists
    start_step = 0
    if ckpt_mgr.latest_step() is not None and cfg.resume:
        step = ckpt_mgr.latest_step()
        restored = ckpt_mgr.restore(step, args=ocp.args.StandardRestore(state))
        state = restored
        start_step = int(state.step) # type: ignore
        print(f"Resumed from checkpoint at step {start_step}")

    # Training loop
    print("Compiling train_step...", flush=True)
    t0 = time.time()
    for step in range(start_step, cfg.num_steps):
        # Sample batch
        batch_idx = np.random.choice(train_indices, size=cfg.batch_size, replace=False)
        batch = jnp.array(collate_batch(dataset, batch_idx)) # type: ignore

        state, loss = train_step(state, batch, model_cfg.vocab_size)

        if (step + 1) % cfg.log_every == 0:
            loss_val = float(loss)  # blocks until computation finishes
            elapsed = time.time() - t0
            steps_done = step + 1 - start_step
            print(f"step {step + 1:>6d} | loss {loss_val:.4f} | {steps_done / elapsed:.1f} steps/s", flush=True)

        if (step + 1) % cfg.val_every == 0:
            val_losses = []
            for _ in range(min(10, max(1, n_val // cfg.batch_size))):
                vi = np.random.choice(val_indices, size=min(cfg.batch_size, n_val), replace=False)
                vb = jnp.array(collate_batch(dataset, vi)) # type: ignore
                vl = eval_step(state, vb, model_cfg.vocab_size)
                val_losses.append(float(vl))
            print(f"         val_loss {np.mean(val_losses):.4f}", flush=True)

        if (step + 1) % cfg.ckpt_every == 0:
            ckpt_mgr.save(step + 1, args=ocp.args.StandardSave(state))
            ckpt_mgr.wait_until_finished()
            print(f"         checkpoint saved at step {step + 1}", flush=True)

    # Final checkpoint
    ckpt_mgr.save(cfg.num_steps, args=ocp.args.StandardSave(state))
    ckpt_mgr.wait_until_finished()
    print("Training complete.", flush=True)


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser()
    for name, fld in TrainConfig.__dataclass_fields__.items():
        t = fld.type
        if t == "bool":
            parser.add_argument(f"--{name}", action="store_true", default=fld.default)
        else:
            # Resolve string type annotations
            type_map = {"str": str, "int": int, "float": float}
            parser.add_argument(f"--{name}", type=type_map.get(t, type(fld.default)), default=fld.default)
    args = parser.parse_args()
    return TrainConfig(**vars(args))


if __name__ == "__main__":
    cfg = parse_args()
    train(cfg)
