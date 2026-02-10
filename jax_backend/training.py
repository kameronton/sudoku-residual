"""JAX training loop for GPT-2 Sudoku model."""

import os
import random
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
from flax.training import train_state
from tqdm import tqdm

from data import SudokuDataset, PAD_TOKEN, SEP_TOKEN, collate_batch
from jax_backend.transformer import GPT2Model, TransformerConfig
from common import TrainConfig, TrainLogger, parse_args


def make_schedule(cfg: TrainConfig, total_steps: int, warmup_steps: int) -> optax.Schedule:
    return optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=cfg.lr,
        warmup_steps=warmup_steps,
        decay_steps=max(total_steps, warmup_steps + 1),
        end_value=cfg.lr * 0.1,
    )


def create_train_state(rng, cfg: TrainConfig, model_cfg: TransformerConfig, total_steps: int, warmup_steps: int):
    model = GPT2Model(model_cfg)
    dummy = jnp.ones((1, 82), dtype=jnp.int32)
    params = model.init(rng, dummy)["params"]
    schedule = make_schedule(cfg, total_steps, warmup_steps)
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
        import json
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
        import json
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
        # Sample batch
        batch_idx = np.random.choice(train_indices, size=cfg.batch_size, replace=False)
        batch = jnp.array(collate_batch(dataset, batch_idx)) # type: ignore

        state, loss = train_step(state, batch, model_cfg.vocab_size)

        logger.log_step(step, float(loss), tokens_per_step)
        pbar.update(tokens_per_step)

        if (step + 1) % cfg.log_every == 0:
            loss_val = float(loss)
            tok_s = logger.tokens_per_sec
            pbar.set_postfix_str(
                f"loss={loss_val:.4f} | tok/s={tok_s / 1000:.1f}K | step={step + 1}"
            )

        if (step + 1) % cfg.val_every == 0:
            val_losses = []
            for _ in range(min(10, max(1, n_val // cfg.batch_size))):
                vi = np.random.choice(val_indices, size=min(cfg.batch_size, n_val), replace=False)
                vb = jnp.array(collate_batch(dataset, vi)) # type: ignore
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
