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

from sudoku.data import PAD_TOKEN, SEP_TOKEN, encode_fill, MAX_SEQ_LEN, VOCAB_SIZE
from sudoku.data_bt import VOCAB_SIZE_BT
from sudoku.model import GPT2Model, TransformerConfig
from sudoku.visualize import print_grid


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
    no_pos_emb: bool = False
    dtype: str = "float32"
    schedule_type: str = "linear"  # or "cosine"
    schedule_frac: float = 1.0     # fraction of total_steps over which schedule runs; remainder holds at end_value
    loss_mask: str = "after_clues"  # "all" or "after_clues"
    pack_len: int = 0              # >0 enables sequence packing; sets context window length


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
    schedule_steps = max(warmup_steps + 1, round(total_steps * cfg.schedule_frac))
    if schedule_type == "linear":
        return optax.linear_onecycle_schedule(
            transition_steps=schedule_steps,
            peak_value=cfg.lr,
            pct_start=warmup_steps / max(schedule_steps, 1),
            div_factor=10.0,
            final_div_factor=100.0,
            )
    elif schedule_type == "cosine":
        return optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=cfg.lr,
            warmup_steps=warmup_steps,
            decay_steps=schedule_steps,
            end_value=cfg.lr * 0.01,
        )
    else:
        raise ValueError(f"Unsupported schedule type: {schedule_type}")


def create_train_state(rng, cfg: TrainConfig, model_cfg: TransformerConfig, total_steps: int, warmup_steps: int):
    model = GPT2Model(model_cfg)
    dummy = jnp.ones((1, model_cfg.max_seq_len), dtype=jnp.int32)
    params = model.init(rng, dummy)["params"]
    schedule = make_schedule(cfg, total_steps, warmup_steps, schedule_type=cfg.schedule_type)
    tx = optax.adamw(learning_rate=schedule, weight_decay=cfg.weight_decay)
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)


def _build_loss_mask(targets, n_clues_batch, loss_mask, has_sep, pad_token):
    """Build per-token loss mask based on masking mode."""
    positions = jnp.arange(targets.shape[1])[None, :]  # (1, T-1)
    if loss_mask == "all":
        return (targets != pad_token).astype(jnp.float32)
    else:  # "after_clues"
        boundary = n_clues_batch[:, None] if has_sep else (n_clues_batch[:, None] - 1)
        return ((positions >= boundary) & (targets != pad_token)).astype(jnp.float32)


@partial(jax.jit, static_argnames=("loss_mask", "has_sep", "pad_token"), donate_argnums=(0,))
def train_step(state, batch, n_clues_batch, loss_mask, has_sep, pad_token):
    """Single training step. batch is (B, T) int32 tokens."""
    inputs = batch[:, :-1]
    targets = batch[:, 1:]
    mask = _build_loss_mask(targets, n_clues_batch, loss_mask, has_sep, pad_token)

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


@partial(jax.jit, static_argnames=("loss_mask", "has_sep", "pad_token"))
def eval_step(state, batch, n_clues_batch, loss_mask, has_sep, pad_token):
    inputs = batch[:, :-1]
    targets = batch[:, 1:]
    mask = _build_loss_mask(targets, n_clues_batch, loss_mask, has_sep, pad_token)
    logits = state.apply_fn({"params": state.params}, inputs)
    per_token_loss = optax.softmax_cross_entropy_with_integer_labels(logits, targets)
    masked_loss = per_token_loss * mask
    loss = jnp.sum(masked_loss) / (jnp.sum(mask) + 1e-9)
    return loss


# ---------------------------------------------------------------------------
# Sequence packing utilities
# ---------------------------------------------------------------------------

def pack_sequences(seqs, n_clues_arr, pack_len, pad_token):
    """Greedily pack variable-length sequences into fixed-length rows.

    Args:
        seqs:        (N, orig_seq_len) int array, padded with pad_token
        n_clues_arr: (N,) int32 — number of clue tokens per sequence
        pack_len:    target row length
        pad_token:   padding token id

    Returns four (M, pack_len) arrays:
        packed_seqs  — same dtype as seqs
        positions    — int32, local position within document (resets per doc)
        is_trace     — bool, True for tokens that should be predicted as targets
        doc_ids      — int32, document index per position; -1 for padding slots
    """
    N = seqs.shape[0]
    # Actual lengths: count non-pad tokens (padding is always a suffix)
    seq_lens = (seqs != pad_token).sum(axis=1).astype(np.int32)

    rows_packed   = []
    rows_positions = []
    rows_is_trace  = []
    rows_doc_ids   = []

    def new_row():
        return (
            np.full(pack_len, pad_token, dtype=seqs.dtype),
            np.zeros(pack_len, dtype=np.int32),
            np.zeros(pack_len, dtype=bool),
            np.full(pack_len, -1, dtype=np.int32),
        )

    pack, pos_arr, is_trace_arr, doc_ids_arr = new_row()
    cursor = 0
    doc_id = 0

    for i in range(N):
        L = int(seq_lens[i])
        if L == 0:
            continue
        if cursor + L > pack_len:
            rows_packed.append(pack)
            rows_positions.append(pos_arr)
            rows_is_trace.append(is_trace_arr)
            rows_doc_ids.append(doc_ids_arr)
            pack, pos_arr, is_trace_arr, doc_ids_arr = new_row()
            cursor = 0
        local_pos = np.arange(L, dtype=np.int32)
        pack[cursor:cursor + L]       = seqs[i, :L]
        pos_arr[cursor:cursor + L]    = local_pos
        # END_CLUES is at index n_clues; trace starts at n_clues+1
        is_trace_arr[cursor:cursor + L] = local_pos > n_clues_arr[i]
        doc_ids_arr[cursor:cursor + L]  = doc_id
        cursor += L
        doc_id += 1

    if cursor > 0:
        rows_packed.append(pack)
        rows_positions.append(pos_arr)
        rows_is_trace.append(is_trace_arr)
        rows_doc_ids.append(doc_ids_arr)

    return (
        np.stack(rows_packed),
        np.stack(rows_positions),
        np.stack(rows_is_trace),
        np.stack(rows_doc_ids),
    )


@jax.jit
def build_attn_mask(doc_ids):
    """Build block-diagonal causal attention mask from document IDs.

    Args:
        doc_ids: (B, T) int32 — document index per position; -1 for padding

    Returns:
        (B, 1, T, T) bool — True where attention is allowed
    """
    same_doc = doc_ids[:, :, None] == doc_ids[:, None, :]    # (B, T, T)
    T = doc_ids.shape[1]
    causal = jnp.tril(jnp.ones((T, T), dtype=jnp.bool_))
    return (same_doc & causal)[:, None, :, :]                 # (B, 1, T, T)


@partial(jax.jit, donate_argnums=(0,))
def train_step_packed(state, batch, is_trace, attn_mask, positions):
    """Training step for packed sequences.

    Args:
        batch:     (B, pack_len) int32
        is_trace:  (B, pack_len) bool — True for positions that are trace tokens
        attn_mask: (B, 1, pack_len, pack_len) bool
        positions: (B, pack_len) int32 — local position within each document
    """
    inputs  = batch[:, :-1]
    targets = batch[:, 1:]
    mask = is_trace[:, 1:].astype(jnp.float32)  # targets we want to predict

    def loss_fn(params):
        logits = state.apply_fn(
            {"params": params}, inputs,
            attn_mask=attn_mask[:, :, :-1, :-1],
            positions=positions[:, :-1],
        )
        per_token_loss = optax.softmax_cross_entropy_with_integer_labels(logits, targets)
        return jnp.sum(per_token_loss * mask) / (jnp.sum(mask) + 1e-9)

    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss


@jax.jit
def eval_step_packed(state, batch, is_trace, attn_mask, positions):
    inputs  = batch[:, :-1]
    targets = batch[:, 1:]
    mask = is_trace[:, 1:].astype(jnp.float32)
    logits = state.apply_fn(
        {"params": state.params}, inputs,
        attn_mask=attn_mask[:, :, :-1, :-1],
        positions=positions[:, :-1],
    )
    per_token_loss = optax.softmax_cross_entropy_with_integer_labels(logits, targets)
    return jnp.sum(per_token_loss * mask) / (jnp.sum(mask) + 1e-9)


def train(cfg: TrainConfig):
    # Seed everything
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    rng = jax.random.PRNGKey(cfg.seed)

    # Load dataset — preload entire array onto device for zero-copy batch indexing
    print("Loading dataset...", flush=True)
    npz = np.load(cfg.traces_path)
    raw_train = npz["sequences_train"]
    raw_val = npz["sequences_val"]
    # Detect vocab size and pad token from NPZ metadata (BT format stores vocab_size=734)
    vocab_size = int(npz["vocab_size"]) if "vocab_size" in npz else VOCAB_SIZE
    pad_token  = PAD_TOKEN  # 730 in both standard and BT formats
    # Load n_clues metadata (fallback: derive from SEP position)
    if "n_clues_train" in npz:
        nc_train = npz["n_clues_train"]
        nc_val = npz["n_clues_val"]
    else:
        nc_train = np.argmax(raw_train == SEP_TOKEN, axis=1).astype(np.int32)
        nc_val = np.argmax(raw_val == SEP_TOKEN, axis=1).astype(np.int32)

    # Detect whether dataset uses SEP tokens (dataset-level property)
    has_sep = bool(SEP_TOKEN in raw_train[0])

    n_train = raw_train.shape[0]
    n_val = raw_val.shape[0]
    orig_seq_len = raw_train.shape[1]
    val_indices = np.arange(n_val)

    packing = cfg.pack_len > 0
    max_seq_len = cfg.pack_len if packing else orig_seq_len

    if packing:
        # Keep CPU copies for per-epoch packing; only val goes to device now
        raw_train_cpu = raw_train.astype(np.int32)
        nc_train_cpu  = nc_train.astype(np.int32)
        device_val    = jax.device_put(jnp.array(raw_val, dtype=jnp.int32))
        device_nc_val = jax.device_put(jnp.array(nc_val, dtype=jnp.int32))
        del raw_train, raw_val, nc_train, nc_val
        print(f"Dataset: {n_train} train, {n_val} val (packing enabled, pack_len={cfg.pack_len})", flush=True)
    else:
        # Upload to device — avoids CPU→device transfer every step
        device_train    = jax.device_put(jnp.array(raw_train, dtype=jnp.int32))
        device_val      = jax.device_put(jnp.array(raw_val,   dtype=jnp.int32))
        device_nc_train = jax.device_put(jnp.array(nc_train,  dtype=jnp.int32))
        device_nc_val   = jax.device_put(jnp.array(nc_val,    dtype=jnp.int32))
        del raw_train, raw_val, nc_train, nc_val
        print(f"Dataset: {n_train} train, {n_val} val (preloaded to {jax.devices()[0].platform})", flush=True)
    print(f"Vocab size: {vocab_size} (pad_token={pad_token})", flush=True)
    print(f"Loss mask: {cfg.loss_mask} (has_sep={has_sep})", flush=True)
    print(f"Max sequence length: {max_seq_len}", flush=True)

    # Compute token budget and steps
    tokens_per_step = cfg.batch_size * (max_seq_len - 1)
    # For packing, steps_per_epoch is determined after packing each epoch; use a rough estimate here
    steps_per_epoch = n_train // cfg.batch_size  # overridden each epoch in packing mode
    warmup_steps = cfg.warmup_tokens // tokens_per_step
    total_steps_ideal = cfg.num_tokens // tokens_per_step
    if cfg.num_checkpoints > 0:
        ckpt_every = max(1, total_steps_ideal // cfg.num_checkpoints)
        # Round total_steps to nearest checkpoint boundary (eliminates fractional remainder)
        total_steps = round(total_steps_ideal / ckpt_every) * ckpt_every
        # Validate divisibility constraint: epochs should divide checkpoints or vice versa
        total_epochs = total_steps / steps_per_epoch
        ne_int = round(total_epochs)
        nc = cfg.num_checkpoints
        if ne_int > 0 and not (ne_int % nc == 0 or nc % ne_int == 0):
            print(f"Warning: ~{ne_int} epochs and {nc} checkpoints violate divisibility constraint", flush=True)
    else:
        ckpt_every = 0
        total_steps = total_steps_ideal
    print(f"Token budget: {cfg.num_tokens:,} tokens -> {total_steps:,} steps ({tokens_per_step} tok/step)", flush=True)
    schedule_steps = max(warmup_steps + 1, round(total_steps * cfg.schedule_frac))
    frac_str = f" (schedule_frac={cfg.schedule_frac} -> {schedule_steps} steps)" if cfg.schedule_frac < 1.0 else ""
    print(f"LR schedule: {cfg.schedule_type}{frac_str}", flush=True)
    if ckpt_every > 0:
        print(f"Checkpointing every {ckpt_every} steps ({cfg.num_checkpoints} checkpoints)", flush=True)

    # Model config
    model_cfg = TransformerConfig(
        n_layers=cfg.n_layers,
        n_heads=cfg.n_heads,
        d_model=cfg.d_model,
        d_ff=cfg.d_ff,
        vocab_size=vocab_size,
        use_pos_emb=not cfg.no_pos_emb,
        max_seq_len=max_seq_len,
        dtype=cfg.dtype,
    )

    # Create train state
    rng, init_rng = jax.random.split(rng)
    state = create_train_state(init_rng, cfg, model_cfg, total_steps, warmup_steps)
    param_count = sum(x.size for x in jax.tree.leaves(state.params))
    print(f"Model parameters: {param_count:,}", flush=True)

    # Create output directories
    os.makedirs(cfg.ckpt_dir, exist_ok=True)
    log_dir = os.path.dirname(cfg.log_path)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    # Checkpoint manager
    ckpt_mgr = ocp.CheckpointManager(
        os.path.abspath(cfg.ckpt_dir),
        options=ocp.CheckpointManagerOptions(max_to_keep=None),
    )

    # Resume from checkpoint if a flag is set *and* a checkpoint exists
    start_step = 0
    if ckpt_mgr.latest_step() is not None and cfg.resume:
        step = ckpt_mgr.latest_step()
        # Restore schedule params saved at checkpoint time
        meta = ckpt_mgr.restore(step, args=ocp.args.Composite(
            step=ocp.args.JsonRestore(),
            schedule_config=ocp.args.JsonRestore(),
        ))
        orig_total_steps = meta.schedule_config["total_steps"]
        orig_warmup_steps = meta.schedule_config["warmup_steps"]
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
        n_batches = min(10, max(1, n_val // cfg.batch_size))
        indices = val_indices if cfg.full_val else None
        vstart_range = range(0, n_val, cfg.batch_size) if cfg.full_val else range(n_batches)
        for v in vstart_range:
            if cfg.full_val:
                vend = min(v + cfg.batch_size, n_val)
                vi = np.arange(v, vend)
            else:
                vi = np.random.choice(val_indices, size=min(cfg.batch_size, n_val), replace=False)
            vb  = device_val[vi]
            vnc = device_nc_val[vi]
            vl  = eval_step(state, vb, vnc, cfg.loss_mask, has_sep, pad_token)
            val_losses.append(float(vl))
        return float(np.mean(val_losses))

    def save_checkpoint(step):
        ckpt_mgr.save(step, args=ocp.args.Composite(
            params=ocp.args.StandardSave(state.params),
            opt_state=ocp.args.StandardSave(state.opt_state),
            step=ocp.args.JsonSave(int(state.step)),
            model_config=ocp.args.JsonSave(dataclasses.asdict(model_cfg)),
            train_config=ocp.args.JsonSave(dataclasses.asdict(cfg)),
            schedule_config=ocp.args.JsonSave({"total_steps": total_steps, "warmup_steps": warmup_steps}),
        ))
        ckpt_mgr.wait_until_finished()
        logger.log_checkpoint(step)
        tqdm.write(f"  checkpoint saved at step {step}")

    step = start_step
    epoch = 0

    if packing:
        # ---------------------------------------------------------------
        # Packing training loop
        # ---------------------------------------------------------------
        while step < total_steps:
            perm = np.random.permutation(n_train)
            packed_seqs, packed_pos, packed_is_trace, packed_doc_ids = pack_sequences(
                raw_train_cpu[perm], nc_train_cpu[perm], cfg.pack_len, pad_token
            )
            n_packed = packed_seqs.shape[0]
            steps_per_epoch = n_packed // cfg.batch_size
            utilization = float((packed_seqs != pad_token).sum()) / packed_seqs.size
            tqdm.write(
                f"  epoch {epoch + 1}: packed {n_train} seqs → {n_packed} rows"
                f" (utilization={utilization:.1%})"
            )

            device_packed   = jax.device_put(jnp.array(packed_seqs,      dtype=jnp.int32))
            device_pos      = jax.device_put(jnp.array(packed_pos,       dtype=jnp.int32))
            device_is_trace = jax.device_put(jnp.array(packed_is_trace,  dtype=jnp.bool_))
            device_doc_ids  = jax.device_put(jnp.array(packed_doc_ids,   dtype=jnp.int32))

            for batch_start in range(0, steps_per_epoch * cfg.batch_size, cfg.batch_size):
                if step >= total_steps:
                    break
                bi = np.arange(batch_start, batch_start + cfg.batch_size)
                batch      = device_packed[bi]
                is_trace_b = device_is_trace[bi]
                positions_b = device_pos[bi]
                doc_ids_b  = device_doc_ids[bi]
                attn_mask_b = build_attn_mask(doc_ids_b)

                state, loss = train_step_packed(state, batch, is_trace_b, attn_mask_b, positions_b)
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
                    save_checkpoint(step)

            epoch += 1
            tqdm.write(f"  epoch {epoch} complete at step {step}")

    else:
        # ---------------------------------------------------------------
        # Standard (non-packing) training loop
        # ---------------------------------------------------------------
        while step < total_steps:
            perm = np.random.permutation(n_train)
            for batch_start in range(0, steps_per_epoch * cfg.batch_size, cfg.batch_size):
                if step >= total_steps:
                    break
                batch_idx = perm[batch_start:batch_start + cfg.batch_size]
                batch    = device_train[batch_idx]
                nc_batch = device_nc_train[batch_idx]

                state, loss = train_step(state, batch, nc_batch, cfg.loss_mask, has_sep, pad_token)
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
                    save_checkpoint(step)

            epoch += 1
            tqdm.write(f"  epoch {epoch} complete at step {step}")

    pbar.close()

    # Final checkpoint only when no intermediate checkpoints were configured.
    # When ckpt_every > 0, total_steps is an exact multiple of ckpt_every so
    # the last in-loop save already fires at step == total_steps.
    if ckpt_every == 0:
        save_checkpoint(step)

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
