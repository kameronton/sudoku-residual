"""Trace generation, activation collection, and dataset I/O."""

import functools
import os
import random

import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp

from data import SEP_TOKEN, PAD_TOKEN, MAX_SEQ_LEN, decode_fill, encode_fill
from model import GPT2Model, TransformerConfig, init_kv_cache


def load_checkpoint(ckpt_dir: str, model_cfg: TransformerConfig = None, ckpt_step: int | None = None):
    """Restore params from the latest checkpoint. Returns (params, model).

    If model_cfg is None, loads config from the checkpoint.
    """
    ckpt_mgr = ocp.CheckpointManager(os.path.abspath(ckpt_dir))
    if ckpt_step is None:
        step = ckpt_mgr.latest_step()
    else:
        step = ckpt_step

    if step is None:
        raise FileNotFoundError(f"No checkpoints found in {ckpt_dir}")
    if model_cfg is None:
        meta = ckpt_mgr.restore(step, args=ocp.args.Composite(
            model_config=ocp.args.JsonRestore(),
        ))
        model_cfg = TransformerConfig(**meta.model_config)
    model = GPT2Model(model_cfg)
    params = model.init(jax.random.PRNGKey(0), jnp.ones((1, model_cfg.max_seq_len), jnp.int32))["params"]
    restored = ckpt_mgr.restore(step, args=ocp.args.Composite(
        params=ocp.args.StandardRestore(params),
    ))
    print(f"Loaded checkpoint at step {step}", flush=True)
    return restored.params, model


def encode_clues(puzzle: str, randomize_order: bool = False, use_sep: bool = True) -> list[int]:
    """Encode puzzle clues as a token list, optionally followed by <sep>."""
    tokens = []
    for i in range(81):
        if puzzle[i] in "123456789":
            r, c = divmod(i, 9)
            tokens.append(encode_fill(r, c, int(puzzle[i])))
    if randomize_order:
        random.shuffle(tokens)
    if use_sep:
        tokens.append(SEP_TOKEN)
    return tokens


def generate_traces_batched_cached(
    model, params, puzzles: list[str], batch_size: int = 64, temperature: float = 0.0,
    use_sep: bool = True,
) -> tuple[list[list[tuple[int, int, int]]], list[list[int]]]:
    """Batched autoregressive trace generation with KV cache.

    Groups puzzles by clue count (prefill length) so each group can share
    a single prefill shape, then uses cached single-token decode steps.

    Returns (traces, sequences) where sequences are the actual token lists
    (clues [+ SEP] + trace tokens) the model produced, preserving clue order.
    """
    cfg = model.config
    n = len(puzzles)
    all_traces: list[list[tuple[int, int, int]] | None] = [None] * n
    all_sequences: list[list[int] | None] = [None] * n

    # Group puzzles by prefill length (n_clues [+ 1 for SEP])
    groups: dict[int, list[tuple[int, str]]] = {}
    for idx, p in enumerate(puzzles):
        clue_tokens = encode_clues(p, use_sep=use_sep)
        prefill_len = len(clue_tokens)
        groups.setdefault(prefill_len, []).append((idx, p))

    # JIT functions closed over `model` — defined once here so JAX can cache
    # compiled versions across all groups and batches within this call.
    # Flax modules are not hashable, so model cannot be a static_argname;
    # closing over it is the standard pattern.

    @jax.jit
    def _prefill(params, sequences, cache):
        logits, new_cache = model.apply(
            {"params": params}, sequences, cache=cache, cache_index=0,
        )
        return logits, new_cache

    @functools.partial(jax.jit, static_argnames=("prefill_len", "n_decode"))
    def _decode_scan(params, sequences, cache, done_mask, steps_taken, max_empties,
                     prefill_len, n_decode):
        """Decode n_decode-1 steps via lax.scan — one XLA dispatch per batch.

        prefill_len and n_decode are static so they are constant-folded.
        One compilation per unique prefill_len (~18 for clue counts 17-35),
        cached across all batches in this call.
        """
        def body(carry, step_i):
            sequences, cache, done_mask, steps_taken = carry
            pos = prefill_len + step_i   # static int + traced int = traced int
            bs = sequences.shape[0]
            input_token = jax.lax.dynamic_slice(sequences, (0, pos - 1), (bs, 1))
            logits, cache = model.apply(
                {"params": params}, input_token, cache=cache, cache_index=pos - 1,
            )
            next_token = jnp.argmax(logits[:, 0, :], axis=-1).astype(jnp.int32)
            valid = (next_token >= 0) & (next_token <= 728)
            active = ~done_mask
            update_mask = active & valid
            new_col = jnp.where(
                update_mask[:, None],
                next_token[:, None],
                jax.lax.dynamic_slice(sequences, (0, pos), (bs, 1)),
            )
            sequences = jax.lax.dynamic_update_slice(sequences, new_col, (0, pos))
            steps_taken = steps_taken + update_mask.astype(jnp.int32)
            done_mask = done_mask | (active & ~valid) | (steps_taken >= max_empties)
            return (sequences, cache, done_mask, steps_taken), None

        (sequences, _, done_mask, steps_taken), _ = jax.lax.scan(
            body,
            (sequences, cache, done_mask, steps_taken),
            jnp.arange(1, n_decode),
            unroll=1,  # emit a while loop — fast compilation, compact HLO
        )
        return sequences, done_mask, steps_taken

    import time

    n_groups = len(groups)
    processed = 0
    t_start = time.perf_counter()

    for group_idx, (prefill_len, group) in enumerate(groups.items()):
        n_decode = MAX_SEQ_LEN - prefill_len
        compile_note = "  (first call — XLA will compile)" if group_idx == 0 else ""
        print(f"  Group {group_idx+1}/{n_groups}: prefill_len={prefill_len}, {len(group)} puzzles{compile_note}", flush=True)

        for batch_start in range(0, len(group), batch_size):
            batch = group[batch_start : batch_start + batch_size]
            actual_bs = len(batch)

            # Pad partial last batch to full batch_size so KV cache shape is always
            # (batch_size, ...) — avoids recompilation for different partial sizes
            if actual_bs < batch_size:
                batch = batch + [batch[-1]] * (batch_size - actual_bs)
            bs = batch_size

            # Encode clues — all same prefill_len in this group
            token_lists = [encode_clues(p, randomize_order=True, use_sep=use_sep) for _, p in batch]
            prefill_tokens = jnp.array(token_lists, dtype=jnp.int32)

            max_empties = jnp.array(
                [sum(1 for ch in p if ch not in "123456789") for _, p in batch],
                dtype=jnp.int32,
            )

            # Build full sequence buffer and fill prefill portion
            sequences = jnp.full((bs, MAX_SEQ_LEN), PAD_TOKEN, dtype=jnp.int32)
            sequences = sequences.at[:, :prefill_len].set(prefill_tokens)

            cache = init_kv_cache(cfg, bs)

            # Prefill — single dispatch
            logits, cache = _prefill(params, sequences, cache)

            # First decode token from SEP-position logit (greedy)
            next_token = jnp.argmax(logits[:, prefill_len - 1, :], axis=-1).astype(jnp.int32)
            valid = (next_token >= 0) & (next_token <= 728)
            sequences = sequences.at[:, prefill_len].set(
                jnp.where(valid, next_token, PAD_TOKEN)
            )
            done_mask = ~valid
            steps_taken = valid.astype(jnp.int32)

            # Decode remaining steps — single XLA dispatch via lax.scan
            sequences, done_mask, steps_taken = _decode_scan(
                params, sequences, cache, done_mask, steps_taken, max_empties,
                prefill_len, n_decode,
            )

            # Transfer entire arrays at once — two device-to-host syncs per batch.
            # Individual int(jax_array[i, p]) calls each incur ~0.4ms round-trip
            # latency on cloud TPU; for 512 examples × 82 tokens that adds up to
            # ~27s per batch. np.array() transfers in one shot regardless of size.
            sequences_np = np.array(sequences)      # (bs, MAX_SEQ_LEN)
            steps_taken_np = np.array(steps_taken)  # (bs,)

            for i in range(actual_bs):
                orig_idx = batch[i][0]
                end_pos = prefill_len + int(steps_taken_np[i])
                trace = []
                for pos in range(prefill_len, end_pos):
                    token = int(sequences_np[i, pos])
                    if 0 <= token <= 728:
                        r, c, d = decode_fill(token)
                        trace.append((r, c, d))
                all_traces[orig_idx] = trace
                all_sequences[orig_idx] = sequences_np[i, :end_pos].tolist()

            processed += actual_bs
            elapsed = time.perf_counter() - t_start
            print(f"  Generated {processed}/{n}  ({elapsed:.1f}s)", flush=True)

    elapsed = time.perf_counter() - t_start
    print(f"  Done: {n} puzzles in {elapsed:.1f}s  ({n/elapsed:.1f} puzzles/s)\n")
    return all_traces, all_sequences


def traces_to_sequences(puzzles: list[str], traces: list[list[tuple[int, int, int]]], use_sep: bool = True) -> list[list[int]]:
    """Convert puzzles + traces into full token sequences (row-major clue order)."""
    sequences = []
    for puzzle, trace in zip(puzzles, traces):
        tokens = encode_clues(puzzle, use_sep=use_sep)
        for r, c, d in trace:
            tokens.append(encode_fill(r, c, d))
        sequences.append(tokens)
    return sequences


def sequences_to_traces(sequences: list[list[int]], n_clues: np.ndarray | None = None) -> list[list[tuple[int, int, int]]]:
    """Extract trace tokens from sequences (everything after the clue/SEP boundary).

    If n_clues is provided, uses it to split: trace starts at n_clues[i]+1 (skipping SEP
    if present) or n_clues[i] (if no SEP). Falls back to scanning for SEP_TOKEN.
    """
    traces = []
    for i, seq in enumerate(sequences):
        trace = []
        if n_clues is not None:
            nc = int(n_clues[i])
            # Skip SEP if present at the boundary
            start = nc + 1 if nc < len(seq) and seq[nc] == SEP_TOKEN else nc
            for tok in seq[start:]:
                if 0 <= tok <= 728:
                    trace.append(decode_fill(tok))
        else:
            after_sep = False
            for tok in seq:
                if tok == SEP_TOKEN:
                    after_sep = True
                    continue
                if after_sep and 0 <= tok <= 728:
                    trace.append(decode_fill(tok))
        traces.append(trace)
    return traces


def make_intermediates_fn(model: GPT2Model):
    @jax.jit
    def forward(params, tokens):
        return model.apply({"params": params}, tokens, return_intermediates=True)
    return forward


def load_puzzles(traces_path: str, n: int) -> list[str]:
    """Load puzzle strings from NPZ test split."""
    npz = np.load(traces_path, allow_pickle=False)
    if "puzzles_test" not in npz:
        raise ValueError(f"No puzzles_test in {traces_path}")
    puzzles = list(npz["puzzles_test"][:n])
    print(f"Loaded {len(puzzles)} test puzzles from {traces_path}")
    return puzzles


def collect_activations(intermediates_fn, params, sequences: list[list[int]], batch_size: int):
    """Forward pass on complete sequences to get activations at all layers/tokens.

    Returns array of shape (n_puzzles, n_layers, max_seq_len, d_model).
    """
    max_length = max(len(s) for s in sequences)
    padded = jnp.array(
        [s + [PAD_TOKEN] * (max_length - len(s)) for s in sequences],
        dtype=jnp.int32,
    )

    all_acts = []
    for start in range(0, len(sequences), batch_size):
        print(f"  Forward pass {start}/{len(sequences)}", end="\r")
        batch = padded[start : start + batch_size]
        _, intermediates = intermediates_fn(params, batch)
        # intermediates: (n_layers, batch, seq_len, d_model) -> (batch, n_layers, seq_len, d_model)
        all_acts.append(np.array(intermediates.transpose(1, 0, 2, 3)))
    print()

    return np.concatenate(all_acts, axis=0).astype(np.float32)


def save_probe_dataset(path: str, activations: np.ndarray | None, puzzles: list[str], sequences: list[list[int]], compress: bool = True, n_clues: np.ndarray | None = None):
    """Save activations, puzzles, and token sequences together."""
    puzzle_arr = np.array(puzzles, dtype=f"U{len(puzzles[0])}")
    # Pad sequences to same length for storage
    max_len = max(len(s) for s in sequences)
    seq_arr = np.full((len(sequences), max_len), PAD_TOKEN, dtype=np.int16)
    for i, s in enumerate(sequences):
        seq_arr[i, :len(s)] = s
    save_fn = np.savez_compressed if compress else np.savez
    arrays = dict(puzzles=puzzle_arr, sequences=seq_arr)
    if activations is not None:
        arrays["activations"] = activations
    if n_clues is not None:
        arrays["n_clues"] = n_clues
    print(f"Saving probe dataset to {path} ({'compressed' if compress else 'uncompressed'})...")
    save_fn(path, **arrays)
    size_mb = os.path.getsize(path) / 1e6 if os.path.exists(path) else 0
    shape_info = str(activations.shape) if activations is not None else "traces only"
    print(f"Saved probe dataset to {path} ({shape_info}, {size_mb:.0f} MB)")


def load_probe_dataset(path: str):
    """Load cached activations, puzzles, sequences, and optionally n_clues."""
    data = np.load(path)
    activations = data["activations"] if "activations" in data else None
    puzzles = list(data["puzzles"])
    seq_arr = data["sequences"]
    # Convert back to list of lists, stripping padding
    sequences = []
    for row in seq_arr:
        seq = row[row != PAD_TOKEN].tolist()
        sequences.append(seq)
    n_clues = data["n_clues"] if "n_clues" in data else None
    shape_info = str(activations.shape) if activations is not None else "traces only"
    print(f"Loaded probe dataset from {path} ({shape_info})")
    return activations, puzzles, sequences, n_clues


def derive_n_clues(puzzles: list[str]) -> np.ndarray:
    """Derive n_clues from puzzle strings (count of non-zero characters)."""
    return np.array([
        sum(1 for ch in p if ch in "123456789") for p in puzzles
    ], dtype=np.int16)


def anchor_positions(n_clues: np.ndarray, anchor: str) -> list[int]:
    """Compute per-puzzle anchor position from n_clues and anchor mode.

    anchor="sep": position of SEP token = n_clues[i] (same as seq.index(SEP_TOKEN))
    anchor="last_clue": position of last clue token = n_clues[i] - 1
    """
    if anchor == "sep":
        return [int(nc) for nc in n_clues]
    elif anchor == "last_clue":
        return [int(nc) - 1 for nc in n_clues]
    else:
        raise ValueError(f"Unknown anchor mode: {anchor}")


def generate_probe_dataset(
    ckpt_dir: str,
    traces_path: str,
    n_puzzles: int = 6400,
    batch_size: int = 64,
    cache_path: str | None = None,
    compress: bool = True,
    ckpt_step: int | None = None,
    traces_only: bool = False,
) -> tuple[np.ndarray | None, list[str], list[list[int]], np.ndarray]:
    """Load checkpoint, generate traces, optionally collect activations, and save.

    When traces_only=True, skips activation collection and returns None for activations.
    Returns (activations, puzzles, sequences, n_clues).
    """
    print(f"Loading checkpoint from {ckpt_dir}" + (f" (step {ckpt_step})" if ckpt_step else ""))
    params, model = load_checkpoint(ckpt_dir, ckpt_step=ckpt_step)
    print("Model loaded")

    puzzles = load_puzzles(traces_path, n_puzzles)
    print(f"Loaded {len(puzzles)} puzzles")

    # Auto-detect whether the training data used a SEP token
    _npz = np.load(traces_path, allow_pickle=False)
    _seq0 = _npz["sequences_test"][0]
    _nc0 = int(_npz["n_clues_test"][0])
    use_sep = bool(_nc0 < len(_seq0) and _seq0[_nc0] == SEP_TOKEN)
    print(f"SEP token: {'yes' if use_sep else 'no'}")

    print("Generating traces...")
    traces, sequences = generate_traces_batched_cached(model, params, puzzles, batch_size, use_sep=use_sep)
    avg_len = np.mean([len(s) for s in sequences])
    print(f"Average sequence length: {avg_len:.1f}")

    n_clues = derive_n_clues(puzzles)

    if traces_only:
        if cache_path:
            save_probe_dataset(cache_path, None, puzzles, sequences, compress=compress, n_clues=n_clues)
        return None, puzzles, sequences, n_clues

    print("Collecting activations...")
    intermediates_fn = make_intermediates_fn(model)
    activations = collect_activations(intermediates_fn, params, sequences, batch_size)

    if cache_path:
        save_probe_dataset(cache_path, activations, puzzles, sequences, compress=compress, n_clues=n_clues)

    return activations, puzzles, sequences, n_clues
