"""Activation collection and dataset I/O for residual stream probing."""

import os

import jax
import jax.numpy as jnp
import numpy as np

from data import PAD_TOKEN
from model import GPT2Model
from evaluate import load_checkpoint, generate_traces_batched_cached


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


def save_probe_dataset(path: str, activations: np.ndarray, puzzles: list[str], sequences: list[list[int]], compress: bool = True, n_clues: np.ndarray | None = None):
    """Save activations, puzzles, and token sequences together."""
    puzzle_arr = np.array(puzzles, dtype=f"U{len(puzzles[0])}")
    # Pad sequences to same length for storage
    max_len = max(len(s) for s in sequences)
    seq_arr = np.full((len(sequences), max_len), PAD_TOKEN, dtype=np.int16)
    for i, s in enumerate(sequences):
        seq_arr[i, :len(s)] = s
    save_fn = np.savez_compressed if compress else np.savez
    arrays = dict(activations=activations, puzzles=puzzle_arr, sequences=seq_arr)
    if n_clues is not None:
        arrays["n_clues"] = n_clues
    print(f"Saving probe dataset to {path} ({'compressed' if compress else 'uncompressed'})...")
    save_fn(path, **arrays)
    size_mb = os.path.getsize(path) / 1e6 if os.path.exists(path) else 0
    print(f"Saved probe dataset to {path} ({activations.shape}, {size_mb:.0f} MB)")


def load_probe_dataset(path: str):
    """Load cached activations, puzzles, sequences, and optionally n_clues."""
    data = np.load(path)
    activations = data["activations"]
    puzzles = list(data["puzzles"])
    seq_arr = data["sequences"]
    # Convert back to list of lists, stripping padding
    sequences = []
    for row in seq_arr:
        seq = row[row != PAD_TOKEN].tolist()
        sequences.append(seq)
    n_clues = data["n_clues"] if "n_clues" in data else None
    print(f"Loaded probe dataset from {path} ({activations.shape})")
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
) -> tuple[np.ndarray, list[str], list[list[int]], np.ndarray]:
    """Load checkpoint, generate traces, collect activations, and optionally save.

    Returns (activations, puzzles, sequences, n_clues).
    """
    print(f"Loading checkpoint from {ckpt_dir}" + (f" (step {ckpt_step})" if ckpt_step else ""))
    params, model = load_checkpoint(ckpt_dir, ckpt_step=ckpt_step)
    print("Model loaded")

    puzzles = load_puzzles(traces_path, n_puzzles)
    print(f"Loaded {len(puzzles)} puzzles")

    print("Generating traces...")
    traces, sequences = generate_traces_batched_cached(model, params, puzzles, batch_size)
    avg_len = np.mean([len(s) for s in sequences])
    print(f"Average sequence length: {avg_len:.1f}")

    print("Collecting activations...")
    intermediates_fn = make_intermediates_fn(model)
    activations = collect_activations(intermediates_fn, params, sequences, batch_size)
    n_clues = derive_n_clues(puzzles)

    if cache_path:
        save_probe_dataset(cache_path, activations, puzzles, sequences, compress=compress, n_clues=n_clues)

    return activations, puzzles, sequences, n_clues
