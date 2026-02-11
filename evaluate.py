"""Load a checkpoint and evaluate the model on N puzzles."""

import argparse
import csv
import os
import random

import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
from flax.training import train_state

from data import SEP_TOKEN, PAD_TOKEN, MAX_SEQ_LEN, decode_fill
from model import GPT2Model, TransformerConfig
from training import make_schedule, TrainConfig, encode_clues, evaluate_puzzle
from visualize import print_grid


def load_checkpoint(ckpt_dir: str, model_cfg: TransformerConfig):
    """Restore params from the latest checkpoint. Returns (params, apply_fn)."""
    model = GPT2Model(model_cfg)
    rng = jax.random.PRNGKey(0)
    dummy = jnp.ones((1, MAX_SEQ_LEN), dtype=jnp.int32)
    params = model.init(rng, dummy)["params"]

    train_cfg = TrainConfig()
    schedule = make_schedule(train_cfg, total_steps=1, warmup_steps=0)
    tx = optax.adamw(learning_rate=schedule, weight_decay=train_cfg.weight_decay)
    state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)

    ckpt_mgr = ocp.CheckpointManager(os.path.abspath(ckpt_dir))
    step = ckpt_mgr.latest_step()
    if step is None:
        raise FileNotFoundError(f"No checkpoints found in {ckpt_dir}")
    state = ckpt_mgr.restore(step, args=ocp.args.StandardRestore(state))
    print(f"Loaded checkpoint at step {int(state.step)}", flush=True)
    return state.params, model


def make_forward_fn(model):
    """Create a JIT-compiled forward function closed over the model."""
    @jax.jit
    def forward(params, tokens):
        return model.apply({"params": params}, tokens)
    return forward


def generate_trace(
    forward_fn, params, puzzle: str, temperature: float = 0.0,
) -> list[tuple[int, int, int]]:
    """Autoregressively generate a solving trace for a puzzle."""
    tokens = encode_clues(puzzle)
    n_empties = sum(1 for ch in puzzle if ch not in "123456789")
    trace = []

    for _ in range(n_empties):
        seq_len = len(tokens)
        if seq_len >= MAX_SEQ_LEN:
            break
        padded = tokens + [PAD_TOKEN] * (MAX_SEQ_LEN - seq_len)
        input_arr = jnp.array([padded], dtype=jnp.int32)
        logits = forward_fn(params, input_arr)
        next_logits = logits[0, seq_len - 1, :]

        if temperature <= 0:
            token = int(jnp.argmax(next_logits))
        else:
            next_logits = next_logits / temperature
            token = int(jax.random.categorical(
                jax.random.PRNGKey(seq_len), next_logits
            ))

        if token in (PAD_TOKEN, SEP_TOKEN) or token < 0 or token > 728:
            break

        r, c, d = decode_fill(token)
        trace.append((r, c, d))
        tokens.append(token)

    return trace


def main():
    parser = argparse.ArgumentParser(description="Evaluate model on Sudoku puzzles")
    parser.add_argument("--ckpt_dir", default="checkpoints")
    parser.add_argument("--data_path", default="sudoku-3m.csv")
    parser.add_argument("--n", type=int, default=10, help="Number of puzzles to evaluate")
    parser.add_argument("--offset", type=int, default=0, help="Start index in CSV")
    parser.add_argument("--random_sample", action="store_true", help="Sample random puzzles")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--quiet", action="store_true", help="Only show summary")
    parser.add_argument("--n_layers", type=int, default=6)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--d_ff", type=int, default=512)
    args = parser.parse_args()

    model_cfg = TransformerConfig(
        n_layers=args.n_layers, n_heads=args.n_heads,
        d_model=args.d_model, d_ff=args.d_ff,
    )
    params, model = load_checkpoint(args.ckpt_dir, model_cfg)
    forward_fn = make_forward_fn(model)

    # Load puzzles
    puzzles = []
    with open(args.data_path) as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if args.random_sample:
                puzzles.append((row["puzzle"], row["solution"]))
            elif args.offset <= i < args.offset + args.n:
                puzzles.append((row["puzzle"], row["solution"]))
            if not args.random_sample and i >= args.offset + args.n:
                break

    if args.random_sample:
        puzzles = random.sample(puzzles, min(args.n, len(puzzles)))
    puzzles = puzzles[:args.n]

    print(f"Compiling...", flush=True)

    all_stats = []
    for idx, (puzzle, solution) in enumerate(puzzles):
        if not args.quiet:
            print(f"\nPuzzle {idx + 1}/{len(puzzles)}:")
            print_grid(list(puzzle))
        trace = generate_trace(forward_fn, params, puzzle, args.temperature)
        stats = evaluate_puzzle(trace, puzzle, solution, verbose=not args.quiet)
        all_stats.append(stats)

    # Summary
    n = len(all_stats)
    avg_acc = np.mean([s["cell_accuracy"] for s in all_stats])
    n_solved = sum(s["puzzle_solved"] for s in all_stats)
    avg_correct = np.mean([s["correct"] for s in all_stats])
    avg_empties = np.mean([s["n_empties"] for s in all_stats])
    total_wc = sum(s["wrong_consistent"] for s in all_stats)
    total_ic = sum(s["inconsistent"] for s in all_stats)
    total_oc = sum(s["overwrites_clue"] for s in all_stats)
    total_of = sum(s["overwrites_fill"] for s in all_stats)
    total_miss = sum(s["missing"] for s in all_stats)
    print(f"\n{'='*60}")
    print(f"Results on {n} puzzles:")
    print(f"  Cell accuracy:     {avg_acc:.1%} ({avg_correct:.1f}/{avg_empties:.1f} avg)")
    print(f"  Puzzles solved:    {n_solved}/{n} ({n_solved/n:.1%})")
    print(f"  Wrong consistent:  {total_wc}")
    print(f"  Inconsistent:      {total_ic}")
    print(f"  Clue overwrites:   {total_oc}")
    print(f"  Fill overwrites:   {total_of}")
    print(f"  Missing fills:     {total_miss}")


if __name__ == "__main__":
    main()
