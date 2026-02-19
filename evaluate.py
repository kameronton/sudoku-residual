"""Load a checkpoint and evaluate the model on N puzzles."""

import argparse
import csv
import os
import random
import sys

import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp

from data import SEP_TOKEN, PAD_TOKEN, MAX_SEQ_LEN, decode_fill, encode_fill, solve
from model import GPT2Model, TransformerConfig, init_kv_cache
from visualize import print_grid


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


def make_forward_fn(model):
    """Create a JIT-compiled forward function closed over the model."""
    @jax.jit
    def forward(params, tokens):
        return model.apply({"params": params}, tokens)
    return forward

# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def encode_clues(puzzle: str, randomize_order: bool = False) -> list[int]:
    """Encode puzzle clues + <sep> as token list."""
    tokens = []
    for i in range(81):
        if puzzle[i] in "123456789":
            r, c = divmod(i, 9)
            tokens.append(encode_fill(r, c, int(puzzle[i])))
    if randomize_order:
        random.shuffle(tokens)
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


def generate_traces_batched(
    forward_fn, params, puzzles: list[str], batch_size: int = 64, temperature: float = 0.0,
) -> list[list[tuple[int, int, int]]]:
    """Batched autoregressive trace generation for multiple puzzles.

    Returns a list of traces (one per puzzle), where each trace is a list of (row, col, digit).
    """
    n = len(puzzles)
    all_traces: list[list[tuple[int, int, int]]] = [[] for _ in range(n)]

    for batch_start in range(0, n, batch_size):
        batch_puzzles = puzzles[batch_start : batch_start + batch_size]
        bs = len(batch_puzzles)

        # Encode clues for each puzzle in the batch
        token_lists = [encode_clues(p) for p in batch_puzzles]
        lengths = [len(t) for t in token_lists]
        max_empties = jnp.array([sum(1 for ch in p if ch not in "123456789") for p in batch_puzzles], dtype=jnp.int32)

        # Pad to MAX_SEQ_LEN
        sequences = jnp.full((bs, MAX_SEQ_LEN), PAD_TOKEN, dtype=jnp.int32)
        for i, toks in enumerate(token_lists):
            sequences = sequences.at[i, :len(toks)].set(jnp.array(toks, dtype=jnp.int32))

        cur_positions = jnp.array(lengths, dtype=jnp.int32)
        steps_taken = jnp.zeros(bs, dtype=jnp.int32)
        done_mask = jnp.zeros(bs, dtype=jnp.bool_)
        batch_idx = jnp.arange(bs)

        max_steps = MAX_SEQ_LEN - min(lengths)
        for step_i in range(max_steps):
            if jnp.all(done_mask):
                break

            logits = forward_fn(params, sequences)

            # Gather logits at each puzzle's current position
            gather_pos = cur_positions - 1
            next_logits = logits[batch_idx, gather_pos, :]  # (bs, vocab)

            # Sample tokens
            if temperature <= 0:
                tokens = jnp.argmax(next_logits, axis=-1).astype(jnp.int32)
            else:
                tokens = jax.random.categorical(
                    jax.random.PRNGKey(step_i), next_logits / temperature
                ).astype(jnp.int32)

            # Validity check: token must be a fill action (0-728)
            valid = (tokens >= 0) & (tokens <= 728)
            active = ~done_mask & (cur_positions < MAX_SEQ_LEN)
            update_mask = active & valid

            # Update sequences where active and valid
            old_tokens = sequences[batch_idx, cur_positions]
            sequences = sequences.at[batch_idx, cur_positions].set(
                jnp.where(update_mask, tokens, old_tokens)
            )
            cur_positions = cur_positions + update_mask.astype(jnp.int32)
            steps_taken = steps_taken + update_mask.astype(jnp.int32)

            # Mark done: invalid token, hit max empties, or hit MAX_SEQ_LEN
            done_mask = done_mask | (active & ~valid) | (steps_taken >= max_empties) | (cur_positions >= MAX_SEQ_LEN)

        # Extract traces from final sequences
        for i in range(bs):
            for pos in range(lengths[i], int(cur_positions[i])):
                token = int(sequences[i, pos])
                r, c, d = decode_fill(token)
                all_traces[batch_start + i].append((r, c, d))

        print(f"  Generated {min(batch_start + bs, n)}/{n}", end="\r")

    print()
    return all_traces


def generate_traces_batched_cached(
    model, params, puzzles: list[str], batch_size: int = 64, temperature: float = 0.0,
) -> tuple[list[list[tuple[int, int, int]]], list[list[int]]]:
    """Batched autoregressive trace generation with KV cache.

    Groups puzzles by clue count (prefill length) so each group can share
    a single prefill shape, then uses cached single-token decode steps.

    Returns (traces, sequences) where sequences are the actual token lists
    (clues + SEP + trace tokens) the model produced, preserving clue order.
    """
    cfg = model.config
    n = len(puzzles)
    all_traces: list[list[tuple[int, int, int]] | None] = [None] * n
    all_sequences: list[list[int] | None] = [None] * n

    # Group puzzles by prefill length (n_clues + 1 for SEP)
    groups: dict[int, list[tuple[int, str]]] = {}
    for idx, p in enumerate(puzzles):
        clue_tokens = encode_clues(p)
        prefill_len = len(clue_tokens)
        groups.setdefault(prefill_len, []).append((idx, p))

    @jax.jit
    def decode_step(params, token, cache, cache_index):
        logits, new_cache = model.apply(
            {"params": params}, token, cache=cache, cache_index=cache_index,
        )
        return logits, new_cache

    processed = 0
    for prefill_len, group in groups.items():
        n_decode = MAX_SEQ_LEN - prefill_len

        @jax.jit
        def prefill(params, tokens, cache):
            logits, new_cache = model.apply(
                {"params": params}, tokens, cache=cache, cache_index=0,
            )
            return logits, new_cache

        for batch_start in range(0, len(group), batch_size):
            batch = group[batch_start : batch_start + batch_size]
            bs = len(batch)

            # Encode clues — all same prefill_len in this group
            token_lists = [encode_clues(p, randomize_order=True) for _, p in batch]
            prefill_tokens = jnp.array(
                [toks for toks in token_lists], dtype=jnp.int32
            )  # (bs, prefill_len)

            max_empties = jnp.array(
                [sum(1 for ch in p if ch not in "123456789") for _, p in batch],
                dtype=jnp.int32,
            )

            # Build full sequence buffer and fill prefill portion
            sequences = jnp.full((bs, MAX_SEQ_LEN), PAD_TOKEN, dtype=jnp.int32)
            sequences = sequences.at[:, :prefill_len].set(prefill_tokens)

            cache = init_kv_cache(cfg, bs)

            # Prefill
            logits, cache = prefill(params, prefill_tokens, cache)

            # First decode token from last prefill logit
            if temperature <= 0:
                next_token = jnp.argmax(logits[:, -1, :], axis=-1).astype(jnp.int32)
            else:
                next_token = jax.random.categorical(
                    jax.random.PRNGKey(0), logits[:, -1, :] / temperature
                ).astype(jnp.int32)

            valid = (next_token >= 0) & (next_token <= 728)
            sequences = sequences.at[:, prefill_len].set(
                jnp.where(valid, next_token, PAD_TOKEN)
            )
            done_mask = ~valid
            steps_taken = valid.astype(jnp.int32)

            # Decode remaining steps
            for step_i in range(1, n_decode):
                if jnp.all(done_mask):
                    break

                pos = prefill_len + step_i
                input_token = sequences[:, pos - 1:pos]  # (bs, 1)
                logits, cache = decode_step(params, input_token, cache, pos - 1)

                if temperature <= 0:
                    next_token = jnp.argmax(logits[:, 0, :], axis=-1).astype(jnp.int32)
                else:
                    next_token = jax.random.categorical(
                        jax.random.PRNGKey(step_i), logits[:, 0, :] / temperature
                    ).astype(jnp.int32)

                valid = (next_token >= 0) & (next_token <= 728)
                active = ~done_mask & (pos < MAX_SEQ_LEN)
                update_mask = active & valid

                sequences = sequences.at[:, pos].set(
                    jnp.where(update_mask, next_token, sequences[:, pos])
                )
                steps_taken = steps_taken + update_mask.astype(jnp.int32)
                done_mask = done_mask | (active & ~valid) | (steps_taken >= max_empties)

            # Extract traces and sequences
            for i in range(bs):
                orig_idx = batch[i][0]
                trace = []
                end_pos = prefill_len + int(steps_taken[i])
                for pos in range(prefill_len, end_pos):
                    token = int(sequences[i, pos])
                    if 0 <= token <= 728:
                        r, c, d = decode_fill(token)
                        trace.append((r, c, d))
                all_traces[orig_idx] = trace
                # Save actual sequence (clues + SEP + trace), stripping padding
                all_sequences[orig_idx] = [int(sequences[i, p]) for p in range(end_pos)]

            processed += bs
            print(f"  Generated {min(processed, n)}/{n}", end="\r")

    print()
    return all_traces, all_sequences


def traces_to_sequences(puzzles: list[str], traces: list[list[tuple[int, int, int]]]) -> list[list[int]]:
    """Convert puzzles + traces into full token sequences (row-major clue order)."""
    sequences = []
    for puzzle, trace in zip(puzzles, traces):
        tokens = encode_clues(puzzle)
        for r, c, d in trace:
            tokens.append(encode_fill(r, c, d))
        sequences.append(tokens)
    return sequences


def sequences_to_traces(sequences: list[list[int]]) -> list[list[tuple[int, int, int]]]:
    """Extract trace tokens from sequences (everything after SEP_TOKEN)."""
    traces = []
    for seq in sequences:
        trace = []
        after_sep = False
        for tok in seq:
            if tok == SEP_TOKEN:
                after_sep = True
                continue
            if after_sep and 0 <= tok <= 728:
                trace.append(decode_fill(tok))
        traces.append(trace)
    return traces


def first_inconsistent_cell(
    trace: list[tuple[int, int, int]], puzzle: str,
) -> tuple[int, int, int] | None:
    """Replay trace on puzzle grid, return (row, col, step_idx) of first inconsistency, or None."""
    grid = list(puzzle)
    for step_idx, (r, c, d) in enumerate(trace):
        pos = r * 9 + c
        if grid[pos] not in ".0":
            # Overwriting a clue or previous fill — treat as inconsistency
            return (r, c, step_idx)
        if not _is_consistent(grid, r, c, d):
            return (r, c, step_idx)
        grid[pos] = str(d)
    return None


def plot_mistake_position_distribution(
    steps_from_end: list[int], output_path: str,
) -> None:
    """Plot histogram of how many steps from the end of the trace the first mistake occurs."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4))
    max_val = max(steps_from_end)
    bins = min(max_val + 1, 50)
    ax.hist(steps_from_end, bins=bins, edgecolor="black", linewidth=0.5)
    ax.set_xlabel("Steps from end of trace")
    ax.set_ylabel("Count")
    ax.set_title(f"First mistake position ({len(steps_from_end)} puzzles with errors)")
    ax.invert_xaxis()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved mistake position distribution to {output_path}")


def plot_first_mistake_heatmap(
    positions: list[tuple[int, int]], output_path: str,
) -> None:
    """Plot 9x9 heatmap of first-mistake positions and save to file."""
    import matplotlib.pyplot as plt

    counts = np.zeros((9, 9), dtype=int)
    for r, c in positions:
        counts[r, c] += 1

    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(counts, cmap="Reds", origin="upper")
    # Annotate cells
    for i in range(9):
        for j in range(9):
            v = counts[i, j]
            if v > 0:
                ax.text(j, i, str(v), ha="center", va="center", fontsize=9,
                        color="white" if v > counts.max() / 2 else "black")
    # Sudoku box lines
    for k in range(0, 10, 3):
        lw = 2
        ax.axhline(k - 0.5, color="black", linewidth=lw)
        ax.axvline(k - 0.5, color="black", linewidth=lw)
    ax.set_xticks(range(9))
    ax.set_yticks(range(9))
    ax.set_title(f"First mistake location ({len(positions)} puzzles with errors)")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved heatmap to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate model on Sudoku puzzles")
    parser.add_argument("--ckpt_dir", default="checkpoints")
    parser.add_argument("--traces_path", default=None, help="NPZ file with test split puzzles")
    parser.add_argument("--data_path", default="sudoku-3m.csv", help="CSV fallback (deprecated)")
    parser.add_argument("--n", type=int, default=10, help="Number of puzzles to evaluate")
    parser.add_argument("--offset", type=int, default=0, help="Start index in CSV")
    parser.add_argument("--random_sample", action="store_true", help="Sample random puzzles")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--quiet", action="store_true", help="Only show summary")
    parser.add_argument("--n_layers", type=int, default=None)
    parser.add_argument("--n_heads", type=int, default=None)
    parser.add_argument("--d_model", type=int, default=None)
    parser.add_argument("--d_ff", type=int, default=None)
    parser.add_argument("--dtype", type=str, default=None, choices=["float32", "bfloat16", "float16"])
    parser.add_argument("--mistake-map", action="store_true", help="Plot first-mistake heatmap from cached traces")
    parser.add_argument("--mistake-position", action="store_true", help="Plot distribution of first-mistake position (steps from end)")
    parser.add_argument("--cache_path", default="probe_acts.npz", help="Path to cached probe dataset")
    parser.add_argument("--output", default=None, help="Output path for plots")
    args = parser.parse_args()

    # Use cached traces if --cache_path was explicitly provided
    use_cache = '--cache_path' in sys.argv or '--cache-path' in sys.argv

    if args.mistake_map or args.mistake_position:
        from probes import load_probe_dataset
        activations, puzzles, sequences = load_probe_dataset(args.cache_path)
        traces = sequences_to_traces(sequences)
        positions = []
        steps_from_end = []
        for puzzle, trace in zip(puzzles, traces):
            result = first_inconsistent_cell(trace, puzzle)
            if result is not None:
                r, c, step_idx = result
                positions.append((r, c))
                steps_from_end.append(len(trace) - step_idx)
        print(f"Found first mistakes in {len(positions)}/{len(puzzles)} puzzles")
        if not positions:
            print("No mistakes found — nothing to plot.")
            return
        if args.mistake_map:
            out = args.output or "first_mistake_heatmap.png"
            plot_first_mistake_heatmap(positions, out)
        if args.mistake_position:
            out = args.output or "first_mistake_position.png"
            plot_mistake_position_distribution(steps_from_end, out)
        return

    if use_cache:
        from probes import load_probe_dataset
        activations, puzzles_cached, sequences = load_probe_dataset(args.cache_path)
        traces = sequences_to_traces(sequences)

        # Solve puzzles to get solutions
        from data import solve
        puzzles = []
        for p in puzzles_cached:
            result = solve(p)
            if result is None:
                raise ValueError(f"Solver failed for puzzle: {p[:20]}...")
            puzzles.append((p, result[0]))

        all_stats = []
        for idx, ((puzzle, solution), trace) in enumerate(zip(puzzles, traces)):
            if not args.quiet:
                print(f"\nPuzzle {idx + 1}/{len(puzzles)}:")
                print_grid(list(puzzle))
            stats = evaluate_puzzle(trace, puzzle, solution, verbose=not args.quiet)
            all_stats.append(stats)

    else:
        model_cfg = None
        if any(v is not None for v in [args.n_layers, args.n_heads, args.d_model, args.d_ff, args.dtype]):
            model_cfg = TransformerConfig(
                n_layers=args.n_layers or 6, n_heads=args.n_heads or 4,
                d_model=args.d_model or 128, d_ff=args.d_ff or 512,
                dtype=args.dtype or "float32",
            )
        params, model = load_checkpoint(args.ckpt_dir, model_cfg)
        forward_fn = make_forward_fn(model)

        # Load puzzles — prefer NPZ test split, fall back to CSV
        puzzles = []
        if args.traces_path and os.path.exists(args.traces_path):
            npz = np.load(args.traces_path, allow_pickle=False)
            if "puzzles_test" in npz:
                puzzle_strs = list(npz["puzzles_test"])
                if args.random_sample:
                    puzzle_strs = random.sample(puzzle_strs, min(args.n, len(puzzle_strs)))
                else:
                    puzzle_strs = puzzle_strs[args.offset:args.offset + args.n]
                for p in puzzle_strs:
                    result = solve(p, random_mrv=False)
                    if result is None:
                        raise ValueError(f"Solver failed: {p[:20]}...")
                    puzzles.append((p, result[0]))
                print(f"Loaded {len(puzzles)} test puzzles from {args.traces_path}")
            else:
                raise ValueError(f"No puzzles_test in {args.traces_path}")
        else:
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

        print(f"Generating traces (batch_size={args.batch_size}, kv_cache=True)...", flush=True)
        puzzle_strs = [p for p, _ in puzzles]
        traces, _sequences = generate_traces_batched_cached(model, params, puzzle_strs, args.batch_size, args.temperature)

        all_stats = []
        for idx, ((puzzle, solution), trace) in enumerate(zip(puzzles, traces)):
            if not args.quiet:
                print(f"\nPuzzle {idx + 1}/{len(puzzles)}:")
                print_grid(list(puzzle))
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
