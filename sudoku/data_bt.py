"""
Backtracking trace adapter.

Reads the binary trace format produced by the C solver and produces an NPZ
file with the same structure as data.py's output, loadable by SudokuDataset.

Extended vocabulary (the 4 new control tokens follow the 729 fill tokens):
  END_CLUES_TOKEN = 729   end of given clues  (same index as SEP_TOKEN — compatible)
  PAD_TOKEN_BT    = 730   padding (same index as PAD_TOKEN in data.py — unified padding)
  PUSH_TOKEN      = 731   entering a new search branch
  POP_TOKEN       = 732   backtracking from a failed branch
  SUCCESS_TOKEN   = 733   solution found, last token in every trace
  VOCAB_SIZE_BT   = 734

The binary event encoding (100*r + 10*c + d) differs from the token encoding
(r*81 + c*9 + (d-1)); this module converts between them using encode_fill().
"""

import argparse

import numpy as np

from sudoku.data import encode_fill

# ---------------------------------------------------------------------------
# Token constants
# ---------------------------------------------------------------------------

END_CLUES_TOKEN = 729   # same as SEP_TOKEN in data.py — compatible with probe code
PAD_TOKEN_BT    = 730   # same as PAD_TOKEN in data.py — unified padding across formats
PUSH_TOKEN      = 731
POP_TOKEN       = 732
SUCCESS_TOKEN   = 733
VOCAB_SIZE_BT   = 734   # model embedding table must cover 0..733


# ---------------------------------------------------------------------------
# Binary file I/O
# ---------------------------------------------------------------------------

def read_bt_traces(path: str, max_puzzles: int | None = None) -> list[np.ndarray]:
    """Read all records from a binary trace file.

    Returns a list of uint16 arrays, one per puzzle, containing the raw event
    values as defined in the format spec.
    """
    raw = open(path, "rb").read()
    buf = np.frombuffer(raw, dtype=np.uint8)
    pos = 0
    traces = []
    while pos + 4 <= len(buf):
        n = int(np.frombuffer(buf[pos:pos + 4], dtype="<u4")[0])
        pos += 4
        if pos + n * 2 > len(buf):
            break
        events = np.frombuffer(buf[pos:pos + n * 2], dtype="<u2").copy()
        pos += n * 2
        traces.append(events)
        if max_puzzles is not None and len(traces) >= max_puzzles:
            break
    return traces


# ---------------------------------------------------------------------------
# Tokenization
# ---------------------------------------------------------------------------

def tokenize_bt_trace(events: np.ndarray, randomize_clues: bool = False) -> np.ndarray:
    """Convert a raw event array (uint16) to a token ID array (int16).

    Placement events (value 1–999) are decoded via the binary encoding
    100*r + 10*c + d and re-encoded as r*81 + c*9 + (d-1) using encode_fill().
    Special events map directly to their token constants.

    If randomize_clues is True, the clue tokens (those before END_CLUES) are
    shuffled in place so the model cannot rely on their scan-order position.
    """
    v = events.astype(np.int32)
    tokens = np.empty(len(v), dtype=np.int16)

    is_placement = v < 1000
    r = v // 100
    c = (v // 10) % 10
    d = v % 10
    # encode_fill(r, c, d) = r*81 + c*9 + (d-1); vectorized here
    fill_tokens = r * 81 + c * 9 + (d - 1)
    tokens[is_placement] = fill_tokens[is_placement].astype(np.int16)

    tokens[v == 1000] = END_CLUES_TOKEN
    tokens[v == 1001] = PUSH_TOKEN       # 731
    tokens[v == 1002] = POP_TOKEN        # 732
    tokens[v == 1003] = SUCCESS_TOKEN    # 733

    if randomize_clues:
        end_clues_pos = int(np.argmax(v == 1000))
        np.random.shuffle(tokens[:end_clues_pos])

    return tokens


def extract_puzzle_str(events: np.ndarray) -> str:
    """Reconstruct the 81-char puzzle string from the clue placements.

    Clue placements are all events before the first END_CLUES (1000).
    Empty cells are represented as '.'.
    """
    board = ["."] * 81
    for v in events:
        v = int(v)
        if v == 1000:
            break
        if 1 <= v <= 999:
            d = v % 10
            c = (v // 10) % 10
            r = v // 100
            board[r * 9 + c] = str(d)
    return "".join(board)


# ---------------------------------------------------------------------------
# Dataset preparation
# ---------------------------------------------------------------------------

def prepare_bt_data(
    bin_path: str,
    output: str,
    max_seq_len: int | None = None,
    max_puzzles: int | None = None,
    randomize_clues: bool = False,
    train_frac: float = 0.9,
    val_frac: float = 0.05,
    test_frac: float = 0.05,
    seed: int = 42,
) -> None:
    """Read binary traces, tokenize, split, and save as NPZ.

    NPZ layout (same keys as data.py output, compatible with SudokuDataset):
      sequences_{split}   int16  (N, max_seq_len), padded with PAD_TOKEN_BT
      puzzles_{split}     U81    (N,)
      n_clues_{split}     int32  (N,)

    Additional metadata scalars stored in the NPZ:
      vocab_size          int32  = VOCAB_SIZE_BT (734)
      max_seq_len         int32  = actual padded length used

    Args:
        bin_path:    Path to the .bin trace file.
        output:      Output .npz path.
        max_seq_len: Pad/truncate to this length. Defaults to the 99th
                     percentile of trace lengths (rounded up to nearest 50),
                     which avoids the memory cost of rare outliers.
        max_puzzles:      Cap on number of puzzles to read (useful for testing).
        randomize_clues:  Shuffle clue tokens within each sequence so the model
                          cannot rely on their scan-order position.
        train_frac, val_frac, test_frac: Split proportions (must sum to 1).
        seed:        RNG seed for the shuffle.
    """
    print(f"Reading {bin_path} ...")
    raw_traces = read_bt_traces(bin_path, max_puzzles=max_puzzles)
    n_total = len(raw_traces)
    print(f"  {n_total:,} traces read")

    # --- tokenize -------------------------------------------------------
    print("Tokenizing ...")
    token_seqs = [tokenize_bt_trace(t, randomize_clues=randomize_clues) for t in raw_traces]
    lengths = np.array([len(s) for s in token_seqs])

    print(
        f"  Trace lengths — min={lengths.min()}  p50={int(np.percentile(lengths, 50))}"
        f"  p95={int(np.percentile(lengths, 95))}  p99={int(np.percentile(lengths, 99))}"
        f"  p99.9={int(np.percentile(lengths, 99.9))}  max={lengths.max()}"
    )

    # --- determine max_seq_len ------------------------------------------
    if max_seq_len is None:
        p99 = int(np.percentile(lengths, 99))
        max_seq_len = int(np.ceil(p99 / 50) * 50)   # round up to nearest 50
        print(f"  max_seq_len not specified; using p99 rounded up: {max_seq_len}")
    n_truncated = int((lengths > max_seq_len).sum())
    if n_truncated:
        print(
            f"  NOTE: {n_truncated:,} traces ({100 * n_truncated / n_total:.1f}%)"
            f" exceed max_seq_len={max_seq_len} and will be truncated"
        )

    # --- extract puzzle metadata ----------------------------------------
    print("Extracting puzzle strings and clue counts ...")
    puzzles = np.empty(n_total, dtype="U81")
    n_clues = np.empty(n_total, dtype=np.int32)
    for i, events in enumerate(raw_traces):
        puzzles[i] = extract_puzzle_str(events)
        # Count placement events before the first END_CLUES (1000)
        end_pos = int(np.argmax(events == 1000))
        n_clues[i] = end_pos

    # --- pad sequences --------------------------------------------------
    print(f"Padding to max_seq_len={max_seq_len} ...")
    sequences = np.full((n_total, max_seq_len), PAD_TOKEN_BT, dtype=np.int16)
    for i, seq in enumerate(token_seqs):
        L = min(len(seq), max_seq_len)
        sequences[i, :L] = seq[:L]

    # --- shuffle and split ----------------------------------------------
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n_total)

    n_train = int(n_total * train_frac)
    n_val   = int(n_total * val_frac)
    # test gets the remainder so that train+val+test == n_total exactly
    train_idx = idx[:n_train]
    val_idx   = idx[n_train:n_train + n_val]
    test_idx  = idx[n_train + n_val:]

    print(
        f"  Split: train={len(train_idx):,}  val={len(val_idx):,}  test={len(test_idx):,}"
    )

    # --- save -----------------------------------------------------------
    print(f"Saving to {output} ...")
    np.savez(
        output,
        sequences_train=sequences[train_idx],
        sequences_val=sequences[val_idx],
        sequences_test=sequences[test_idx],
        puzzles_train=puzzles[train_idx],
        puzzles_val=puzzles[val_idx],
        puzzles_test=puzzles[test_idx],
        n_clues_train=n_clues[train_idx],
        n_clues_val=n_clues[val_idx],
        n_clues_test=n_clues[test_idx],
        vocab_size=np.array(VOCAB_SIZE_BT, dtype=np.int32),
        max_seq_len=np.array(max_seq_len, dtype=np.int32),
    )
    print("Done.")


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------

def sanity_check_bt(bin_path: str, n: int = 5) -> None:
    """Parse the first n traces and print a human-readable summary."""
    traces = read_bt_traces(bin_path, max_puzzles=n)
    for i, events in enumerate(traces):
        puzzle = extract_puzzle_str(events)
        clue_count = int(np.argmax(events == 1000))
        has_backtrack = bool(1001 in events)
        push_count = int(np.sum(events == 1001))
        pop_count  = int(np.sum(events == 1002))
        tokens = tokenize_bt_trace(events)
        fill_count = int(np.sum(tokens < 729))
        print(
            f"[{i}] len={len(events)}  clues={clue_count}  fill={fill_count}"
            f"  push={push_count}  pop={pop_count}"
            f"  backtrack={has_backtrack}  puzzle[:20]={puzzle[:20]!r}"
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare backtracking trace dataset from binary file."
    )
    sub = parser.add_subparsers(dest="cmd")

    p = sub.add_parser("prepare", help="Convert binary traces to NPZ")
    p.add_argument("--bin_path",    required=True,       help="Input .bin trace file")
    p.add_argument("--output",      required=True,       help="Output .npz path")
    p.add_argument("--max_seq_len", type=int,            help="Pad/truncate length (default: p99 rounded to 50)")
    p.add_argument("--max_puzzles",    type=int,            help="Limit number of puzzles processed")
    p.add_argument("--randomize_clues", action="store_true", help="Shuffle clue tokens within each sequence")
    p.add_argument("--train_frac",  type=float, default=0.9)
    p.add_argument("--val_frac",    type=float, default=0.05)
    p.add_argument("--test_frac",   type=float, default=0.05)
    p.add_argument("--seed",        type=int,   default=42)

    s = sub.add_parser("check", help="Print summary of first few traces")
    s.add_argument("--bin_path", required=True, help="Input .bin trace file")
    s.add_argument("--n",        type=int, default=5, help="Number of traces to inspect")

    args = parser.parse_args()

    if args.cmd == "prepare":
        prepare_bt_data(
            bin_path=args.bin_path,
            output=args.output,
            max_seq_len=args.max_seq_len,
            max_puzzles=args.max_puzzles,
            randomize_clues=args.randomize_clues,
            train_frac=args.train_frac,
            val_frac=args.val_frac,
            test_frac=args.test_frac,
            seed=args.seed,
        )
    elif args.cmd == "check":
        sanity_check_bt(args.bin_path, n=args.n)
    else:
        parser.print_help()
