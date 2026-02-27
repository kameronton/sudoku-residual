"""Check all puzzles in data/chunk_1.csv for solution uniqueness using the C solver.

Status byte: 0=no solution, 1=unique, 2=non-unique.
"""

import struct
import subprocess
import sys

CSV_PATH = "data/chunk_1.csv"
SOLVER_CMD = ["./solver", "--check-unique"]
RECORD_HEADER = 1 + 81 + 1  # status + solution + trace_len
PROGRESS_INTERVAL = 100_000


def iter_puzzles(path):
    with open(path) as f:
        for line in f:
            line = line.strip()
            if len(line) < 81:
                continue
            first = line[0]
            if not (first.isdigit() or first == "."):
                continue
            yield line[:81]


def main():
    proc = subprocess.Popen(
        SOLVER_CMD,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        bufsize=1 << 20,
    )

    puzzles = list(iter_puzzles(CSV_PATH))
    total = len(puzzles)
    print(f"Loaded {total:,} puzzles, piping to solver...", flush=True)

    # Write all puzzles to solver stdin in a background thread
    import threading

    def writer():
        for puz in puzzles:
            proc.stdin.write((puz + "\n").encode())
        proc.stdin.close()

    t = threading.Thread(target=writer, daemon=True)
    t.start()

    # Read and parse binary output
    buf = proc.stdout
    n_no_sol = 0
    n_unique = 0
    n_nonunique = 0
    nonunique_examples = []

    for idx, puz in enumerate(puzzles):
        header = buf.read(RECORD_HEADER)
        if len(header) < RECORD_HEADER:
            print(f"\nUnexpected EOF at puzzle {idx}", file=sys.stderr)
            break
        status = header[0]
        trace_len = header[82]
        # consume trace bytes
        if trace_len > 0:
            buf.read(trace_len * 3)

        if status == 0:
            n_no_sol += 1
        elif status == 1:
            n_unique += 1
        elif status == 2:
            n_nonunique += 1
            if len(nonunique_examples) < 10:
                nonunique_examples.append((idx, puz))

        if (idx + 1) % PROGRESS_INTERVAL == 0:
            print(
                f"  {idx+1:>9,} / {total:,}  "
                f"no_sol={n_no_sol}  unique={n_unique}  non_unique={n_nonunique}",
                flush=True,
            )

    t.join()
    proc.wait()

    print(f"\n{'='*50}")
    print(f"Total puzzles : {total:,}")
    print(f"No solution   : {n_no_sol}")
    print(f"Unique        : {n_unique:,}")
    print(f"Non-unique    : {n_nonunique}")
    if nonunique_examples:
        print("\nFirst non-unique examples (index, puzzle):")
        for idx, puz in nonunique_examples:
            print(f"  [{idx}] {puz}")


if __name__ == "__main__":
    main()
