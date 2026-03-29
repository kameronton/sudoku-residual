"""Norvig-style Sudoku solver with constraint propagation + backtracking."""

import random


_ALL_BITS = (1 << 9) - 1  # 0b111111111 — all digits 1–9

# Precompute peer and unit tables as tuples of tuples (immutable, faster iteration)
_UNITS_INT: tuple[tuple[tuple[int, ...], ...], ...]
_PEERS_INT: tuple[tuple[int, ...], ...]

def _init_tables():
    global _UNITS_INT, _PEERS_INT
    unitlist = (
        [tuple(r * 9 + c for r in range(9)) for c in range(9)]
        + [tuple(r * 9 + c for c in range(9)) for r in range(9)]
        + [tuple((br*3+dr)*9 + bc*3+dc for dr in range(3) for dc in range(3))
           for br in range(3) for bc in range(3)]
    )
    _UNITS_INT = tuple(
        tuple(u for u in unitlist if i in u) for i in range(81)
    )
    _PEERS_INT = tuple(
        tuple(set().union(*_UNITS_INT[i]) - {i}) for i in range(81)
    )

_init_tables()

def _bit(d: int) -> int:
    return 1 << (d - 1)


class _SolverState:
    __slots__ = ("values", "trace", "clue_set", "peers", "units", "elim_order")
    def __init__(self, values, trace, clue_set, peers, units, elim_order):
        self.values = values
        self.trace = trace
        self.clue_set = clue_set
        self.peers = peers
        self.units = units
        self.elim_order = elim_order


def _shuffled_tables():
    """Return per-solve shuffled copies of peer and unit tables."""
    peers = tuple(tuple(random.sample(p, len(p))) for p in _PEERS_INT)
    units = tuple(
        tuple(tuple(random.sample(u, len(u))) for u in random.sample(cell_units, len(cell_units)))
        for cell_units in _UNITS_INT
    )
    elim_order = list(range(9))
    random.shuffle(elim_order)
    return peers, units, elim_order


def _eliminate(s: _SolverState, cell: int, d_bit: int) -> bool:
    old = s.values[cell]
    if not (old & d_bit):
        return True  # already eliminated
    new = old & ~d_bit
    if new == 0:
        return False
    s.values[cell] = new
    # Naked single: propagate to peers
    if new & (new - 1) == 0:  # popcount == 1
        if cell not in s.clue_set:
            r, c = divmod(cell, 9)
            s.trace.append((r, c, new.bit_length()))
        for peer in s.peers[cell]:
            if not _eliminate(s, peer, new):
                return False
    # Hidden single: for each unit of cell, check if d_bit has only one place
    for unit in s.units[cell]:
        count = 0
        place = -1
        for sq in unit:
            if s.values[sq] & d_bit:
                count += 1
                if count > 1:
                    break
                place = sq
        if count == 0:
            return False
        if count == 1:
            if not _assign(s, place, d_bit):
                return False
    return True


def _assign(s: _SolverState, cell: int, d_bit: int) -> bool:
    other = s.values[cell] & ~d_bit
    # Eliminate all other digits in shuffled order
    for i in s.elim_order:
        bit = 1 << i
        if other & bit:
            if not _eliminate(s, cell, bit):
                return False
    return True


def _search(s: _SolverState) -> bool:
    # Check if solved — MRV heuristic with random tie-breaking
    min_count = 10
    best_cells: list[int] = []
    for i in range(81):
        v = s.values[i]
        if v == 0:
            return False
        cnt = v.bit_count()
        if cnt > 1:
            if cnt < min_count:
                min_count = cnt
                best_cells = [i]
            elif cnt == min_count:
                best_cells.append(i)
    if not best_cells:
        return True  # all cells solved

    cell = random.choice(best_cells)
    bits = s.values[cell]
    # Try digits in shuffled order
    for i in s.elim_order:
        bit = 1 << i
        if bits & bit:
            trace_snap = len(s.trace)
            saved_values = s.values[:]
            if _assign(s, cell, bit):
                if _search(s):
                    return True
            s.values[:] = saved_values
            del s.trace[trace_snap:]
    return False


def solve(puzzle: str) -> tuple[str, list[tuple[int, int, int]]] | None:
    """Solve an 81-char puzzle string. Returns (solution_str, constraint_guided_trace) or None."""
    peers, units, elim_order = _shuffled_tables()
    s = _SolverState(
        values=[_ALL_BITS] * 81,
        trace=[],
        clue_set=set(),
        peers=peers,
        units=units,
        elim_order=elim_order,
    )
    # Parse grid: assign clues in random order
    clue_indices = [(i, ch) for i, ch in enumerate(puzzle) if '1' <= ch <= '9']
    random.shuffle(clue_indices)
    for i, ch in clue_indices:
        s.clue_set.add(i)
        if not _assign(s, i, _bit(int(ch))):
            return None

    # Check if solved
    if all(v.bit_count() == 1 for v in s.values):
        solution = "".join(str(v.bit_length()) for v in s.values)
        return solution, s.trace

    # Need search
    if _search(s):
        solution = "".join(str(v.bit_length()) for v in s.values)
        return solution, s.trace
    return None
