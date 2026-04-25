"""Sudoku board-state helpers shared by analysis notebooks."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np

from sudoku.data import PAD_TOKEN, SEP_TOKEN, decode_fill
from sudoku.data_bt import END_CLUES_TOKEN, PAD_TOKEN_BT, POP_TOKEN, PUSH_TOKEN, SUCCESS_TOKEN


UNITS = (
    [[r * 9 + c for c in range(9)] for r in range(9)]
    + [[r * 9 + c for r in range(9)] for c in range(9)]
    + [
        [(br * 3 + dr) * 9 + (bc * 3 + dc) for dr in range(3) for dc in range(3)]
        for br in range(3)
        for bc in range(3)
    ]
)


class Placement(NamedTuple):
    cell: int
    digit: int
    row: int
    col: int

    @property
    def token(self) -> int:
        return self.row * 81 + self.col * 9 + (self.digit - 1)

    def label(self) -> str:
        return f"R{self.row + 1}C{self.col + 1}={self.digit}"

    def __repr__(self) -> str:
        return self.label()


@dataclass
class TechniqueGroups:
    naked_single: list[dict]
    hidden_single: list[dict]
    guess: list[dict]
    push: list[dict]
    pop: list[dict]
    complete: list[dict]
    success: list[dict]

    def as_dict(self) -> dict[str, list[dict]]:
        return {
            "naked_single": self.naked_single,
            "hidden_single": self.hidden_single,
            "guess": self.guess,
            "push": self.push,
            "pop": self.pop,
            "complete": self.complete,
            "success": self.success,
        }

    def __getitem__(self, key: str) -> list[dict]:
        return self.as_dict()[key]

    def keys(self):
        return self.as_dict().keys()

    def items(self):
        return self.as_dict().items()


def token_label(tok: int) -> str:
    tok = int(tok)
    if 0 <= tok <= 728:
        r, c, d = decode_fill(tok)
        return f"R{r + 1}C{c + 1}={d}"
    return {
        SEP_TOKEN: "SEP",
        END_CLUES_TOKEN: "END_CLUES",
        PUSH_TOKEN: "PUSH",
        POP_TOKEN: "POP",
        SUCCESS_TOKEN: "SUCCESS",
        PAD_TOKEN: "PAD",
        PAD_TOKEN_BT: "PAD",
    }.get(tok, f"UNK({tok})")


def sequence_length(seq: list[int]) -> int:
    for i, tok in enumerate(seq):
        if int(tok) in (PAD_TOKEN, PAD_TOKEN_BT):
            return i
    return len(seq)


def board_after_position(seq: list[int], pos: int) -> dict[int, int]:
    """Replay a standard/BT trace through pos and return {cell: digit}."""
    board: dict[int, int] = {}
    stack: list[dict[int, int]] = []
    for raw_tok in seq[: pos + 1]:
        tok = int(raw_tok)
        if 0 <= tok <= 728:
            board[tok // 9] = (tok % 9) + 1
        elif tok == PUSH_TOKEN:
            stack.append(dict(board))
        elif tok == POP_TOKEN and stack:
            board = stack.pop()
    return board


def board_to_grid(board: dict[int, int]) -> str:
    return "".join(str(board.get(cell, 0)) for cell in range(81))


def candidates(board: dict[int, int]) -> list[int]:
    """Return 9-bit candidate masks for every cell."""
    row_used = [0] * 9
    col_used = [0] * 9
    box_used = [0] * 9
    for cell, digit in board.items():
        r, c = divmod(cell, 9)
        bit = 1 << (digit - 1)
        row_used[r] |= bit
        col_used[c] |= bit
        box_used[(r // 3) * 3 + (c // 3)] |= bit

    full = (1 << 9) - 1
    out = [0] * 81
    for cell in range(81):
        if cell in board:
            continue
        r, c = divmod(cell, 9)
        out[cell] = full & ~row_used[r] & ~col_used[c] & ~box_used[(r // 3) * 3 + (c // 3)]
    return out


def cell_candidates_from_grid(grid: str, cell: int) -> list[int]:
    board = {i: int(ch) for i, ch in enumerate(grid) if ch in "123456789"}
    mask = candidates(board)[cell]
    return [1 if mask & (1 << d) else 0 for d in range(9)]


def naked_singles(cands: list[int]) -> list[Placement]:
    out = []
    for cell, mask in enumerate(cands):
        if mask and not (mask & (mask - 1)):
            r, c = divmod(cell, 9)
            out.append(Placement(cell, mask.bit_length(), r, c))
    return out


def hidden_singles(cands: list[int], exclude_cells: frozenset[int] = frozenset()) -> list[Placement]:
    seen = set()
    out = []
    for unit in UNITS:
        where = [[] for _ in range(10)]
        for cell in unit:
            mask = cands[cell]
            if not mask:
                continue
            for digit in range(1, 10):
                if mask & (1 << (digit - 1)):
                    where[digit].append(cell)
        for digit in range(1, 10):
            if len(where[digit]) == 1:
                cell = where[digit][0]
                if cell not in exclude_cells and (cell, digit) not in seen:
                    seen.add((cell, digit))
                    r, c = divmod(cell, 9)
                    out.append(Placement(cell, digit, r, c))
    return out


def build_technique_groups(session) -> TechniqueGroups:
    """Classify activation-index entries by board state after each token."""
    sidx = session.index
    by_puzzle: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for k in range(len(sidx)):
        by_puzzle[int(sidx.puzzle_idx[k])].append((k, int(sidx.seq_pos[k])))
    for entries in by_puzzle.values():
        entries.sort(key=lambda x: x[1])

    groups = TechniqueGroups([], [], [], [], [], [], [])
    for pi in range(session.n_puzzles):
        seq = session.sequences[pi]
        entries = by_puzzle.get(pi, [])
        if not entries:
            continue
        pos_to_k = {sp: k for k, sp in entries}
        board: dict[int, int] = {}
        stack: list[dict[int, int]] = []

        for pos, raw_tok in enumerate(seq):
            tok = int(raw_tok)
            if 0 <= tok <= 728:
                board[tok // 9] = (tok % 9) + 1
            elif tok == PUSH_TOKEN:
                stack.append(dict(board))
            elif tok == POP_TOKEN and stack:
                board = stack.pop()

            if pos not in pos_to_k:
                continue
            k = pos_to_k[pos]
            if tok == SUCCESS_TOKEN:
                groups.success.append({"flat_idx": k})
                continue
            if tok == PUSH_TOKEN:
                groups.push.append({"flat_idx": k})
                continue
            if tok == POP_TOKEN:
                groups.pop.append({"flat_idx": k})
                continue
            if len(board) == 81:
                groups.complete.append({"flat_idx": k})
                continue

            cands = candidates(board)
            naked = naked_singles(cands)
            hidden = hidden_singles(cands, frozenset(p.cell for p in naked))
            if naked:
                groups.naked_single.append({"flat_idx": k, "placements": naked})
            if hidden:
                groups.hidden_single.append({"flat_idx": k, "placements": hidden})
            if not naked and not hidden:
                groups.guess.append({"flat_idx": k})
    return groups


def group_index(session, group_entries: list[dict]):
    ks = np.array([e["flat_idx"] for e in group_entries], dtype=np.int32)
    return session.index[ks]


def filter_unique_correct(session, group_entries: list[dict]) -> list[dict]:
    """Keep entries with one placement and next token equal to that placement."""
    result = []
    for entry in group_entries:
        placements = entry.get("placements", [])
        if len(placements) != 1:
            continue
        k = entry["flat_idx"]
        pi = int(session.index.puzzle_idx[k])
        sp = int(session.index.seq_pos[k])
        seq = session.sequences[pi]
        if sp + 1 >= len(seq):
            continue
        next_tok = int(seq[sp + 1])
        if next_tok == placements[0].token:
            result.append(entry)
    return result


def format_board(board: dict[int, int], clues: str | None = None) -> str:
    lines = []
    for r in range(9):
        row = ""
        for c in range(9):
            cell = r * 9 + c
            if cell in board:
                ch = str(board[cell])
                row += f"[{ch}]" if clues and clues[cell] != "0" else f" {ch} "
            else:
                row += " . "
            if c in (2, 5):
                row += "|"
        lines.append(row)
        if r in (2, 5):
            lines.append("---------+---------+---------")
    return "\n".join(lines)
