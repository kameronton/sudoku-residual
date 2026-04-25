"""Probe training/evaluation helpers for analysis notebooks."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.multioutput import MultiOutputClassifier

from scripts.analysis.sudoku_state import cell_candidates_from_grid


def make_lr(C: float = 1.0) -> LogisticRegression:
    return LogisticRegression(C=C, max_iter=1000, solver="lbfgs")


def split_simple_hard(sequences: list[list[int]], push_token: int) -> np.ndarray:
    """Return boolean mask: True for traces with no PUSH token."""
    return np.array([not any(int(tok) == push_token for tok in seq) for seq in sequences])


def build_candidate_targets(grids: list[str], cell_idx: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (targets, is_empty) for cell candidate probes.

    targets is shape (N, 9), with one bit per digit.
    """
    targets = np.zeros((len(grids), 9), dtype=np.float32)
    is_empty = np.zeros(len(grids), dtype=bool)
    for i, grid in enumerate(grids):
        if grid[cell_idx] == "0":
            is_empty[i] = True
            targets[i] = cell_candidates_from_grid(grid, cell_idx)
    return targets, is_empty


def fit_multilabel_probe(X: np.ndarray, y: np.ndarray) -> MultiOutputClassifier:
    clf = MultiOutputClassifier(make_lr())
    clf.fit(X, y.astype(int))
    return clf


def eval_multilabel_probe(clf: MultiOutputClassifier, X: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Return mean AUC and mean Brier across valid labels."""
    if len(X) == 0:
        return float("nan"), float("nan")
    probas = clf.predict_proba(X)
    aucs = []
    briers = []
    for d, proba in enumerate(probas):
        yy = y[:, d]
        pp = proba[:, 1]
        if len(np.unique(yy)) > 1:
            aucs.append(roc_auc_score(yy, pp))
        briers.append(brier_score_loss(yy, pp))
    return float(np.mean(aucs)) if aucs else float("nan"), float(np.mean(briers))


def build_structure_targets(grids: list[str], subtype: str, sidx: int) -> np.ndarray:
    """Targets for row/col/box candidate probes: digit absent from structure."""
    targets = np.ones((len(grids), 9), dtype=np.float32)
    for i, grid in enumerate(grids):
        if subtype == "row":
            cells = grid[sidx * 9 : (sidx + 1) * 9]
        elif subtype == "col":
            cells = [grid[r * 9 + sidx] for r in range(9)]
        elif subtype == "box":
            br, bc = (sidx // 3) * 3, (sidx % 3) * 3
            cells = [grid[(br + dr) * 9 + (bc + dc)] for dr in range(3) for dc in range(3)]
        else:
            raise ValueError(f"Unknown subtype: {subtype}")
        for ch in cells:
            if ch in "123456789":
                targets[i, int(ch) - 1] = 0.0
    return targets


def train_cell_candidate_probes(session, train_idx, layer: int, n_cells: int = 81) -> dict[int, MultiOutputClassifier]:
    X_train = session.acts(train_idx, layer=layer)
    grids_train = session.grids(train_idx)
    probes = {}
    for cell_idx in range(n_cells):
        y, is_empty = build_candidate_targets(grids_train, cell_idx)
        if is_empty.sum() == 0:
            continue
        probes[cell_idx] = fit_multilabel_probe(X_train[is_empty], y[is_empty])
    return probes


def train_structure_candidate_probes(session, train_idx, layer: int) -> dict[tuple[str, int], MultiOutputClassifier]:
    X_train = session.acts(train_idx, layer=layer)
    grids_train = session.grids(train_idx)
    probes = {}
    for subtype in ("row", "col", "box"):
        for sidx in range(9):
            y = build_structure_targets(grids_train, subtype, sidx)
            probes[(subtype, sidx)] = fit_multilabel_probe(X_train, y)
    return probes


def evaluate_cell_probes_at_index(session, probes, index, layer: int) -> tuple[float, float]:
    X = session.acts(index, layer=layer)
    grids = session.grids(index)
    aucs = []
    briers = []
    for cell_idx, clf in probes.items():
        y, is_empty = build_candidate_targets(grids, cell_idx)
        if is_empty.sum() == 0:
            continue
        auc, brier = eval_multilabel_probe(clf, X[is_empty], y[is_empty])
        if not np.isnan(auc):
            aucs.append(auc)
        if not np.isnan(brier):
            briers.append(brier)
    return float(np.mean(aucs)) if aucs else float("nan"), float(np.mean(briers)) if briers else float("nan")


def evaluate_structure_probes_at_index(session, probes, index, layer: int) -> tuple[float, float]:
    X = session.acts(index, layer=layer)
    grids = session.grids(index)
    aucs = []
    briers = []
    for (subtype, sidx), clf in probes.items():
        y = build_structure_targets(grids, subtype, sidx)
        auc, brier = eval_multilabel_probe(clf, X, y)
        if not np.isnan(auc):
            aucs.append(auc)
        if not np.isnan(brier):
            briers.append(brier)
    return float(np.mean(aucs)) if aucs else float("nan"), float(np.mean(briers)) if briers else float("nan")
