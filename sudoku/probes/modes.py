"""Probe mode strategies.

Each mode encapsulates the four steps that vary across modes:
  build_targets  — what to predict (from puzzle strings)
  prepare_samples — which samples are relevant and how to form y
  fit            — how to fit the classifier
  evaluate       — how to score it

Usage:
    mode = MODES["candidates"]
    targets, labels = mode.build_targets(puzzles, cell_idx)
    idx, X, y = mode.prepare_samples(acts, targets, labels)
    clf = mode.fit(X_train, y_train)
    auc, brier, *_ = mode.evaluate(clf, X_test, y_test)
"""

from abc import ABC, abstractmethod

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import roc_auc_score


def _make_lr(C: float = 1.0) -> LogisticRegression:
    return LogisticRegression(C=C, max_iter=1000)


def _cell_labels(puzzles: list[str], cell_idx: int) -> np.ndarray:
    """(n,) int array — digit 1-9 for filled cells, 0 for empty."""
    return np.array(
        [int(p[cell_idx]) if p[cell_idx] in "123456789" else 0 for p in puzzles]
    )


def _fit_multilabel(X_train: np.ndarray, y_train: np.ndarray):
    clf = MultiOutputClassifier(_make_lr())
    clf.fit(X_train, y_train.astype(int))
    return clf


def _eval_multilabel(
    clf, X_test: np.ndarray, y_test: np.ndarray
) -> tuple[float, float, np.ndarray, np.ndarray]:
    """Returns (mean_auc, mean_brier, per_digit_auc, per_digit_brier)."""
    proba_list = clf.predict_proba(X_test)
    probas = np.column_stack([p[:, 1] for p in proba_list])
    per_digit_auc = np.array([
        roc_auc_score(y_test[:, d], probas[:, d])
        if len(np.unique(y_test[:, d])) > 1 else float("nan")
        for d in range(9)
    ])
    per_digit_brier = np.mean((probas - y_test) ** 2, axis=0)
    with np.errstate(all="ignore"):
        mean_auc = float(np.nanmean(per_digit_auc))
    return mean_auc, float(np.mean(per_digit_brier)), per_digit_auc, per_digit_brier


def cell_candidates(puzzle: str, cell_idx: int) -> list[int]:
    """Candidate digits (1-9) for a cell based on row/col/box constraints.

    Returns a 9-element binary list: 1 if the digit is a candidate, 0 if eliminated.
    Assumes the cell is empty; result is meaningless for filled cells.
    """
    r, c = divmod(cell_idx, 9)
    used: set[int] = set()
    for j in range(9):
        for ch in (puzzle[r * 9 + j], puzzle[j * 9 + c]):
            if ch in "123456789":
                used.add(int(ch))
    br, bc = (r // 3) * 3, (c // 3) * 3
    for dr in range(3):
        for dc in range(3):
            ch = puzzle[(br + dr) * 9 + (bc + dc)]
            if ch in "123456789":
                used.add(int(ch))
    return [1 if d not in used else 0 for d in range(1, 10)]


class ProbeMode(ABC):
    """Abstract base for cell-level probe modes (filled, state_filled, candidates)."""

    name: str
    metric_name: str = "AUC"

    @abstractmethod
    def build_targets(
        self, puzzles: list[str], cell_idx: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (targets, labels) for all puzzles at cell_idx.

        labels : (n,) int — digit 1-9 for filled cells, 0 for empty
        targets: (n, k) float — format depends on mode
        """

    @abstractmethod
    def prepare_samples(
        self, X: np.ndarray, targets: np.ndarray, labels: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Select relevant samples and form y.

        Returns (rel_idx, X_filtered, y_filtered).
        """

    @abstractmethod
    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        """Fit and return a probe classifier."""

    @abstractmethod
    def evaluate(
        self, clf, X_test: np.ndarray, y_test: np.ndarray
    ) -> tuple[float, float, np.ndarray, np.ndarray | None, np.ndarray | None]:
        """Score a fitted probe.

        Returns (auc, brier, y_true, per_digit_auc, per_digit_brier).
        per_digit_* are (9,) arrays for multi-label modes, None otherwise.
        """


class FilledMode(ProbeMode):
    name = "filled"

    def build_targets(self, puzzles, cell_idx):
        labels = _cell_labels(puzzles, cell_idx)
        targets = np.eye(2)[(labels > 0).astype(int)]
        return targets, labels

    def prepare_samples(self, X, targets, labels):
        # All cells are relevant; y is binary filled/empty.
        return np.arange(len(labels)), X, (labels > 0).astype(int)

    def fit(self, X_train, y_train):
        clf = _make_lr()
        clf.fit(X_train, y_train)
        return clf

    def evaluate(self, clf, X_test, y_test):
        probas = clf.predict_proba(X_test)[:, 1]
        if len(np.unique(y_test)) < 2:
            return float("nan"), float("nan"), y_test, None, None
        brier = float(np.mean((probas - y_test) ** 2))
        return roc_auc_score(y_test, probas), brier, y_test, None, None


class StateFilledMode(ProbeMode):
    name = "state_filled"

    def build_targets(self, puzzles, cell_idx):
        labels = _cell_labels(puzzles, cell_idx)
        targets = np.zeros((len(puzzles), 9))
        filled = labels > 0
        targets[filled] = np.eye(9)[labels[filled] - 1]
        return targets, labels

    def prepare_samples(self, X, targets, labels):
        # Only filled cells — y is the digit class (1-9).
        rel_idx = np.where(labels > 0)[0]
        return rel_idx, X[rel_idx], labels[rel_idx]

    def fit(self, X_train, y_train):
        clf = _make_lr()
        clf.fit(X_train, y_train)
        return clf

    def evaluate(self, clf, X_test, y_test):
        probas = clf.predict_proba(X_test)
        full_probas = np.zeros((len(X_test), 9))
        for j, cls in enumerate(clf.classes_):
            full_probas[:, int(cls) - 1] = probas[:, j]
        present = np.unique(y_test)
        if len(present) < 2:
            return float("nan"), float("nan"), y_test, None, None
        col_idx = [int(c) - 1 for c in present]
        auc = roc_auc_score(y_test, full_probas[:, col_idx], multi_class="ovr", average="macro")
        y_onehot = np.zeros((len(y_test), 9))
        y_onehot[np.arange(len(y_test)), y_test - 1] = 1.0
        brier = float(np.mean((full_probas - y_onehot) ** 2))
        return auc, brier, y_test, None, None


class CandidatesMode(ProbeMode):
    name = "candidates"

    def build_targets(self, puzzles, cell_idx):
        labels = _cell_labels(puzzles, cell_idx)
        targets = np.zeros((len(puzzles), 9))
        for i in np.where(labels == 0)[0]:
            targets[i] = cell_candidates(puzzles[i], cell_idx)
        return targets, labels

    def prepare_samples(self, X, targets, labels):
        # Only empty cells — y is the 9-dim candidate vector.
        rel_idx = np.where(labels == 0)[0]
        return rel_idx, X[rel_idx], targets[rel_idx]

    def fit(self, X_train, y_train):
        return _fit_multilabel(X_train, y_train)

    def evaluate(self, clf, X_test, y_test):
        auc, brier, per_digit_auc, per_digit_brier = _eval_multilabel(clf, X_test, y_test)
        return auc, brier, y_test, per_digit_auc, per_digit_brier


class StructureMode:
    """Probe mode for row/col/box structure.

    Operates per-substructure rather than per-cell, so build_targets has a
    different signature from ProbeMode and prepare_samples is not needed
    (all puzzles are always used).
    """

    name = "structure"
    metric_name = "AUC"

    def build_targets(self, puzzles: list[str], subtype: str, idx: int) -> np.ndarray:
        """Return (n, 9) binary array: targets[i, d] = 1 if digit d+1 present in the substructure."""
        n = len(puzzles)
        targets = np.zeros((n, 9))
        for i, puzzle in enumerate(puzzles):
            if subtype == "row":
                cells = puzzle[idx * 9:(idx + 1) * 9]
            elif subtype == "col":
                cells = [puzzle[r * 9 + idx] for r in range(9)]
            else:  # box
                br, bc = (idx // 3) * 3, (idx % 3) * 3
                cells = [puzzle[(br + dr) * 9 + (bc + dc)] for dr in range(3) for dc in range(3)]
            for ch in cells:
                if ch in "123456789":
                    targets[i, int(ch) - 1] = 1.0
        return targets

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        return _fit_multilabel(X_train, y_train)

    def evaluate(self, clf, X_test: np.ndarray, y_test: np.ndarray) -> tuple[float, float]:
        """Returns (mean_auc, mean_brier) over all 9 digits."""
        auc, brier, _, _ = _eval_multilabel(clf, X_test, y_test)
        return auc, brier


# Registry for cell-level modes. StructureMode is separate due to its different interface.
MODES: dict[str, ProbeMode] = {
    "filled": FilledMode(),
    "state_filled": StateFilledMode(),
    "candidates": CandidatesMode(),
}

STRUCTURE = StructureMode()
