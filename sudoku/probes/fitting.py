"""Probe fitting and evaluation using sklearn."""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from .targets import build_probe_targets, build_structure_targets, filter_by_mode


def metric_name_for_mode(mode: str) -> str:
    return "AUC"


def fit_probe(X_train: np.ndarray, y_train: np.ndarray, mode: str, C: float = 1.0):
    """Fit a logistic probe.

    y_train format:
    - filled: (n,) binary {0, 1}
    - state_filled: (n,) int {1..9}
    - candidates / structure: (n, 9) binary multi-label
    """
    if mode in ("candidates", "structure"):
        clf = MultiOutputClassifier(LogisticRegression(C=C, max_iter=1000))
        clf.fit(X_train, y_train.astype(int))
    else:
        clf = LogisticRegression(C=C, max_iter=1000)
        clf.fit(X_train, y_train)
    return clf


def eval_probe(clf, X_test: np.ndarray, y_test: np.ndarray, mode: str):
    """Evaluate a fitted probe using AUC and Brier score.

    Returns (auc, brier, y_true, per_digit_auc, per_digit_brier).
    per_digit_* are (9,) arrays for candidates/structure, None otherwise.
    """
    if mode == "filled":
        probas = clf.predict_proba(X_test)[:, 1]
        if len(np.unique(y_test)) < 2:
            return float("nan"), float("nan"), y_test, None, None
        brier = float(np.mean((probas - y_test) ** 2))
        return roc_auc_score(y_test, probas), brier, y_test, None, None

    elif mode == "state_filled":
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

    elif mode in ("candidates", "structure"):
        proba_list = clf.predict_proba(X_test)
        probas = np.column_stack([p[:, 1] for p in proba_list])
        per_digit_auc = np.array([
            roc_auc_score(y_test[:, d], probas[:, d])
            if len(np.unique(y_test[:, d])) > 1 else float("nan")
            for d in range(9)
        ])
        per_digit_brier = np.mean((probas - y_test) ** 2, axis=0)
        return np.nanmean(per_digit_auc), float(np.mean(per_digit_brier)), y_test, per_digit_auc, per_digit_brier

    else:
        raise ValueError(f"Unsupported mode: {mode}")


def probe_cell(activations: np.ndarray, puzzles: list[str], cell_idx: int, mode: str = "candidates"):
    """Train and evaluate a logistic probe for a single cell.

    Returns (auc, brier, y_true, per_digit_auc, per_digit_brier).
    """
    targets, labels = build_probe_targets(puzzles, cell_idx, mode)
    rel_idx, X, y = filter_by_mode(activations, targets, labels, mode)

    if len(X) < 4:
        return float("nan"), float("nan"), y, (np.full(9, float("nan")) if mode == "candidates" else None), None

    idx = np.arange(len(rel_idx))
    idx_train, idx_test = train_test_split(idx, test_size=0.2, random_state=42)
    clf = fit_probe(X[idx_train], y[idx_train], mode)
    return eval_probe(clf, X[idx_test], y[idx_test], mode)


def probe_structure(acts: np.ndarray, puzzles: list[str], subtype: str, idx: int) -> tuple[float, float]:
    """Fit and evaluate a structure probe for one row/col/box. Returns (mean AUC, mean Brier)."""
    targets = build_structure_targets(puzzles, subtype, idx)
    indices = np.arange(len(puzzles))
    idx_train, idx_test = train_test_split(indices, test_size=0.2, random_state=42)
    clf = fit_probe(acts[idx_train], targets[idx_train], mode="structure")
    proba_list = clf.predict_proba(acts[idx_test])
    probas = np.column_stack([p[:, 1] for p in proba_list])
    per_digit_auc = [
        roc_auc_score(targets[idx_test, d], probas[:, d])
        if len(np.unique(targets[idx_test, d])) > 1 else float("nan")
        for d in range(9)
    ]
    per_digit_brier = np.mean((probas - targets[idx_test]) ** 2, axis=0)
    return float(np.nanmean(per_digit_auc)), float(np.mean(per_digit_brier))
