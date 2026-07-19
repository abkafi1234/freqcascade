"""Imbalance-aware evaluation metrics (§5 of EXPERIMENTAL_DESIGN.md).
Raw accuracy is deliberately not computed here as a headline number —
see the design doc for why."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    f1_score,
    matthews_corrcoef,
    recall_score,
)


def macro_f1(y_true, y_pred) -> float:
    return f1_score(y_true, y_pred, average="macro", zero_division=0)


def macro_recall(y_true, y_pred) -> float:
    return recall_score(y_true, y_pred, average="macro", zero_division=0)


def gmean(y_true, y_pred) -> float:
    """Geometric mean of per-class recall — 0 if any class is never
    recalled, which is intentional: it should punish a method that
    ignores a class entirely, unlike macro-F1 which merely lowers."""
    recalls = recall_score(y_true, y_pred, average=None, zero_division=0)
    recalls = np.clip(recalls, 1e-12, 1.0)
    return float(np.exp(np.mean(np.log(recalls))))


def mcc(y_true, y_pred) -> float:
    return matthews_corrcoef(y_true, y_pred)


def bottom_quartile_recall(y_true, y_pred, train_labels) -> float:
    """Mean recall over the 25% rarest classes, rarity measured on the
    *training* label distribution (not test, which may be balanced) —
    directly targets what per-node rebalancing is meant to fix."""
    train_labels = np.asarray(train_labels)
    classes, train_counts = np.unique(train_labels, return_counts=True)
    order = np.argsort(train_counts)  # ascending: rarest first
    n_bottom = max(1, len(classes) // 4)
    rare_classes = classes[order[:n_bottom]]

    recalls = recall_score(
        y_true, y_pred, labels=list(rare_classes), average=None, zero_division=0
    )
    return float(np.mean(recalls))


def evaluate(y_true, y_pred, train_labels) -> dict[str, float]:
    return {
        "macro_f1": macro_f1(y_true, y_pred),
        "macro_recall": macro_recall(y_true, y_pred),
        "gmean": gmean(y_true, y_pred),
        "mcc": mcc(y_true, y_pred),
        "bottom_quartile_recall": bottom_quartile_recall(y_true, y_pred, train_labels),
    }
