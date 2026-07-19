"""Multi-label imbalance-aware evaluation metrics (§5 of
EXPERIMENTAL_DESIGN.md, multi-label / FOCC track). Mirrors the single-label
`rfoed/metrics.py` in spirit and in which numbers are the headline: **macro**
metrics (label-based macro-F1, macro-recall, bottom-quartile-label recall)
weight every label equally regardless of how common it is, which is exactly
what this whole research question is about. Micro-averaged metrics are
dominated by frequent labels and would mask that effect -- reported here
(micro-F1, for continuity with prior Reuters-21578 literature, which mostly
reports it) but never as a headline number, the same treatment `metrics.py`
gives raw accuracy in the single-label track. Subset accuracy (exact-match)
is included for completeness but is known to be a harsh, low-informative
metric once the label set is large (a single missed/extra label out of ~90
zeroes out that document's score) -- also not a headline metric.

All functions accept `Y_true`/`Y_pred` as (n_docs, n_labels) binary
indicator matrices (the format `rfoed/data.py:load_reuters21578` and
`FOCCClassifier.predict` both use) -- sklearn's classification metrics
handle this "multilabel-indicator" format natively, no `MultiLabelBinarizer`
round-trip needed.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, hamming_loss, recall_score


def label_macro_f1(Y_true, Y_pred) -> float:
    """Per-label F1 (each of the n_labels columns treated as its own
    independent binary classification problem), averaged unweighted across
    labels -- the multi-label analog of metrics.py's macro_f1."""
    return float(f1_score(Y_true, Y_pred, average="macro", zero_division=0))


def label_macro_recall(Y_true, Y_pred) -> float:
    return float(recall_score(Y_true, Y_pred, average="macro", zero_division=0))


def micro_f1(Y_true, Y_pred) -> float:
    """Pools TP/FP/FN across every label before computing F1 -- dominated by
    frequent labels, reported only for continuity with prior Reuters-21578
    literature (see module docstring); not a headline metric here."""
    return float(f1_score(Y_true, Y_pred, average="micro", zero_division=0))


def example_f1(Y_true, Y_pred) -> float:
    """Example-based (per-document) F1: precision/recall of the predicted
    label set against the true label set for each document, averaged over
    documents. sklearn's `average="samples"` is exactly this quantity."""
    return float(f1_score(Y_true, Y_pred, average="samples", zero_division=0))


def hamming_loss_score(Y_true, Y_pred) -> float:
    """Fraction of individual (document, label) cells predicted wrong."""
    return float(hamming_loss(Y_true, Y_pred))


def subset_accuracy(Y_true, Y_pred) -> float:
    """Exact-match accuracy: fraction of documents whose entire predicted
    label *set* equals the true label set. sklearn's `accuracy_score` on a
    multilabel-indicator matrix already requires a full-row match, so no
    extra row-wise comparison is needed here. Known to be harsh/low-
    informative for large label sets (module docstring) -- reported, not
    emphasized, matching design §5."""
    return float(accuracy_score(Y_true, Y_pred))


def bottom_quartile_label_recall(Y_true, Y_pred, Y_train) -> float:
    """Mean recall over the rarest 25% of labels, rarity measured on the
    *training* label distribution (not test, which a CV fold may not
    reflect) -- directly the multi-label analog of
    `rfoed/metrics.py:bottom_quartile_recall`, same pattern: rank labels by
    training frequency ascending, take the bottom quartile (at least one
    label), score each rare label's own recall as an independent binary
    problem via simple column-slicing (multilabel-indicator format, no
    `labels=` kwarg needed since each column already *is* one label), and
    average unweighted across just those columns."""
    Y_train = np.asarray(Y_train)
    Y_true = np.asarray(Y_true)
    Y_pred = np.asarray(Y_pred)

    label_counts = Y_train.sum(axis=0)
    order = np.argsort(label_counts)  # ascending: rarest first
    n_bottom = max(1, len(label_counts) // 4)
    rare_cols = order[:n_bottom]

    recalls = recall_score(
        Y_true[:, rare_cols], Y_pred[:, rare_cols], average=None, zero_division=0
    )
    return float(np.mean(recalls))


def evaluate_multilabel(Y_true, Y_pred, Y_train) -> dict[str, float]:
    """One-call bundle of every metric above, mirroring
    `rfoed/metrics.py:evaluate`'s dict shape/signature convention.
    `Y_train` is only used for `bottom_quartile_label_recall`'s rarity
    ranking (passed separately, like `train_labels` in the single-label
    version) -- it is not itself scored."""
    return {
        "label_macro_f1": label_macro_f1(Y_true, Y_pred),
        "label_macro_recall": label_macro_recall(Y_true, Y_pred),
        "bottom_quartile_label_recall": bottom_quartile_label_recall(Y_true, Y_pred, Y_train),
        "example_f1": example_f1(Y_true, Y_pred),
        "hamming_loss": hamming_loss_score(Y_true, Y_pred),
        "subset_accuracy": subset_accuracy(Y_true, Y_pred),
        "micro_f1": micro_f1(Y_true, Y_pred),
    }
