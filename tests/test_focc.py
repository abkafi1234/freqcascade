"""Correctness smoke tests for freqcascade.focc / freqcascade.multilabel_baselines
/ freqcascade.multilabel_metrics, using synthetic multi-label data (no
external dataset dependency) so this test suite is fully self-contained
for a standalone package install.

Run with: pip install -e ".[dev]" && pytest tests/test_focc.py -v
"""

from __future__ import annotations

import sys

import numpy as np
import pytest
import scipy.sparse as sp
from sklearn.datasets import make_multilabel_classification

from freqcascade.focc import FOCCClassifier, make_focc_nn, make_focc_rf
from freqcascade.multilabel_baselines import (
    make_balanced_br_rf,
    make_br_nn,
    make_br_rf,
    make_cc_rf,
    make_ecc_rf,
)
from freqcascade.multilabel_metrics import evaluate_multilabel

N_LABELS = 8

# Small/fast configs -- CPU-bound (device="cpu" forced for the NN path so
# this test suite doesn't require a GPU), bounded n_jobs for RF.
RF_KWARGS = dict(n_estimators=30, n_jobs=4)
NN_KWARGS = dict(n_members=3, hidden_size=16, max_epochs=15, device="cpu")


@pytest.fixture(scope="module")
def small_multilabel():
    X, Y = make_multilabel_classification(
        n_samples=500, n_features=30, n_classes=N_LABELS, n_labels=3,
        length=50, allow_unlabeled=False, random_state=0,
    )
    X_train, X_test = X[:350], X[350:]
    Y_train, Y_test = Y[:350], Y[350:]

    # Every label must have enough positive examples on both sides for the
    # per-label metrics below to be non-degenerate.
    assert (Y_train.sum(axis=0) >= 5).all()
    assert (Y_test.sum(axis=0) >= 2).all()

    return {
        "X_train_sparse": sp.csr_matrix(X_train), "X_test_sparse": sp.csr_matrix(X_test),
        "X_train_dense": X_train.astype(np.float32), "X_test_dense": X_test.astype(np.float32),
        "Y_train": Y_train, "Y_test": Y_test,
    }


def _assert_valid_predictions_and_metrics(Y_test, Y_train, pred, proba=None):
    n_test, n_labels = Y_test.shape
    assert pred.shape == (n_test, n_labels)
    assert set(np.unique(pred).tolist()) <= {0, 1}
    assert pred.sum() > 0, "predictions collapsed to the all-zero indicator matrix"

    if proba is not None:
        assert proba.shape == (n_test, n_labels)
        assert np.all((proba >= 0.0) & (proba <= 1.0))

    metrics = evaluate_multilabel(Y_test, pred, Y_train)
    expected_keys = {
        "label_macro_f1", "label_macro_recall", "bottom_quartile_label_recall",
        "example_f1", "hamming_loss", "subset_accuracy", "micro_f1",
    }
    assert set(metrics) == expected_keys
    for k, v in metrics.items():
        assert not np.isnan(v), f"{k} is NaN"
        assert 0.0 <= v <= 1.0, f"{k}={v} out of [0, 1]"
    return metrics


# ---------------------------------------------------------------------------
# FOCCClassifier: order x rebalance x base-learner-type, 2x2x2 = 8 combos
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("learner", ["rf", "nn"])
@pytest.mark.parametrize("rebalance", [True, False])
@pytest.mark.parametrize("order", ["frequency", "random"])
def test_focc_runs_and_produces_valid_output(small_multilabel, order, rebalance, learner):
    if learner == "rf":
        X_train, X_test = small_multilabel["X_train_sparse"], small_multilabel["X_test_sparse"]
        clf = make_focc_rf(order=order, rebalance=rebalance, random_state=0, **RF_KWARGS)
    else:
        X_train, X_test = small_multilabel["X_train_dense"], small_multilabel["X_test_dense"]
        clf = make_focc_nn(order=order, rebalance=rebalance, random_state=0, **NN_KWARGS)

    Y_train, Y_test = small_multilabel["Y_train"], small_multilabel["Y_test"]
    clf.fit(X_train, Y_train)

    assert len(clf.label_order_) == N_LABELS
    assert sorted(clf.label_order_) == list(range(N_LABELS))  # a permutation, nothing dropped/duplicated

    pred = clf.predict(X_test)
    proba = clf.predict_proba(X_test)
    _assert_valid_predictions_and_metrics(Y_test, Y_train, pred, proba)


def test_focc_frequency_order_matches_training_label_counts(small_multilabel):
    """order="frequency" must rank labels by descending count on the actual
    Y passed to fit."""
    Y_train = small_multilabel["Y_train"]
    clf = make_focc_rf(order="frequency", rebalance=True, random_state=0, **RF_KWARGS)
    clf.fit(small_multilabel["X_train_sparse"], Y_train)

    counts = Y_train.sum(axis=0)
    ranked = sorted(range(N_LABELS), key=lambda i: (-counts[i], i))
    assert clf.label_order_ == ranked


def test_focc_random_order_differs_from_frequency_order(small_multilabel):
    """Sanity check that order="random" is actually doing something
    different from order="frequency" (not silently falling back to it)."""
    Y_train = small_multilabel["Y_train"]
    freq_clf = make_focc_rf(order="frequency", rebalance=True, random_state=0, **RF_KWARGS)
    freq_clf.fit(small_multilabel["X_train_sparse"], Y_train)

    saw_a_difference = False
    for seed in range(5):
        rand_clf = FOCCClassifier(
            base_learner_factory=freq_clf.base_learner_factory, order="random",
            rebalance=True, random_state=seed,
        )
        rand_clf.fit(small_multilabel["X_train_sparse"], Y_train)
        if rand_clf.label_order_ != freq_clf.label_order_:
            saw_a_difference = True
            break
    assert saw_a_difference


# ---------------------------------------------------------------------------
# Baselines: BR, Balanced-BR, CC, ECC (RF-based) + BR-NN to prove the NN path
# threads through the shared baseline machinery too, not just FOCC itself.
# ---------------------------------------------------------------------------


def test_binary_relevance_rf(small_multilabel):
    clf = make_br_rf(random_state=0, **RF_KWARGS)
    clf.fit(small_multilabel["X_train_sparse"], small_multilabel["Y_train"])
    pred = clf.predict(small_multilabel["X_test_sparse"])
    _assert_valid_predictions_and_metrics(small_multilabel["Y_test"], small_multilabel["Y_train"], pred)


def test_balanced_binary_relevance_rf(small_multilabel):
    clf = make_balanced_br_rf(random_state=0, **RF_KWARGS)
    assert clf.rebalance is True
    clf.fit(small_multilabel["X_train_sparse"], small_multilabel["Y_train"])
    pred = clf.predict(small_multilabel["X_test_sparse"])
    _assert_valid_predictions_and_metrics(small_multilabel["Y_test"], small_multilabel["Y_train"], pred)


def test_classifier_chains_rf(small_multilabel):
    clf = make_cc_rf(random_state=0, **RF_KWARGS)
    assert clf.order == "random" and clf.rebalance is False
    clf.fit(small_multilabel["X_train_sparse"], small_multilabel["Y_train"])
    pred = clf.predict(small_multilabel["X_test_sparse"])
    _assert_valid_predictions_and_metrics(small_multilabel["Y_test"], small_multilabel["Y_train"], pred)


def test_ensemble_classifier_chains_rf(small_multilabel):
    clf = make_ecc_rf(n_chains=2, random_state=0, **RF_KWARGS)
    clf.fit(small_multilabel["X_train_sparse"], small_multilabel["Y_train"])
    assert len(clf.chains_) == 2
    pred = clf.predict(small_multilabel["X_test_sparse"])
    _assert_valid_predictions_and_metrics(small_multilabel["Y_test"], small_multilabel["Y_train"], pred)


def test_binary_relevance_nn(small_multilabel):
    clf = make_br_nn(random_state=0, **NN_KWARGS)
    clf.fit(small_multilabel["X_train_dense"], small_multilabel["Y_train"])
    pred = clf.predict(small_multilabel["X_test_dense"])
    _assert_valid_predictions_and_metrics(small_multilabel["Y_test"], small_multilabel["Y_train"], pred)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
