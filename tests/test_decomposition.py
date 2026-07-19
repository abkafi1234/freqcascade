"""Correctness smoke tests for freqcascade.decomposition (RFOEDClassifier,
HierarchicalRFOEDClassifier), using synthetic imbalanced multiclass data.

Run with: pip install -e ".[dev]" && pytest tests/test_decomposition.py -v
"""

from __future__ import annotations

import sys

import numpy as np
import pytest
from sklearn.datasets import make_classification

from freqcascade.base_learners import RFBaseLearner
from freqcascade.decomposition import HierarchicalRFOEDClassifier, RFOEDClassifier
from freqcascade.metrics import evaluate

N_CLASSES = 6
RF_KWARGS = dict(n_estimators=30, n_jobs=4)


@pytest.fixture(scope="module")
def imbalanced_multiclass():
    """A synthetic, deliberately imbalanced multiclass dataset: class
    sizes decay geometrically so every ordering/rebalancing code path
    gets exercised against real skew, not a balanced toy problem."""
    rng = np.random.default_rng(0)
    X_parts, y_parts = [], []
    class_sizes = np.geomspace(300, 20, num=N_CLASSES).astype(int)
    for cls, n in enumerate(class_sizes):
        Xc, _ = make_classification(
            n_samples=n, n_features=15, n_informative=8, n_classes=1,
            n_clusters_per_class=1, random_state=cls,
        )
        # Shift each class's cluster so classes are actually separable.
        Xc = Xc + rng.normal(scale=4.0, size=15) * cls
        X_parts.append(Xc)
        y_parts.append(np.full(n, cls))
    X = np.vstack(X_parts)
    y = np.concatenate(y_parts)
    idx = rng.permutation(len(y))
    X, y = X[idx], y[idx]
    split = int(0.7 * len(y))
    return dict(X_train=X[:split], y_train=y[:split], X_test=X[split:], y_test=y[split:])


def _fit_predict(X_train, y_train, X_test, order="frequency", decision="cascade", rebalance=True, **extra):
    clf = RFOEDClassifier(
        base_learner_factory=lambda i: RFBaseLearner(rebalance=rebalance, random_state=i, **RF_KWARGS),
        order=order, decision=decision, random_state=0, **extra,
    )
    clf.fit(X_train, y_train)
    return clf, clf.predict(X_test)


@pytest.mark.parametrize("order", ["frequency", "random"])
@pytest.mark.parametrize("decision", ["cascade", "argmax_proba"])
@pytest.mark.parametrize("rebalance", [True, False])
def test_rfoed_runs_and_produces_valid_predictions(imbalanced_multiclass, order, decision, rebalance):
    d = imbalanced_multiclass
    clf, pred = _fit_predict(d["X_train"], d["y_train"], d["X_test"], order=order, decision=decision, rebalance=rebalance)

    assert len(pred) == len(d["y_test"])
    assert set(pred.tolist()) <= set(np.unique(d["y_train"]).tolist())
    assert len(clf.nodes_) == N_CLASSES - 1
    assert len(clf.class_order_) == N_CLASSES
    assert sorted(clf.class_order_) == sorted(np.unique(d["y_train"]).tolist())

    metrics = evaluate(d["y_test"], pred, d["y_train"])
    for k, v in metrics.items():
        assert not np.isnan(v), f"{k} is NaN"
        assert -1.0 <= v <= 1.0, f"{k}={v} out of a sane metric range"


def test_frequency_order_matches_training_class_counts(imbalanced_multiclass):
    d = imbalanced_multiclass
    clf, _ = _fit_predict(d["X_train"], d["y_train"], d["X_test"], order="frequency")
    classes, counts = np.unique(d["y_train"], return_counts=True)
    expected = list(classes[np.argsort(-counts)])
    assert clf.class_order_ == expected


def test_predict_proba_rows_sum_to_one(imbalanced_multiclass):
    d = imbalanced_multiclass
    clf, _ = _fit_predict(d["X_train"], d["y_train"], d["X_test"])
    proba = clf.predict_proba(d["X_test"])
    assert proba.shape == (len(d["y_test"]), N_CLASSES)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)


def test_cascade_cap_uses_tail_classifier(imbalanced_multiclass):
    from sklearn.ensemble import RandomForestClassifier

    d = imbalanced_multiclass
    clf = RFOEDClassifier(
        base_learner_factory=lambda i: RFBaseLearner(rebalance=True, random_state=i, **RF_KWARGS),
        order="frequency", decision="cascade", random_state=0,
        cascade_cap=2,
        tail_learner_factory=lambda: RandomForestClassifier(n_estimators=30, random_state=0, n_jobs=4),
    )
    clf.fit(d["X_train"], d["y_train"])
    assert len(clf.nodes_) == 2
    assert clf.tail_classifier_ is not None
    pred = clf.predict(d["X_test"])
    assert set(pred.tolist()) <= set(np.unique(d["y_train"]).tolist())


def test_hierarchical_rfoed(imbalanced_multiclass):
    d = imbalanced_multiclass
    # 2 coarse groups, unrelated to class identity -- just needs to be a
    # valid grouping to exercise the group-then-leaf-cascade code path.
    groups_train = (d["y_train"] % 2).astype(str)

    clf = HierarchicalRFOEDClassifier(
        base_learner_factory=lambda i: RFBaseLearner(rebalance=True, random_state=i, **RF_KWARGS),
        order="frequency", random_state=0,
    )
    clf.fit(d["X_train"], d["y_train"], groups_train)
    pred = clf.predict(d["X_test"])
    assert len(pred) == len(d["y_test"])
    assert set(pred.tolist()) <= set(np.unique(d["y_train"]).tolist())
    with pytest.raises(NotImplementedError):
        clf.predict_proba(d["X_test"])


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
