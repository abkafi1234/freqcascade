"""Mock-based tests for RFOEDClassifier's orchestration logic: which rows
get routed to which node, in what order, with what data -- verified with
recording fakes (tests/_helpers.RecordingBinaryLearner) instead of real
classifiers, so assertions are exact and don't depend on any model
actually learning anything. Complements test_decomposition.py, which uses
real RandomForestClassifier-backed learners and checks end-to-end output
validity/metrics rather than exact interaction contracts.

Run with: pytest tests/test_mocked_decomposition.py -v
"""

from __future__ import annotations

import sys

import numpy as np
import pytest
from unittest.mock import Mock, call

from freqcascade.decomposition import RFOEDClassifier
from _helpers import RecordingBinaryLearner


def _toy_data():
    # 3 classes, frequency order by construction: 0 (3x), 1 (2x), 2 (1x).
    X = np.arange(6).reshape(6, 1).astype(float)
    y = np.array([0, 0, 0, 1, 1, 2])
    return X, y


def test_first_node_fits_on_full_active_set():
    X, y = _toy_data()
    learners = []

    def factory(i):
        learner = RecordingBinaryLearner(lambda X: np.full(X.shape[0], 0.5))
        learners.append(learner)
        return learner

    clf = RFOEDClassifier(base_learner_factory=factory, order="frequency", random_state=0)
    clf.fit(X, y)

    # 3 classes -> 2 explicit peel nodes (class 2 is the implicit residual).
    assert len(learners) == 2
    node0_X, node0_y = learners[0].fit_calls[0]
    np.testing.assert_array_equal(node0_X, X)
    np.testing.assert_array_equal(node0_y, (y == 0).astype(int))


def test_second_node_fits_only_on_rows_unresolved_by_the_first():
    X, y = _toy_data()
    learners = []

    def factory(i):
        learner = RecordingBinaryLearner(lambda X: np.full(X.shape[0], 0.5))
        learners.append(learner)
        return learner

    clf = RFOEDClassifier(base_learner_factory=factory, order="frequency", random_state=0)
    clf.fit(X, y)

    node1_X, node1_y = learners[1].fit_calls[0]
    # Rows where y != 0 -- i.e. the class-0 rows must have been removed
    # from contention before node 1 (class 1) ever sees the data.
    expected_X = X[y != 0]
    expected_y = (y[y != 0] == 1).astype(int)
    np.testing.assert_array_equal(node1_X, expected_X)
    np.testing.assert_array_equal(node1_y, expected_y)


def test_base_learner_factory_called_once_per_node_with_sequential_index():
    X, y = _toy_data()
    factory = Mock(side_effect=lambda i: RecordingBinaryLearner(lambda X: np.full(X.shape[0], 0.5)))

    clf = RFOEDClassifier(base_learner_factory=factory, order="frequency", random_state=0)
    clf.fit(X, y)

    assert factory.call_args_list == [call(0), call(1)]


def test_cascade_stops_calling_later_nodes_once_everything_is_resolved():
    """If every active row resolves at node 0, node 1's predict_proba
    should never be invoked at predict-time -- the `if not
    active_mask.any(): break` short-circuit in RFOEDClassifier.predict."""
    X_train = np.zeros((4, 1))
    y_train = np.array([0, 0, 1, 2])  # order: 0 (2x), 1 (1x), 2 (1x) -> 2 nodes

    learners = []

    def factory(i):
        # Node 0 always fires positive -> resolves everything immediately.
        p = 1.0 if i == 0 else 0.5
        learner = RecordingBinaryLearner(lambda X, p=p: np.full(X.shape[0], p))
        learners.append(learner)
        return learner

    clf = RFOEDClassifier(base_learner_factory=factory, order="frequency", random_state=0)
    clf.fit(X_train, y_train)
    X_test = np.zeros((5, 1))
    pred = clf.predict(X_test)

    np.testing.assert_array_equal(pred, np.zeros(5, dtype=pred.dtype))
    # node 0's predict_proba is called once for training diagnostics and
    # once at predict-time; node 1 is only ever touched during its own
    # training diagnostics (once) -- never at predict-time, since nothing
    # was left active for it to see.
    assert learners[1].predict_proba.call_count == 1  # training diagnostics only


def test_threshold_boundary_routes_exactly_at_the_cutoff():
    X_train = np.zeros((5, 1))
    y_train = np.array([0, 0, 0, 1, 1])  # class_order_ = [0, 1], 1 explicit node

    node_p = {}

    def factory(i):
        learner = RecordingBinaryLearner(lambda X: X[:, 0])
        node_p[i] = learner
        return learner

    clf = RFOEDClassifier(base_learner_factory=factory, order="frequency", random_state=0, threshold=0.5)
    clf.fit(X_train, y_train)

    X_test = np.array([[0.49], [0.50], [0.51]])
    pred = clf.predict(X_test)

    # p=0.49 -> below threshold -> falls through to the implicit tail
    # class (1); p=0.50 and p=0.51 -> >= threshold -> resolved as class 0.
    np.testing.assert_array_equal(pred, np.array([1, 0, 0]))


def test_prior_correct_elkan_formula_matches_hand_computation():
    """`_correct_p_positive` implements Elkan (2001)'s prior-recalibration
    formula for a node trained on an exactly-balanced (p_train=0.5)
    bootstrap. Bypasses fit() entirely -- constructs a classifier in a
    pre-fitted state with hand-picked diagnostics, so the formula itself
    can be checked against known algebraic properties without any
    training noise."""
    clf = RFOEDClassifier(base_learner_factory=lambda i: None, prior_correct=True)
    clf.node_diagnostics_ = [{"n_positive": 10, "n_active": 1000}]  # prior_true = 0.01

    # A raw score of exactly 0.5 (what a perfectly-balanced-trained
    # classifier outputs when maximally uncertain) must map back to
    # exactly the true prior -- this is the formula's defining property.
    out = clf._correct_p_positive(0, np.array([0.5]))
    np.testing.assert_allclose(out, [0.01], atol=1e-12)

    # p=1.0 (fully confident positive) and p=0.0 (fully confident
    # negative) must be fixed points of the correction.
    np.testing.assert_allclose(clf._correct_p_positive(0, np.array([1.0])), [1.0], atol=1e-12)
    np.testing.assert_allclose(clf._correct_p_positive(0, np.array([0.0])), [0.0], atol=1e-12)


def test_prior_correct_false_is_a_no_op():
    clf = RFOEDClassifier(base_learner_factory=lambda i: None, prior_correct=False)
    clf.node_diagnostics_ = [{"n_positive": 10, "n_active": 1000}]
    p = np.array([0.1, 0.5, 0.9])
    out = clf._correct_p_positive(0, p)
    np.testing.assert_array_equal(out, p)


def test_tail_classifier_handles_rows_unresolved_after_cascade_cap():
    """Constructs a pre-fitted RFOEDClassifier by hand (bypassing fit())
    to test predict()'s tail-routing branch in isolation: a mock tail
    classifier with known predict_proba/classes_ should receive exactly
    the rows the capped cascade left unresolved."""
    from freqcascade.decomposition import _Node

    clf = RFOEDClassifier(base_learner_factory=lambda i: None, cascade_cap=1, tail_learner_factory=lambda: None)
    clf.class_order_ = [0, 1, 2]
    clf.label_dtype_ = np.array([0, 1, 2]).dtype

    never_fires = RecordingBinaryLearner(lambda X: np.zeros(X.shape[0]))
    clf.nodes_ = [_Node(class_label=0, classifier=never_fires)]

    tail = Mock()
    tail.predict_proba = Mock(return_value=np.array([[0.1, 0.2, 0.7], [0.8, 0.1, 0.1]]))
    tail.classes_ = np.array([0, 1, 2])
    clf.tail_classifier_ = tail

    X_test = np.zeros((2, 1))
    pred = clf.predict(X_test)

    np.testing.assert_array_equal(pred, np.array([2, 0]))
    tail.predict_proba.assert_called_once()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
