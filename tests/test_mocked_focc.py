"""Mock-based tests for FOCCClassifier's chain orchestration: verifying
that fit() augments each link with *ground-truth* earlier labels while
predict() augments with each link's own *predicted* labels -- the one
subtlety (Read et al. 2011 classifier chains) that's easy to get backwards
and that a black-box output-only test would never catch. Uses recording
fakes (tests/_helpers.RecordingBinaryLearner), no real classifiers.

Run with: pytest tests/test_mocked_focc.py -v
"""

from __future__ import annotations

import sys

import numpy as np
import pytest
from unittest.mock import Mock, call

from freqcascade.focc import FOCCClassifier
from _helpers import RecordingBinaryLearner


def _toy_multilabel():
    # 4 docs, 3 labels. Frequency order by column sums: label1 (3), label0
    # (2), label2 (1) -> label_order_ should be [1, 0, 2].
    X = np.arange(4).reshape(4, 1).astype(float)
    Y = np.array(
        [
            [1, 1, 0],
            [0, 1, 0],
            [1, 1, 1],
            [0, 0, 0],
        ]
    )
    return X, Y


def test_label_order_is_descending_frequency():
    X, Y = _toy_multilabel()
    learners = []

    def factory(rank, rebalance):
        learner = RecordingBinaryLearner(lambda X: np.full(X.shape[0], 0.5))
        learners.append(learner)
        return learner

    clf = FOCCClassifier(base_learner_factory=factory, order="frequency", random_state=0)
    clf.fit(X, Y)

    assert clf.label_order_ == [1, 0, 2]


def test_fit_augments_each_link_with_ground_truth_columns_not_predictions():
    X, Y = _toy_multilabel()
    learners = []

    def factory(rank, rebalance):
        learner = RecordingBinaryLearner(lambda X: np.full(X.shape[0], 0.5))
        learners.append(learner)
        return learner

    clf = FOCCClassifier(base_learner_factory=factory, order="frequency", random_state=0)
    clf.fit(X, Y)

    # label_order_ = [1, 0, 2]. Link 0 (label 1) sees no augmentation.
    link0_X, link0_y = learners[0].fit_calls[0]
    np.testing.assert_array_equal(link0_X, X)
    np.testing.assert_array_equal(link0_y, Y[:, 1])

    # Link 1 (label 0) must be augmented with the *true* label-1 column
    # (Y[:, 1]), since fit() uses ground truth, not link 0's predictions.
    link1_X, link1_y = learners[1].fit_calls[0]
    np.testing.assert_array_equal(link1_X[:, -1], Y[:, 1])
    np.testing.assert_array_equal(link1_y, Y[:, 0])

    # Link 2 (label 2) is augmented with true labels 1 and 0, in that order.
    link2_X, link2_y = learners[2].fit_calls[0]
    np.testing.assert_array_equal(link2_X[:, -2], Y[:, 1])
    np.testing.assert_array_equal(link2_X[:, -1], Y[:, 0])
    np.testing.assert_array_equal(link2_y, Y[:, 2])


def test_predict_augments_with_own_predictions_not_ground_truth():
    """Link 0 is configured to always predict positive regardless of X,
    which disagrees with the true label-1 column on rows 1 and 3 of the
    toy data. Link 1 must see link 0's (wrong) *predicted* column at
    inference time, never the true one -- ground truth isn't available
    at predict-time by construction."""
    X, Y = _toy_multilabel()
    learners = []

    def factory(rank, rebalance):
        # Link 0 (label 1): always fires positive, disagreeing with truth
        # on rows where Y[:,1] == 0. Link 1: value doesn't matter here.
        p = 1.0 if rank == 0 else 0.5
        learner = RecordingBinaryLearner(lambda X, p=p: np.full(X.shape[0], p))
        learners.append(learner)
        return learner

    clf = FOCCClassifier(base_learner_factory=factory, order="frequency", random_state=0)
    clf.fit(X, Y)

    X_test = X  # reuse the same 4 rows for inference
    clf.predict(X_test)

    # fit() only ever calls learner.fit, never predict_proba, so this is
    # link 1's first and only predict_proba call -- the inference-time one.
    assert learners[1].predict_proba.call_count == 1
    X_aug_at_predict = learners[1].predict_proba.call_args[0][0]
    # Link 0 always predicts positive -> augmentation column is all 1s,
    # even though the true label-1 column (Y[:, 1]) is [1, 1, 1, 0].
    np.testing.assert_array_equal(X_aug_at_predict[:, -1], np.ones(4))


def test_link_factory_receives_rank_and_rebalance_flag():
    X, Y = _toy_multilabel()
    factory = Mock(side_effect=lambda rank, rebalance: RecordingBinaryLearner(lambda X: np.full(X.shape[0], 0.5)))

    clf = FOCCClassifier(base_learner_factory=factory, order="frequency", rebalance=True, random_state=0)
    clf.fit(X, Y)

    assert factory.call_args_list == [call(0, True), call(1, True), call(2, True)]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
