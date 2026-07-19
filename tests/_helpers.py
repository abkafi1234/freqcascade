"""Shared test doubles for the mock-based test suite. Not collected by
pytest itself (filename doesn't match test_*.py) -- imported directly by
tests/test_mocked_*.py.
"""

from __future__ import annotations

from typing import Callable
from unittest.mock import Mock

import numpy as np


class RecordingBinaryLearner:
    """A fake base learner standing in for RFBaseLearner/NNEnsembleBaseLearner
    in orchestration tests: no real model is ever fit, `predict_proba` just
    evaluates a caller-supplied function of X. `.fit` and `.predict_proba`
    are `unittest.mock.Mock` objects (via `side_effect`), so tests can
    assert on `call_count`, `call_args_list`, etc. -- the point is to test
    RFOEDClassifier/FOCCClassifier's *orchestration* (which rows go to
    which node, in what order, with what data) in complete isolation from
    any actual classifier's behavior.
    """

    def __init__(self, p_positive_fn: Callable[[np.ndarray], np.ndarray]):
        self.p_positive_fn = p_positive_fn
        self.fit_calls: list[tuple] = []
        self.fit = Mock(side_effect=self._fit)
        self.predict_proba = Mock(side_effect=self._predict_proba)

    def _fit(self, X, y):
        self.fit_calls.append((np.asarray(X).copy(), np.asarray(y).copy()))
        return self

    def _predict_proba(self, X):
        p1 = np.asarray(self.p_positive_fn(np.asarray(X)), dtype=float)
        return np.stack([1.0 - p1, p1], axis=1)
