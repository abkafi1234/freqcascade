"""Per-node base-learner tests: the balanced-bootstrap cap is shared and
configurable across all three neural base learners, and
`_align_proba_columns` rejects non-binary targets.

Run with: pytest tests/test_base_learners.py -v
"""

from __future__ import annotations

import sys
from unittest.mock import patch

import numpy as np
import pytest

from freqcascade.base_learners import NNEnsembleBaseLearner, _align_proba_columns
from freqcascade.rebalance import balanced_bootstrap_indices


def _skewed_binary(n_pos=40, n_neg=3000, n_features=5, seed=0):
    rng = np.random.default_rng(seed)
    y = np.array([1] * n_pos + [0] * n_neg)
    X = rng.normal(size=(len(y), n_features))
    return X, y


def test_balanced_bootstrap_cap_bounds_each_class_draw():
    y = np.array([1] * 20 + [0] * 5000)
    idx = balanced_bootstrap_indices(y, np.random.default_rng(0), max_per_class=100)
    drawn = y[idx]
    assert len(idx) == 200
    assert (drawn == 0).sum() == 100
    assert (drawn == 1).sum() == 100


def test_balanced_bootstrap_uncapped_draws_up_to_the_majority_class():
    y = np.array([1] * 20 + [0] * 300)
    idx = balanced_bootstrap_indices(y, np.random.default_rng(0))
    assert len(idx) == 600  # 300 per class


def test_nn_ensemble_default_cap_is_2000():
    assert NNEnsembleBaseLearner().max_bootstrap_per_class == 2000


@pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning")
def test_nn_ensemble_threads_the_cap_into_each_members_balanced_draw():
    X, y = _skewed_binary()
    learner = NNEnsembleBaseLearner(
        n_members=2, max_iter=10, random_state=0, max_bootstrap_per_class=64
    )
    seen: list = []
    real = balanced_bootstrap_indices

    def spy(y_, rng_, max_per_class=None):
        seen.append(max_per_class)
        return real(y_, rng_, max_per_class=max_per_class)

    with patch("freqcascade.base_learners.balanced_bootstrap_indices", side_effect=spy):
        learner.fit(X, y)

    assert seen == [64, 64]  # one balanced draw per member, at the configured cap


@pytest.mark.parametrize("import_path", ["torch_ensemble", "text_cnn"])
def test_torch_base_learners_expose_the_same_cap_param(import_path):
    # __init__ only -- constructing these needs no torch.
    if import_path == "torch_ensemble":
        from freqcascade.torch_ensemble import TorchNNEnsembleBaseLearner as C
    else:
        from freqcascade.text_cnn import TextCNNEnsembleBaseLearner as C
    assert C().max_bootstrap_per_class == 2000
    assert C(max_bootstrap_per_class=123).max_bootstrap_per_class == 123


def test_align_proba_columns_rejects_non_binary_labels():
    proba = np.array([[0.2, 0.8], [0.6, 0.4]])
    with pytest.raises(ValueError, match="binary"):
        _align_proba_columns(proba, np.array([1, 2]))


def test_align_proba_columns_accepts_zero_one():
    out = _align_proba_columns(np.array([[0.3, 0.7]]), np.array([0, 1]))
    np.testing.assert_allclose(out, [[0.3, 0.7]])


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
