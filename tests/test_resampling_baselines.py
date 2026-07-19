"""Tests for freqcascade.resampling_baselines (the `[imbalance]` extra).

This module had no test coverage in the package; exercising every
resampler_name directly caught a real bug: a class with a single example
made `_safe_k_neighbors` silently compute k_neighbors=1 (via its
`max(1, ...)` floor) instead of recognizing that SMOTE/ADASYN cannot run
at all in that case, so the failure surfaced ~3 layers down as an opaque
imblearn/sklearn NearestNeighbors error instead of a clear one. These
tests cover the full RESAMPLER_NAMES matrix plus that edge case.

Run with: pip install -e ".[dev]" && pytest tests/test_resampling_baselines.py -v
"""

from __future__ import annotations

import sys

import numpy as np
import pytest

from freqcascade.resampling_baselines import (
    ResamplingBaseline,
    adasyn_floored_minority_strategy,
    easy_ensemble_classifier,
    rusboost_classifier,
)


def _imbalanced_data(seed=0, minority_n=30):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(200 + minority_n, 8))
    y = np.array([0] * 150 + [1] * 50 + [2] * minority_n)
    return X, y


@pytest.mark.parametrize("resampler_name", ResamplingBaseline.RESAMPLER_NAMES)
def test_every_resampler_runs_end_to_end(resampler_name):
    X, y = _imbalanced_data(minority_n=30)
    rb = ResamplingBaseline(resampler_name=resampler_name, n_estimators=20, n_jobs=1, random_state=0)
    rb.fit(X, y)

    pred = rb.predict(X)
    proba = rb.predict_proba(X)
    assert pred.shape == (len(y),)
    assert proba.shape == (len(y), 3)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)


def test_rejects_unknown_resampler_name():
    with pytest.raises(ValueError):
        ResamplingBaseline(resampler_name="not_a_real_resampler")


@pytest.mark.parametrize("resampler_name", ["smote", "adasyn", "smoteenn"])
def test_k_neighbors_is_capped_for_a_small_minority_class(resampler_name):
    X, y = _imbalanced_data(minority_n=4)  # smallest class has 4 examples -> k must cap at 3
    rb = ResamplingBaseline(resampler_name=resampler_name, n_estimators=20, n_jobs=1, random_state=0)
    rb.fit(X, y)
    assert rb.k_neighbors_used_ == 3
    assert rb.capped_ is True


@pytest.mark.parametrize("resampler_name", ["smote", "adasyn", "smoteenn"])
def test_singleton_minority_class_raises_a_clear_actionable_error(resampler_name):
    """A class with exactly 1 example has zero valid neighbors -- SMOTE
    cannot run at all, and this must fail with a clear message pointing
    at the real constraint, not imblearn's much more opaque internal
    NearestNeighbors error."""
    X, y = _imbalanced_data(minority_n=1)
    rb = ResamplingBaseline(resampler_name=resampler_name, n_estimators=20, n_jobs=1, random_state=0)
    with pytest.raises(ValueError, match="at least 2 examples"):
        rb.fit(X, y)


def test_none_resampler_is_a_pure_passthrough_random_forest():
    X, y = _imbalanced_data()
    rb = ResamplingBaseline(resampler_name="none", n_estimators=20, n_jobs=1, random_state=0)
    rb.fit(X, y)
    assert rb.k_neighbors_used_ is None
    assert rb.capped_ is False


def test_adasyn_floored_minority_strategy_targets_only_below_median_classes():
    y = np.array([0] * 100 + [1] * 50 + [2] * 10)  # counts: 100, 50, 10 -> median 50
    strategy = adasyn_floored_minority_strategy(y)
    # Only class 2 (10 < 50) should be targeted, raised to the median (50).
    assert strategy == {2: 50}


def test_easy_ensemble_and_rusboost_run_end_to_end():
    X, y = _imbalanced_data(minority_n=30)
    for clf in (
        easy_ensemble_classifier(n_estimators=5, random_state=0),
        rusboost_classifier(n_estimators=5, random_state=0),
    ):
        clf.fit(X, y)
        pred = clf.predict(X)
        assert pred.shape == (len(y),)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
