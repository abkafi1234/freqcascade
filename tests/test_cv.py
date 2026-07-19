"""Tests for freqcascade.cv -- the iterative-stratified (multi-label) and
repeated-stratified (single-label) cross-validation fold builders.

This module previously had NO test coverage in this package, and a
ruff `E741` lint pass caught a real bug: `_iterative_stratify`'s inner
loop referenced an undefined name (`c[:, l]`, where the loop variable had
been named `label`), which meant `iterative_stratified_kfold` /
`repeated_iterative_stratified_kfold` raised `NameError` on essentially
every call -- i.e. the entire multi-label CV path was non-functional.
These tests exist to make sure that doesn't silently regress again.

Run with: pip install -e ".[dev]" && pytest tests/test_cv.py -v
"""

from __future__ import annotations

import sys

import numpy as np
import pytest

from freqcascade.cv import (
    FoldSplit,
    iterative_stratified_kfold,
    load_folds,
    repeated_iterative_stratified_kfold,
    repeated_stratified_kfold,
    save_folds,
)


def _synthetic_multilabel(n=200, n_labels=6, seed=0):
    rng = np.random.default_rng(seed)
    Y = (rng.random((n, n_labels)) < 0.15).astype(int)
    # No all-zero rows -- matches the module's documented assumption.
    empty = Y.sum(axis=1) == 0
    Y[empty, 0] = 1
    return Y


# ---------------------------------------------------------------------------
# iterative_stratified_kfold (multi-label)
# ---------------------------------------------------------------------------


def test_folds_partition_every_index_exactly_once():
    Y = _synthetic_multilabel()
    n = Y.shape[0]
    splits = iterative_stratified_kfold(Y, n_splits=5, seed=42)

    assert len(splits) == 5
    all_test = np.concatenate([test for _, test in splits])
    assert sorted(all_test.tolist()) == list(range(n))  # every index appears in exactly one test fold


def test_train_and_test_are_disjoint_and_cover_everything_within_each_fold():
    Y = _synthetic_multilabel()
    n = Y.shape[0]
    for train_idx, test_idx in iterative_stratified_kfold(Y, n_splits=4, seed=1):
        assert set(train_idx.tolist()) & set(test_idx.tolist()) == set()
        assert len(train_idx) + len(test_idx) == n


def test_deterministic_given_same_seed():
    Y = _synthetic_multilabel()
    a = iterative_stratified_kfold(Y, n_splits=3, seed=123)
    b = iterative_stratified_kfold(Y, n_splits=3, seed=123)
    for (tr_a, te_a), (tr_b, te_b) in zip(a, b):
        np.testing.assert_array_equal(tr_a, tr_b)
        np.testing.assert_array_equal(te_a, te_b)


def test_different_seeds_give_different_folds():
    Y = _synthetic_multilabel()
    a = iterative_stratified_kfold(Y, n_splits=3, seed=1)
    b = iterative_stratified_kfold(Y, n_splits=3, seed=2)
    assert any(not np.array_equal(te_a, te_b) for (_, te_a), (_, te_b) in zip(a, b))


def test_rare_label_is_spread_across_folds_not_dumped_in_one():
    """The whole point of iterative stratification over naive random
    splitting: a label with only a handful of positives should still
    appear in every fold's test set, not collapse into a single one."""
    rng = np.random.default_rng(0)
    n, n_labels = 300, 4
    Y = (rng.random((n, n_labels)) < 0.3).astype(int)
    # Make label 0 genuinely rare: only 10 positives total.
    Y[:, 0] = 0
    rare_positions = rng.choice(n, size=10, replace=False)
    Y[rare_positions, 0] = 1
    empty = Y.sum(axis=1) == 0
    Y[empty, 1] = 1

    splits = iterative_stratified_kfold(Y, n_splits=5, seed=0)
    per_fold_rare_count = [int(Y[test_idx, 0].sum()) for _, test_idx in splits]
    assert all(c >= 1 for c in per_fold_rare_count), per_fold_rare_count


def test_rejects_fold_fractions_that_do_not_sum_to_one():
    from freqcascade.cv import _iterative_stratify

    Y = _synthetic_multilabel(n=20)
    with pytest.raises(ValueError):
        _iterative_stratify(Y, [0.5, 0.4], np.random.default_rng(0))


# ---------------------------------------------------------------------------
# repeated_iterative_stratified_kfold
# ---------------------------------------------------------------------------


def test_repeated_produces_n_repeats_times_n_splits_cells():
    Y = _synthetic_multilabel()
    out = repeated_iterative_stratified_kfold(Y, n_repeats=3, n_splits=2, seed=7)
    assert len(out) == 6
    assert all(isinstance(fs, FoldSplit) for fs in out)
    assert sorted((fs.repeat, fs.fold) for fs in out) == [(r, f) for r in range(3) for f in range(2)]


def test_repeated_folds_disjoint_within_each_repeat():
    Y = _synthetic_multilabel()
    out = repeated_iterative_stratified_kfold(Y, n_repeats=2, n_splits=2, seed=7)
    n = Y.shape[0]
    for r in range(2):
        cells = [fs for fs in out if fs.repeat == r]
        all_test = np.concatenate([fs.test_idx for fs in cells])
        assert sorted(all_test.tolist()) == list(range(n))


def test_repeats_use_independent_seeds_not_identical_partitions():
    Y = _synthetic_multilabel()
    out = repeated_iterative_stratified_kfold(Y, n_repeats=2, n_splits=2, seed=7)
    fold0_repeat0 = next(fs for fs in out if fs.repeat == 0 and fs.fold == 0)
    fold0_repeat1 = next(fs for fs in out if fs.repeat == 1 and fs.fold == 0)
    assert not np.array_equal(fold0_repeat0.test_idx, fold0_repeat1.test_idx)


def test_repeated_is_deterministic_given_same_seed():
    Y = _synthetic_multilabel()
    a = repeated_iterative_stratified_kfold(Y, n_repeats=2, n_splits=2, seed=99)
    b = repeated_iterative_stratified_kfold(Y, n_repeats=2, n_splits=2, seed=99)
    for fs_a, fs_b in zip(a, b):
        np.testing.assert_array_equal(fs_a.train_idx, fs_b.train_idx)
        np.testing.assert_array_equal(fs_a.test_idx, fs_b.test_idx)


# ---------------------------------------------------------------------------
# repeated_stratified_kfold (single-label, thin sklearn wrapper)
# ---------------------------------------------------------------------------


def test_repeated_stratified_kfold_single_label():
    y = np.array([0] * 40 + [1] * 30 + [2] * 10)
    out = repeated_stratified_kfold(y, n_repeats=2, n_splits=5, seed=0)

    assert len(out) == 10  # 2 repeats x 5 splits
    n = len(y)
    for r in range(2):
        cells = [fs for fs in out if fs.repeat == r]
        assert len(cells) == 5
        all_test = np.concatenate([fs.test_idx for fs in cells])
        assert sorted(all_test.tolist()) == list(range(n))


# ---------------------------------------------------------------------------
# save_folds / load_folds round-trip
# ---------------------------------------------------------------------------


def test_save_and_load_folds_round_trip(tmp_path):
    Y = _synthetic_multilabel(n=50)
    folds = repeated_iterative_stratified_kfold(Y, n_repeats=2, n_splits=2, seed=3)
    path = tmp_path / "folds.npz"
    save_folds(folds, path)
    loaded = load_folds(path)

    assert len(loaded) == len(folds)

    def sort_key(fs):
        return (fs.repeat, fs.fold)

    for fs_orig, fs_loaded in zip(sorted(folds, key=sort_key), sorted(loaded, key=sort_key)):
        assert fs_orig.repeat == fs_loaded.repeat
        assert fs_orig.fold == fs_loaded.fold
        np.testing.assert_array_equal(fs_orig.train_idx, fs_loaded.train_idx)
        np.testing.assert_array_equal(fs_orig.test_idx, fs_loaded.test_idx)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
