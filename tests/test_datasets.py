"""Dataset-layer tests: the TextDataset contract and the pure helpers
(geometric imbalance injection, indicator-matrix construction). The loaders
themselves hit the network / large downloads and are exercised separately
in the data-ingest step, not in CI.
"""

from __future__ import annotations

import sys

import numpy as np
import pytest

from freqcascade.datasets import MULTI_LABEL, REGISTRY, SINGLE_LABEL, TextDataset
from freqcascade.datasets._base import (
    clean_text,
    indicator_matrix,
    inject_geometric_imbalance,
)


def test_registry_is_split_into_single_and_multi_label():
    assert set(REGISTRY) == set(SINGLE_LABEL) | set(MULTI_LABEL)
    assert not (set(SINGLE_LABEL) & set(MULTI_LABEL))
    assert {"reuters21578", "hoc", "litcovid"} <= set(MULTI_LABEL)
    assert {"clinc150", "20newsgroups", "wos46985", "drug_reviews", "ohsumed_23"} <= set(SINGLE_LABEL)


def test_textdataset_rejects_shape_mismatches():
    with pytest.raises(ValueError, match="targets"):
        TextDataset("x", np.array(["a", "b"], dtype=object), np.array([0]), ["c"], False)
    with pytest.raises(ValueError, match="2-D"):
        TextDataset("x", np.array(["a"], dtype=object), np.array([1]), ["c"], is_multilabel=True)
    with pytest.raises(ValueError, match="1-D"):
        TextDataset("x", np.array(["a"], dtype=object), np.zeros((1, 3)), ["c"], is_multilabel=False)


def test_textdataset_properties_single_and_multi_label():
    sl = TextDataset(
        "sl", np.array(list("aaaabbc"), dtype=object),
        np.array(list("XXXXYYZ")), ["X", "Y", "Z"], False,
    )
    assert sl.n_classes == 3
    assert sl.class_frequencies.tolist() == [4, 2, 1]
    assert sl.imbalance_ratio == 4.0

    Y = np.array([[1, 0, 0], [1, 1, 0], [1, 0, 0]], dtype=np.int8)
    ml = TextDataset("ml", np.array(["a", "b", "c"], dtype=object), Y, ["p", "q", "r"], True)
    assert ml.n_classes == 3
    assert ml.class_frequencies.tolist() == [3, 1, 0]
    assert ml.imbalance_ratio == 3.0  # rarest *present* label has 1


def test_inject_geometric_imbalance_hits_the_target_ratio_and_is_seeded():
    y = np.repeat(np.arange(5), 400)
    idx_a = inject_geometric_imbalance(y, target_ratio=20.0, seed=7)
    idx_b = inject_geometric_imbalance(y, target_ratio=20.0, seed=7)
    np.testing.assert_array_equal(idx_a, idx_b)

    _, counts = np.unique(y[idx_a], return_counts=True)
    assert round(counts.max() / counts.min()) == 20
    # monotone non-increasing across the frequency rank
    assert list(np.sort(counts)[::-1]) == sorted(counts, reverse=True)


def test_indicator_matrix_column_order_follows_label_names():
    Y = indicator_matrix([["b", "a"], [], ["c"]], ["a", "b", "c"])
    np.testing.assert_array_equal(Y, [[1, 1, 0], [0, 0, 0], [0, 0, 1]])
    assert Y.dtype == np.int8


def test_clean_text_collapses_whitespace_and_survives_non_str():
    assert clean_text("  a\n\t b   c ") == "a b c"
    assert clean_text(None) == ""
    assert clean_text(123) == "123"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
