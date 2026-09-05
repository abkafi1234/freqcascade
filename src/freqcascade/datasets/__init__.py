"""Dataset loaders for the freqcascade benchmark suite.

    from freqcascade.datasets import load, REGISTRY
    ds = load("reuters21578")
    ds.summary()   # {'name': ..., 'n_classes': 90, 'multilabel': True, ...}

Every loader returns a :class:`TextDataset` (raw text + targets). Featurization
and cross-validation are done by the benchmark scripts, so the same object
feeds the TF-IDF/RF track and the sentence-embedding/NN track unchanged.

``import freqcascade.datasets`` pulls in only numpy; each loader imports its
own heavy backend (``datasets``, ``nltk``, ``scikit-learn``) lazily.
"""

from __future__ import annotations

from ._base import DatasetUnavailable, TextDataset, cache_dir
from .loaders import MULTI_LABEL, REGISTRY, SINGLE_LABEL, load

__all__ = [
    "TextDataset",
    "DatasetUnavailable",
    "cache_dir",
    "load",
    "REGISTRY",
    "SINGLE_LABEL",
    "MULTI_LABEL",
]
