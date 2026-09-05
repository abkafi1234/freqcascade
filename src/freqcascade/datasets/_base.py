"""Shared types and helpers for the dataset loaders.

Every loader in :mod:`freqcascade.datasets.loaders` returns a
:class:`TextDataset`: raw text plus targets, nothing featurized. The
benchmark scripts (``scripts/``) do featurization (``freqcascade.features``)
and cross-validation (``freqcascade.cv``) themselves, so the same loader
output feeds the TF-IDF/RF track and the sentence-embedding/NN track
unchanged.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


def cache_dir() -> Path:
    """Where downloaded corpora and derived artifacts live. Override with
    the ``FREQCASCADE_DATA`` environment variable."""
    root = os.environ.get("FREQCASCADE_DATA")
    path = Path(root) if root else Path.home() / ".cache" / "freqcascade"
    path.mkdir(parents=True, exist_ok=True)
    return path


@dataclass(frozen=True)
class TextDataset:
    """One text-classification benchmark, before featurization.

    ``target`` is ``(n,)`` label array for single-label datasets (labels are
    whatever the source uses -- str or int), or an ``(n, n_labels)`` binary
    indicator matrix for multi-label datasets. ``split`` is a ``(n,)`` array
    of ``"train"``/``"val"``/``"test"`` for datasets evaluated on a fixed
    split (only CLINC150 here); it is ``None`` for datasets evaluated by
    repeated cross-validation over the pooled data.
    """

    name: str
    texts: np.ndarray            # (n,) dtype=object of str
    target: np.ndarray           # (n,) labels  OR  (n, n_labels) int indicator
    label_names: list[str]
    is_multilabel: bool
    split: np.ndarray | None = None
    notes: str = ""

    def __post_init__(self):
        n = len(self.texts)
        if len(self.target) != n:
            raise ValueError(f"{self.name}: {n} texts but {len(self.target)} targets")
        if self.split is not None and len(self.split) != n:
            raise ValueError(f"{self.name}: split has {len(self.split)} entries, expected {n}")
        if self.is_multilabel and self.target.ndim != 2:
            raise ValueError(f"{self.name}: multi-label target must be 2-D, got shape {self.target.shape}")
        if not self.is_multilabel and self.target.ndim != 1:
            raise ValueError(f"{self.name}: single-label target must be 1-D, got shape {self.target.shape}")

    @property
    def n_samples(self) -> int:
        return len(self.texts)

    @property
    def n_classes(self) -> int:
        return self.target.shape[1] if self.is_multilabel else len(np.unique(self.target))

    @property
    def class_frequencies(self) -> np.ndarray:
        """Per-class (single-label) or per-label (multi-label) positive counts,
        descending. Measured on the training rows when there is a fixed split
        (the paper's imbalance ratios are training-set quantities)."""
        target = self.target
        if self.split is not None:
            target = target[self.split == "train"]
        if self.is_multilabel:
            counts = np.asarray(target).sum(axis=0)
        else:
            _, counts = np.unique(target, return_counts=True)
        return np.sort(counts)[::-1]

    @property
    def imbalance_ratio(self) -> float:
        """Max/min class frequency (per-label for multi-label). The headline
        skew number in the paper's dataset table."""
        c = self.class_frequencies
        c = c[c > 0]
        return float(c.max() / c.min()) if len(c) else float("nan")

    def summary(self) -> dict:
        return {
            "name": self.name,
            "n_samples": self.n_samples,
            "n_classes": self.n_classes,
            "multilabel": self.is_multilabel,
            "imbalance_ratio": round(self.imbalance_ratio, 1),
            "has_fixed_split": self.split is not None,
        }


_WS = re.compile(r"\s+")


def clean_text(s: str) -> str:
    """Collapse whitespace; drop control chars. Deliberately minimal -- the
    featurizers (TF-IDF, sentence encoder) do the rest, and every method in a
    comparison must see the same text."""
    if not isinstance(s, str):
        s = "" if s is None else str(s)
    s = s.replace("\x00", " ")
    return _WS.sub(" ", s).strip()


def inject_geometric_imbalance(
    labels: np.ndarray, target_ratio: float, seed: int
) -> np.ndarray:
    """Return row indices selecting a geometrically imbalanced subsample.

    Classes are ranked by their original frequency; the k-th most frequent is
    kept at ``n_max * target_ratio ** (-k / (K - 1))`` examples (capped at
    what's available), sampled without replacement. This is the construction
    the paper uses to put a controlled IR on 20 Newsgroups, which has no
    natural imbalance.
    """
    labels = np.asarray(labels)
    rng = np.random.default_rng(seed)
    classes, counts = np.unique(labels, return_counts=True)
    order = np.argsort(-counts)
    classes, counts = classes[order], counts[order]
    k = len(classes)
    if k < 2:
        return np.arange(len(labels))

    n_max = int(counts[0])
    ranks = np.arange(k)
    targets = np.round(n_max * target_ratio ** (-ranks / (k - 1))).astype(int)
    targets = np.minimum(targets, counts)
    targets = np.maximum(targets, 1)

    keep: list[np.ndarray] = []
    for cls, n_keep in zip(classes, targets):
        idx = np.flatnonzero(labels == cls)
        keep.append(rng.choice(idx, size=int(n_keep), replace=False))
    out = np.concatenate(keep)
    rng.shuffle(out)
    return np.sort(out)


def indicator_matrix(label_lists, label_names: list[str]) -> np.ndarray:
    """List-of-label-lists -> (n, n_labels) int8 indicator, column order given
    by ``label_names``."""
    col = {name: j for j, name in enumerate(label_names)}
    Y = np.zeros((len(label_lists), len(label_names)), dtype=np.int8)
    for i, labs in enumerate(label_lists):
        for lab in labs:
            j = col.get(lab)
            if j is not None:
                Y[i, j] = 1
    return Y


class DatasetUnavailable(RuntimeError):
    """Raised when a corpus can't be fetched automatically and needs a manual
    step. The message says exactly what to do."""
