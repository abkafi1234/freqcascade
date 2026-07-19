"""Cross-validation fold construction, seeded and shared across every
method that consumes a given dataset -- this identity (every baseline and
every proposed method seeing literally the same train/test index arrays
for a given repeat/fold cell) is what makes the paired significance tests
in EXPERIMENTAL_DESIGN.md §6/§6b valid (TIER2_PLAN.md T2.7 names this file
as the shared home for the single-label 5x2 CV harness too; this module
currently implements the multi-label half, T3.3).

Multi-label targets (Reuters/FOCC) can't use `sklearn.model_selection`'s
`StratifiedKFold` -- it only balances a single categorical label, and
Reuters documents can carry up to 16 topics simultaneously (verified,
rfoed/data.py:load_reuters21578). This module ports the iterative-
stratification algorithm of Sechidis, Tsoumakas & Vlahavas (2011), "On the
Stratification of Multi-Label Data", Algorithm 1 (order=1: balances each
label's *marginal* frequency across folds, not higher-order label
co-occurrence patterns -- sufficient for this project's reporting/paired-
test use and matches the ~40-line scope the plan called for).

Why ported instead of using `scikit-multilearn`'s `IterativeStratification`
(tried first, per instructions -- "verify import before relying on it"):
it *imports* and *runs* cleanly under this project's sklearn 1.9, but its
public reproducibility knob is broken. Confirmed directly (2026-07-15):

    >>> IterativeStratification(n_splits=5, order=2, random_state=42).split(X, y)
    ValueError: Setting a random_state has no effect since shuffle is
    False. You should leave random_state to its default (None), or set
    shuffle=True.

`IterativeStratification.__init__` hardcodes `shuffle=False` when calling
`sklearn.model_selection._split._BaseKFold.__init__`, and current sklearn
validates that `random_state` is only meaningful when `shuffle=True` --
so the documented `random_state` parameter cannot be used at all. Worse:
even without that guard, skmultilearn's own tie-breaking
(`iterative_stratification.py`, the `_get_most_desired_combination`/fold
selection steps) calls `np.random.choice(...)` directly, i.e. it reads and
mutates the *global* numpy RNG rather than the `random_state` it claims to
accept -- `random_state` is silently unused even where it's accepted
(confirmed: `check_random_state(self.random_state)` is called and its
result immediately discarded). A same-seed-reproducibility test bears
this out: calling `random_state=42` raises; working around it by calling
global `np.random.seed(42)` before `.split()` *does* reproduce, but that
makes fold identity depend on unrelated code never touching numpy's global
RNG between calls -- exactly the kind of hidden coupling this project's
shared-fold requirement (identical folds for every FOCC method, called at
different times from different scripts) cannot tolerate. Hence the
self-contained port below: every function here takes its own
`np.random.Generator` (or builds one from an explicit integer seed) and
never reads or writes `numpy`'s global random state, so `(y, seed)
-> folds` is a pure, literally-reproducible function.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class FoldSplit:
    """One (repeat, fold) cell of a repeated k-fold CV protocol."""
    repeat: int
    fold: int
    train_idx: np.ndarray
    test_idx: np.ndarray


def _iterative_stratify(
    y: np.ndarray, fold_fractions: list[float], rng: np.random.Generator
) -> list[np.ndarray]:
    """Core routine: Sechidis et al. 2011, Algorithm 1 (order=1).

    Repeatedly finds the label with the fewest remaining unassigned
    examples (ties broken via `rng`), then greedily assigns each of that
    label's remaining examples to whichever fold currently has the
    largest *unmet* desired count for that label (ties broken by overall
    fold deficit, then by `rng`). This directly balances every label's
    per-fold frequency, which naive random splitting does not (rare
    labels with only a handful of positive examples can easily land
    entirely in one fold under random assignment).

    `y`: (n_samples, n_labels) binary indicator matrix. Every row must
    have at least one positive label for the algorithm as specified;
    any all-zero rows (shouldn't occur here since load_reuters21578
    already drops empty-label documents, but handled defensively) are
    assigned last, purely by which fold has the largest remaining size
    deficit.
    `fold_fractions`: desired fraction of the total in each fold; must
    sum to 1.
    Returns a list of index arrays, one per fold, partitioning
    `range(n_samples)`.
    """
    n_samples, n_labels = y.shape
    k = len(fold_fractions)
    fold_fractions = np.asarray(fold_fractions, dtype=float)
    if not np.isclose(fold_fractions.sum(), 1.0):
        raise ValueError(f"fold_fractions must sum to 1, got {fold_fractions.sum()}")

    label_totals = y.sum(axis=0).astype(float)
    c = np.outer(fold_fractions, label_totals)  # (k, n_labels): desired remaining count per label per fold
    c_total = fold_fractions * n_samples  # (k,): desired remaining overall count per fold

    examples_per_label = [set(np.flatnonzero(y[:, i]).tolist()) for i in range(n_labels)]
    label_counts = np.array([len(s) for s in examples_per_label], dtype=float)

    assigned = np.full(n_samples, -1, dtype=int)
    n_unassigned = n_samples

    while n_unassigned > 0:
        active = np.flatnonzero(label_counts > 0)
        if active.size == 0:
            # Leftover examples with no remaining positive label (only
            # possible for all-zero rows). Place by largest remaining
            # fold-size deficit, in a random order.
            leftover = np.flatnonzero(assigned == -1)
            order = rng.permutation(leftover.size)
            for pos in order:
                idx = leftover[pos]
                m = int(np.argmax(c_total))
                assigned[idx] = m
                c_total[m] -= 1
                n_unassigned -= 1
            break

        min_count = label_counts[active].min()
        rarest = active[label_counts[active] == min_count]
        label = int(rarest[rng.integers(rarest.size)])

        idx_list = list(examples_per_label[label])
        order = rng.permutation(len(idx_list))
        for pos in order:
            idx = idx_list[pos]
            if assigned[idx] != -1:
                continue  # shouldn't happen (label's set only holds unassigned idx), kept defensive

            col = c[:, label]
            best = col.max()
            top = np.flatnonzero(col == best)
            if top.size > 1:
                best_total = c_total[top].max()
                top = top[c_total[top] == best_total]
            m = int(top[rng.integers(top.size)]) if top.size > 1 else int(top[0])

            assigned[idx] = m
            n_unassigned -= 1
            for lab in np.flatnonzero(y[idx]):
                c[m, lab] -= 1
                examples_per_label[lab].discard(idx)
                label_counts[lab] -= 1
            c_total[m] -= 1

    return [np.flatnonzero(assigned == j) for j in range(k)]


def iterative_stratified_kfold(
    y: np.ndarray, n_splits: int = 2, seed: int = 0
) -> list[tuple[np.ndarray, np.ndarray]]:
    """One iterative-stratified k-fold partition of `y` (n_docs, n_labels
    binary indicator matrix). Returns `n_splits` (train_idx, test_idx)
    pairs -- standard k-fold semantics, each fold used as the test set
    exactly once. Deterministic: identical `(y, n_splits, seed)` always
    returns identical folds (a fresh `np.random.default_rng(seed)` drives
    every tie-break; no global RNG state is touched)."""
    rng = np.random.default_rng(seed)
    fold_fractions = [1.0 / n_splits] * n_splits
    fold_indices = _iterative_stratify(y, fold_fractions, rng)

    all_idx = np.arange(y.shape[0])
    splits = []
    for i in range(n_splits):
        test_idx = np.sort(fold_indices[i])
        train_idx = np.setdiff1d(all_idx, test_idx, assume_unique=True)
        splits.append((train_idx, test_idx))
    return splits


def repeated_iterative_stratified_kfold(
    y: np.ndarray, n_repeats: int = 5, n_splits: int = 2, seed: int = 0
) -> list[FoldSplit]:
    """The project's standard multi-label CV protocol (design §4/§6b):
    `n_repeats` x `n_splits` repeated iterative-stratified CV -- 5x2 by
    default, matching Dietterich (1998)'s 5x2cv paired-test protocol that
    `rfoed/stats.py` implements downstream. Each repeat draws an
    independent, seeded sub-generator (via `np.random.SeedSequence.spawn`),
    so the entire sequence of `n_repeats * n_splits` folds is a pure
    deterministic function of `seed`. Every FOCC method (baselines and
    proposed) must call this with the *same* `(y, n_repeats, n_splits,
    seed)` to see identical folds -- that identity is what the paired
    significance tests in EXPERIMENTAL_DESIGN.md §6b require. `y` should
    be the *pooled* train+test indicator matrix (see
    `MultiLabelDataset.pooled_Y` in rfoed/data.py), since Reuters/FOCC
    doesn't rely on a fixed split (design §4).
    """
    seed_seq = np.random.SeedSequence(seed)
    child_seeds = seed_seq.spawn(n_repeats)

    all_idx = np.arange(y.shape[0])
    fold_fractions = [1.0 / n_splits] * n_splits
    out: list[FoldSplit] = []
    for r, child in enumerate(child_seeds):
        rng = np.random.default_rng(child)
        fold_indices = _iterative_stratify(y, fold_fractions, rng)
        for f in range(n_splits):
            test_idx = np.sort(fold_indices[f])
            train_idx = np.setdiff1d(all_idx, test_idx, assume_unique=True)
            out.append(FoldSplit(repeat=r, fold=f, train_idx=train_idx, test_idx=test_idx))
    return out


def repeated_stratified_kfold(
    y, n_repeats: int = 5, n_splits: int = 2, seed: int = 0
) -> list[FoldSplit]:
    """T2.7 — the single-label counterpart to
    `repeated_iterative_stratified_kfold` above. Single-label targets
    (one class per document) don't have the multi-label co-occurrence
    problem that motivated porting Sechidis et al. — `sklearn`'s
    `StratifiedKFold` balances a single categorical label correctly and
    its `random_state` actually works (unlike skmultilearn's, see this
    module's docstring), so this is a thin wrapper rather than another
    from-scratch port. Kept in this module (not scattered per-script) so
    every single-label method that needs the design's 5x2 CV protocol
    (EXPERIMENTAL_DESIGN.md §4: 20 Newsgroups, WOS46985 — CLINC150 keeps
    its fixed split + repeated seeds instead) draws from the same
    `(y, n_repeats, n_splits, seed) -> folds` function, which is what
    makes the paired significance tests in §6 valid (every method must
    see literally the same fold indices).

    Returns the same `FoldSplit` dataclass as the multi-label side so
    `save_folds`/`load_folds` and any downstream code work unchanged
    across both tracks."""
    from sklearn.model_selection import RepeatedStratifiedKFold

    y = np.asarray(y)
    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    out: list[FoldSplit] = []
    for i, (train_idx, test_idx) in enumerate(rskf.split(np.zeros(len(y)), y)):
        repeat, fold = divmod(i, n_splits)
        out.append(FoldSplit(repeat=repeat, fold=fold, train_idx=train_idx, test_idx=test_idx))
    return out


def save_folds(folds: list[FoldSplit], path: Path | str) -> None:
    """Persist a fold assignment to a single .npz so every later-added
    FOCC method (T3.4/T3.5) can load the *exact* same folds from disk
    instead of re-deriving them -- belt-and-braces on top of seeded
    determinism, protecting against the fold-generation code itself
    drifting (e.g. an unrelated refactor of `_iterative_stratify`'s
    tie-break order) between when different methods are run."""
    path = Path(path)
    payload: dict[str, np.ndarray] = {}
    repeats, fold_ids = [], []
    for fs in folds:
        payload[f"train_{fs.repeat}_{fs.fold}"] = fs.train_idx
        payload[f"test_{fs.repeat}_{fs.fold}"] = fs.test_idx
        repeats.append(fs.repeat)
        fold_ids.append(fs.fold)
    payload["_meta_repeats"] = np.array(repeats)
    payload["_meta_folds"] = np.array(fold_ids)
    np.savez(path, **payload)


def load_folds(path: Path | str) -> list[FoldSplit]:
    """Inverse of `save_folds`."""
    data = np.load(Path(path))
    repeats = data["_meta_repeats"]
    fold_ids = data["_meta_folds"]
    out = []
    for r, f in zip(repeats.tolist(), fold_ids.tolist()):
        out.append(
            FoldSplit(
                repeat=int(r),
                fold=int(f),
                train_idx=data[f"train_{r}_{f}"],
                test_idx=data[f"test_{r}_{f}"],
            )
        )
    return out
