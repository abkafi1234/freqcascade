"""Per-node binary base learners. RFBaseLearner and NNEnsembleBaseLearner
are the RFOED-RF / RFOED-NN instantiations of the same idea: a bagged
ensemble of members, each fit on its own bootstrap draw, where the
`rebalance` flag controls whether that draw is a stratified balanced
bootstrap (Chen et al. 2004 Balanced-RF style) or an ordinary one. This
keeps "base learner type" and "rebalance on/off" as orthogonal factors,
matching the 2x2x2 ablation in EXPERIMENTAL_DESIGN.md §3."""

from __future__ import annotations

from typing import Protocol

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

from .rebalance import balanced_bootstrap_indices


class BaseLearner(Protocol):
    def fit(self, X, y: np.ndarray) -> "BaseLearner": ...
    def predict_proba(self, X) -> np.ndarray: ...


class _SingleClassStub:
    """Fallback for a degenerate bootstrap draw containing only one
    class — returns that class with probability 1 rather than raising."""

    def __init__(self, only_class: int):
        self.only_class = only_class

    def predict_proba(self, X) -> np.ndarray:
        n = X.shape[0]
        out = np.zeros((n, 2))
        out[:, self.only_class] = 1.0
        return out


def _plain_bootstrap_indices(y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    n = len(y)
    return rng.integers(0, n, size=n)


class RFBaseLearner:
    """RFOED-RF's node classifier. `rebalance=True` reproduces Balanced
    Random Forest (Chen et al. 2004): each tree's bootstrap is stratified
    to equalize the two node classes. `rebalance=False` is plain RF,
    used for the structure-only (SBC-equivalent) ablation cells."""

    def __init__(
        self,
        n_estimators: int = 200,
        rebalance: bool = True,
        max_depth: int | None = None,
        min_samples_leaf: int = 1,
        random_state: int | None = None,
        n_jobs: int = 4,
    ):
        self.n_estimators = n_estimators
        self.rebalance = rebalance
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        # Bounded, not -1: RFOEDClassifier fits one of these per node (up
        # to ~150 nodes on WOS46985). n_jobs=-1 spawns a fresh 28-worker
        # loky process pool on *every* node fit, which — measured on this
        # machine — thrashes RAM/swap long before it finishes rather than
        # actually speeding anything up. A small bounded pool avoids that;
        # cross-node parallelism (nodes are independent) is a better place
        # to spend cores than intra-tree parallelism, and is not done here yet.
        self.n_jobs = n_jobs
        self._model: RandomForestClassifier | None = None

    def fit(self, X, y: np.ndarray) -> "RFBaseLearner":
        # Balanced bootstrap is implemented via sklearn's built-in
        # `class_weight="balanced_subsample"`, which reweights each
        # tree's bootstrap sample by inverse class frequency — the
        # standard scikit-learn-native equivalent of Chen et al.'s
        # Balanced Random Forest.
        class_weight = "balanced_subsample" if self.rebalance else None
        self._model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            class_weight=class_weight,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )
        self._model.fit(X, y)
        return self

    def predict_proba(self, X) -> np.ndarray:
        assert self._model is not None, "call fit() first"
        proba = self._model.predict_proba(X)
        return _align_proba_columns(proba, self._model.classes_)


class NNEnsembleBaseLearner:
    """RFOED-NN's node classifier: a bagged ensemble of K small neural
    classifiers (MLP over the shared featurizer's output), each trained
    on its own bootstrap draw — the direct DL analog of RF's bagging.
    `rebalance=True` makes each member's draw a stratified balanced
    bootstrap instead of a plain one."""

    def __init__(
        self,
        n_members: int = 25,
        rebalance: bool = True,
        hidden_layer_sizes: tuple[int, ...] = (64,),
        max_iter: int = 200,
        random_state: int | None = None,
        max_bootstrap_per_class: int = 2000,
    ):
        self.n_members = n_members
        self.rebalance = rebalance
        self.hidden_layer_sizes = hidden_layer_sizes
        self.max_iter = max_iter
        self.random_state = random_state
        # Cap on each class's balanced-bootstrap draw, shared with
        # TorchNNEnsembleBaseLearner and TextCNNEnsembleBaseLearner so all
        # three neural base learners rebalance to the same budget by default
        # (paper Eq. 5, n_cap). Previously uncapped here, which let a shallow
        # cascade node with a huge "rest" class draw tens of thousands of
        # rows per member. Only the balanced draw is capped; the plain
        # (rebalance=False) bootstrap stays standard size-n bagging.
        self.max_bootstrap_per_class = max_bootstrap_per_class
        self._members: list = []

    def fit(self, X, y: np.ndarray) -> "NNEnsembleBaseLearner":
        rng = np.random.default_rng(self.random_state)
        self._members = []
        for k in range(self.n_members):
            if self.rebalance:
                idx = balanced_bootstrap_indices(y, rng, max_per_class=self.max_bootstrap_per_class)
            else:
                idx = _plain_bootstrap_indices(y, rng)
            X_k, y_k = X[idx], y[idx]
            if len(np.unique(y_k)) < 2:
                self._members.append(_SingleClassStub(int(y_k[0])))
                continue
            member = MLPClassifier(
                hidden_layer_sizes=self.hidden_layer_sizes,
                max_iter=self.max_iter,
                random_state=None if self.random_state is None else self.random_state + k,
                early_stopping=True,
            )
            member.fit(X_k, y_k)
            self._members.append(member)
        return self

    def predict_proba(self, X) -> np.ndarray:
        assert self._members, "call fit() first"
        probas = []
        for member in self._members:
            proba = member.predict_proba(X)
            classes = getattr(member, "classes_", np.array([0, 1]))
            probas.append(_align_proba_columns(proba, classes))
        return np.mean(probas, axis=0)


def _align_proba_columns(proba: np.ndarray, classes: np.ndarray) -> np.ndarray:
    """sklearn only emits columns for classes actually seen during fit;
    realign to a fixed [P(class=0), P(class=1)] layout so ensemble
    members (and RF vs NN outputs) are always directly comparable/averageable.

    Binary targets must be encoded as {0, 1} -- both RFOEDClassifier and
    FOCCClassifier always pass that (``(y == c).astype(int)`` / an indicator
    column), and the fixed 2-column layout depends on it. A custom base
    learner handed other labels would silently write to the wrong column, so
    that is rejected here instead."""
    out = np.zeros((proba.shape[0], 2))
    for col, c in enumerate(classes):
        ci = int(c)
        if ci not in (0, 1):
            raise ValueError(
                f"_align_proba_columns expects binary {{0, 1}} targets, got class label {c!r}."
            )
        out[:, ci] = proba[:, col]
    return out
