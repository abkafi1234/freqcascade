"""Multi-label baselines (EXPERIMENTAL_DESIGN.md §3b), the multi-label
counterpart of `rfoed/resampling_baselines.py`. Every class/factory here
exposes the same `fit(X, Y)` / `predict(X)` / `predict_proba(X)` interface
as `FOCCClassifier` (rfoed/focc.py) so all six multi-label methods (BR, CC,
ECC, Balanced-BR, FOCC-RF, FOCC-NN) are runnable through one shared harness.

Four baselines, deliberately built from as little new machinery as
possible:

- **Binary Relevance (BR)** / **Balanced Binary Relevance**: one
  independent binary classifier per label, no chaining, no ordering.
  `BinaryRelevanceClassifier(rebalance=False)` is BR; `rebalance=True` is
  Balanced-BR (isolates "does per-label rebalancing help" independent of
  chaining -- design §3b baseline 4). Both reuse RFBaseLearner /
  TorchNNEnsembleBaseLearner directly via the same link-factory helpers
  FOCC uses, rather than sklearn's MultiOutputClassifier, so BR's per-label
  unit is *exactly* FOCC's per-link unit with rebalancing as the only
  independent variable -- keeping "chaining" and "rebalancing" orthogonal
  factors across the whole baseline family, the same way
  rfoed/base_learners.py keeps "base learner type" and "rebalance" orthogonal
  for RFOED.

- **Classifier Chains (CC)** / **Ensemble of Classifier Chains (ECC)**:
  per the task instructions, these reuse `FOCCClassifier` directly
  (`order="random", rebalance=False`) rather than a separate
  implementation or `sklearn.multioutput.ClassifierChain` -- "FOCC with
  random order + no rebalance IS essentially plain classifier chains."
  ECC is `n_chains` independent random-order FOCCClassifier instances with
  predicted probabilities averaged (majority vote in expectation).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .focc import FOCCClassifier, nn_link_factory, rf_link_factory


@dataclass
class BinaryRelevanceClassifier:
    """BR (`rebalance=False`) / Balanced-BR (`rebalance=True`): independent
    per-label binary classifiers, no chain augmentation at all -- link i's
    classifier only ever sees the raw feature matrix X, never any other
    label's predictions or ground truth. `base_learner_factory` has the
    same `(rank, rebalance) -> BaseLearner` signature as
    FOCCClassifier.base_learner_factory (rf_link_factory/nn_link_factory
    are directly reusable here for exactly that reason)."""

    base_learner_factory: Callable[[int, bool], object]
    rebalance: bool = False
    random_state: int | None = None

    learners_: list = field(default_factory=list, init=False)
    n_labels_: int = field(default=0, init=False)

    def fit(self, X, Y: np.ndarray) -> "BinaryRelevanceClassifier":
        Y = np.asarray(Y)
        self.n_labels_ = Y.shape[1]
        self.learners_ = []
        for j in range(self.n_labels_):
            learner = self.base_learner_factory(j, self.rebalance)
            learner.fit(X, Y[:, j])
            self.learners_.append(learner)
        return self

    def predict_proba(self, X) -> np.ndarray:
        assert self.learners_, "call fit() first"
        proba = np.zeros((X.shape[0], self.n_labels_))
        for j, learner in enumerate(self.learners_):
            proba[:, j] = learner.predict_proba(X)[:, 1]
        return proba

    def predict(self, X) -> np.ndarray:
        return (self.predict_proba(X) >= 0.5).astype(np.int8)


@dataclass
class EnsembleClassifierChains:
    """ECC: `n_chains` independent FOCCClassifier chains, each with an
    independently shuffled random label order (`order="random"`), predicted
    probabilities averaged across chains -- the direct multi-label analog
    of RFOED's own bagging idea, one level up (ensembling whole chains
    rather than per-node members). `rebalance=False` by default to match
    the plain-CC baseline on every axis except "one chain vs. several"
    (isolating what ensembling chains adds, independent of rebalancing,
    which Balanced-BR already isolates on its own)."""

    base_learner_factory: Callable[[int, bool], object]
    n_chains: int = 5
    rebalance: bool = False
    random_state: int | None = None

    chains_: list[FOCCClassifier] = field(default_factory=list, init=False)
    n_labels_: int = field(default=0, init=False)

    def fit(self, X, Y: np.ndarray) -> "EnsembleClassifierChains":
        Y = np.asarray(Y)
        self.n_labels_ = Y.shape[1]
        self.chains_ = []
        for c in range(self.n_chains):
            seed = None if self.random_state is None else self.random_state * 100 + c
            chain = FOCCClassifier(
                base_learner_factory=self.base_learner_factory,
                order="random",
                rebalance=self.rebalance,
                random_state=seed,
            )
            chain.fit(X, Y)
            self.chains_.append(chain)
        return self

    def predict_proba(self, X) -> np.ndarray:
        assert self.chains_, "call fit() first"
        probas = [chain.predict_proba(X) for chain in self.chains_]
        return np.mean(probas, axis=0)

    def predict(self, X) -> np.ndarray:
        return (self.predict_proba(X) >= 0.5).astype(np.int8)


# ---------------------------------------------------------------------------
# Ready-made factories, RF and NN variants of each baseline -- mirrors
# focc.py's make_focc_rf/make_focc_nn so every method in the six-way
# comparison is constructed the same way (one function call, seed in,
# configured estimator out) for the shared T3.5 harness.
# ---------------------------------------------------------------------------


def make_br_rf(
    rebalance: bool = False, random_state: int | None = None, n_estimators: int = 200, n_jobs: int = 4
) -> BinaryRelevanceClassifier:
    return BinaryRelevanceClassifier(
        base_learner_factory=rf_link_factory(n_estimators=n_estimators, n_jobs=n_jobs, base_seed=random_state),
        rebalance=rebalance,
        random_state=random_state,
    )


def make_br_nn(
    rebalance: bool = False,
    random_state: int | None = None,
    n_members: int = 50,
    hidden_size: int = 128,
    max_epochs: int = 250,
    device: str | None = None,
) -> BinaryRelevanceClassifier:
    return BinaryRelevanceClassifier(
        base_learner_factory=nn_link_factory(
            n_members=n_members, hidden_size=hidden_size, max_epochs=max_epochs,
            device=device, base_seed=random_state,
        ),
        rebalance=rebalance,
        random_state=random_state,
    )


def make_balanced_br_rf(random_state: int | None = None, **rf_kwargs) -> BinaryRelevanceClassifier:
    """Balanced-BR (design §3b baseline 4): BR + per-label rebalancing, no
    chaining -- isolates rebalancing's effect independent of chain
    structure."""
    return make_br_rf(rebalance=True, random_state=random_state, **rf_kwargs)


def make_balanced_br_nn(random_state: int | None = None, **nn_kwargs) -> BinaryRelevanceClassifier:
    return make_br_nn(rebalance=True, random_state=random_state, **nn_kwargs)


def make_cc_rf(random_state: int | None = None, n_estimators: int = 200, n_jobs: int = 4) -> FOCCClassifier:
    """CC (design §3b baseline 2): FOCCClassifier(order="random",
    rebalance=False) -- see module docstring for why this reuses
    FOCCClassifier rather than a separate implementation."""
    return FOCCClassifier(
        base_learner_factory=rf_link_factory(n_estimators=n_estimators, n_jobs=n_jobs, base_seed=random_state),
        order="random",
        rebalance=False,
        random_state=random_state,
    )


def make_cc_nn(
    random_state: int | None = None,
    n_members: int = 50,
    hidden_size: int = 128,
    max_epochs: int = 250,
    device: str | None = None,
) -> FOCCClassifier:
    return FOCCClassifier(
        base_learner_factory=nn_link_factory(
            n_members=n_members, hidden_size=hidden_size, max_epochs=max_epochs,
            device=device, base_seed=random_state,
        ),
        order="random",
        rebalance=False,
        random_state=random_state,
    )


def make_ecc_rf(
    n_chains: int = 5, random_state: int | None = None, n_estimators: int = 200, n_jobs: int = 4
) -> EnsembleClassifierChains:
    """ECC (design §3b baseline 3): `n_chains` random-order chains, RF per
    link, predictions averaged."""
    return EnsembleClassifierChains(
        base_learner_factory=rf_link_factory(n_estimators=n_estimators, n_jobs=n_jobs, base_seed=random_state),
        n_chains=n_chains,
        rebalance=False,
        random_state=random_state,
    )


def make_ecc_nn(
    n_chains: int = 5,
    random_state: int | None = None,
    n_members: int = 50,
    hidden_size: int = 128,
    max_epochs: int = 250,
    device: str | None = None,
) -> EnsembleClassifierChains:
    return EnsembleClassifierChains(
        base_learner_factory=nn_link_factory(
            n_members=n_members, hidden_size=hidden_size, max_epochs=max_epochs,
            device=device, base_seed=random_state,
        ),
        n_chains=n_chains,
        rebalance=False,
        random_state=random_state,
    )
