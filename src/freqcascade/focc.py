"""FOCC: Frequency-Ordered Classifier Chains -- the multi-label sibling of
RFOEDClassifier (rfoed/decomposition.py), for Reuters-21578 (design §3b).

Same underlying mechanism as RFOED -- order by frequency, use an ensemble
(RF or bagged-NN) at each node, rebalance each node's training draw -- but
restructured as a *chain* (Read et al. 2011 classifier chains) rather than a
*peel*. RFOED's peel assumes mutual exclusivity: once a node fires for a
document, that document is removed from contention for every later node.
That assumption is exactly wrong for genuinely multi-label data (a Reuters
document can carry up to 16 topics at once, rfoed/data.py:load_reuters21578)
-- peeling would silently drop every label after a document's first match.
FOCC instead never removes a document: every one of the n_labels binary
"member" classifiers sees every document, and each link's classifier is
given the *augmented* feature matrix `[X | already-processed labels]`
(standard classifier-chain feature augmentation) so later, rarer labels can
condition on earlier, more frequent ones.

Ordering: `order="frequency"` sorts labels by descending training-set
frequency (mirroring RFOEDClassifier's default and the paper's core
frequency-ordering claim, RQ1/H1, transplanted to the multi-label setting
per RQ6/H4); `order="random"` reproduces plain classifier chains for the
mini-ablation (design §3b) and, combined with `rebalance=False`, IS the
"Classifier Chains" baseline itself -- see rfoed/multilabel_baselines.py,
which reuses this class rather than reimplementing chaining.

Base learners are NOT reimplemented here: `base_learner_factory` plugs in
`RFBaseLearner` (rfoed/base_learners.py) or `TorchNNEnsembleBaseLearner`
(rfoed/torch_ensemble.py), the exact same ensemble-per-node +
per-node-rebalancing units RFOED uses. Unlike RFOEDClassifier's factory
(`Callable[[int], object]`, since rebalance is baked into whatever the
caller's lambda constructs), FOCCClassifier's factory takes an *explicit*
`rebalance` argument at call time (`Callable[[int, bool], object]`) --
needed so a single factory can be reused unchanged across FOCC-RF/NN
(rebalance=True) and the CC/ECC baselines (rebalance=False) just by
flipping `FOCCClassifier.rebalance`, per the design's explicit instruction
to reuse this class for the plain-chain baselines rather than duplicate the
chaining logic (TIER3_PLAN.md T3.4 / EXPERIMENTAL_DESIGN.md §3b).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Literal

import numpy as np
import scipy.sparse as sp

from .base_learners import RFBaseLearner
from .torch_ensemble import TorchNNEnsembleBaseLearner


def _augment(X, extra_cols: np.ndarray):
    """Hstack X with extra binary columns (the chain's feature
    augmentation), preserving sparse-vs-dense to match X's own type. RF's
    base learner traditionally gets sparse TF-IDF (RandomForestClassifier
    handles sparse input natively); NN's gets dense embeddings/SVD features
    (rfoed/features.py). `extra_cols` is always a small dense array (at
    most n_labels-1 columns), so wrapping it in a sparse matrix just for the
    hstack is cheap relative to X itself."""
    if extra_cols.shape[1] == 0:
        return X
    if sp.issparse(X):
        return sp.hstack([X, sp.csr_matrix(extra_cols, dtype=float)], format="csr")
    return np.hstack([np.asarray(X), extra_cols])


@dataclass
class _Link:
    label_index: int
    classifier: object


@dataclass
class FOCCClassifier:
    """One classifier chain over a multi-label indicator matrix.

    `base_learner_factory(rank, rebalance) -> BaseLearner`: called once per
    link, in chain order (`rank` is the link's position, 0-indexed, NOT the
    label's column index -- use it the same way RFOEDClassifier's factory
    uses node depth, e.g. to namespace `random_state`). `rebalance` is
    `self.rebalance`, threaded through so the same factory works whether
    this instance is configured as FOCC-RF/NN (rebalance=True) or a plain
    chain baseline (rebalance=False) -- see rf_link_factory/nn_link_factory
    below and rfoed/multilabel_baselines.py.

    `fit(X, Y)`: Y is an (n_docs, n_labels) binary indicator matrix
    (rfoed/data.py's MultiLabelSplit.Y). Link i's classifier is trained on
    `[X | Y[:, already-processed label columns]]` -- the *true* labels of
    earlier-in-chain labels, standard classifier-chain training (Read et
    al. 2011): using ground truth during fit avoids compounding a nascent
    classifier's own errors into training data for every later link.

    `predict(X)` / `predict_proba(X)`: walk the same chain at inference
    time, but (necessarily, since true labels aren't available) augment
    with each earlier link's own *predicted* (thresholded) labels instead.
    """

    base_learner_factory: Callable[[int, bool], object]
    order: Literal["frequency", "random"] = "frequency"
    rebalance: bool = True
    random_state: int | None = None

    label_order_: list[int] = field(default_factory=list, init=False)
    links_: list[_Link] = field(default_factory=list, init=False)
    n_labels_: int = field(default=0, init=False)

    def fit(self, X, Y: np.ndarray) -> "FOCCClassifier":
        Y = np.asarray(Y)
        n_docs, n_labels = Y.shape
        self.n_labels_ = n_labels
        rng = np.random.default_rng(self.random_state)

        label_counts = Y.sum(axis=0)
        if self.order == "frequency":
            # Descending frequency; ties broken by column index so ordering
            # is deterministic (mirrors RFOEDClassifier's fixed-once-from-
            # initial-distribution ordering).
            self.label_order_ = list(np.lexsort((np.arange(n_labels), -label_counts)))
        elif self.order == "random":
            order = list(range(n_labels))
            rng.shuffle(order)
            self.label_order_ = order
        else:
            raise ValueError(f"unknown order: {self.order}")

        self.links_ = []
        processed_cols: list[int] = []
        for rank, label_idx in enumerate(self.label_order_):
            if processed_cols:
                extra = Y[:, processed_cols]
            else:
                extra = np.zeros((n_docs, 0), dtype=Y.dtype)
            X_aug = _augment(X, extra)
            y_bin = Y[:, label_idx]

            learner = self.base_learner_factory(rank, self.rebalance)
            learner.fit(X_aug, y_bin)
            self.links_.append(_Link(label_index=label_idx, classifier=learner))
            processed_cols.append(label_idx)

        return self

    def predict_proba(self, X) -> np.ndarray:
        """P(label=1) per label, in original column order (not chain
        order). Each link's own hard (>=0.5) prediction is what gets fed
        forward as that label's augmentation column for later links --
        ground truth isn't available at inference time, so the chain must
        condition on its own predictions here, unlike during fit()."""
        assert self.links_, "call fit() first"
        n_docs = X.shape[0]
        proba = np.zeros((n_docs, self.n_labels_))
        pred_cols = np.zeros((n_docs, 0), dtype=np.int8)

        for link in self.links_:
            X_aug = _augment(X, pred_cols)
            p = link.classifier.predict_proba(X_aug)[:, 1]
            proba[:, link.label_index] = p
            hard = (p >= 0.5).astype(np.int8).reshape(-1, 1)
            pred_cols = np.hstack([pred_cols, hard])

        return proba

    def predict(self, X) -> np.ndarray:
        """Binary (n_docs, n_labels) indicator matrix."""
        return (self.predict_proba(X) >= 0.5).astype(np.int8)

    @property
    def classes_(self) -> list[int]:
        return list(range(self.n_labels_))


# ---------------------------------------------------------------------------
# Link-factory helpers: reused verbatim by rfoed/multilabel_baselines.py so
# every method in the multi-label track (FOCC-RF/NN, BR, Balanced-BR, CC,
# ECC) draws its per-link base learner from the exact same construction
# logic -- only `order`/`rebalance` (FOCCClassifier-level) differ between
# methods, never the base-learner plumbing itself.
# ---------------------------------------------------------------------------


def rf_link_factory(
    n_estimators: int = 200, n_jobs: int = 4, base_seed: int | None = None
) -> Callable[[int, bool], RFBaseLearner]:
    """Per-link RF factory. `n_jobs` bounded per the project-wide RAM/swap
    lesson (see base_learners.RFBaseLearner) -- FOCC fits one RF per label
    (up to 90 for Reuters), same "don't spawn a full process pool per node"
    concern as RFOED's cascade."""

    def factory(rank: int, rebalance: bool) -> RFBaseLearner:
        seed = None if base_seed is None else base_seed * 1000 + rank
        return RFBaseLearner(
            n_estimators=n_estimators, rebalance=rebalance, random_state=seed, n_jobs=n_jobs
        )

    return factory


def nn_link_factory(
    n_members: int = 50,
    hidden_size: int = 128,
    max_epochs: int = 250,
    device: str | None = None,
    base_seed: int | None = None,
) -> Callable[[int, bool], TorchNNEnsembleBaseLearner]:
    """Per-link NN-ensemble factory via TorchNNEnsembleBaseLearner. Defaults
    (n_members=50, hidden_size=128, max_epochs=250) match RFOED-NN's current
    production config (scripts/run_full_benchmark.py) -- i.e. §3a candidate
    unit 1 (frozen sentence-embedding + MLP), which is what's actually in
    use today, NOT yet the §3a pilot's eventual winner: that pilot (T2.3) is
    a separate, still-in-progress task. Swap this factory's body (or just
    the caller's kwargs) once §3a picks a winning unit -- FOCC-NN is defined
    to track "whatever RFOED-NN's node classifier is," not a hardcoded
    architecture, so no other code here needs to change when that happens.
    """

    def factory(rank: int, rebalance: bool) -> TorchNNEnsembleBaseLearner:
        seed = None if base_seed is None else base_seed * 1000 + rank
        return TorchNNEnsembleBaseLearner(
            n_members=n_members,
            rebalance=rebalance,
            hidden_size=hidden_size,
            max_epochs=max_epochs,
            device=device,
            random_state=seed,
        )

    return factory


def make_focc_rf(
    order: Literal["frequency", "random"] = "frequency",
    rebalance: bool = True,
    random_state: int | None = None,
    n_estimators: int = 200,
    n_jobs: int = 4,
) -> FOCCClassifier:
    """FOCC-RF: frequency-ordered chain, RF per link, per-link rebalancing
    (design §3b, method 5 -- the RF-based proposed method)."""
    return FOCCClassifier(
        base_learner_factory=rf_link_factory(n_estimators=n_estimators, n_jobs=n_jobs, base_seed=random_state),
        order=order,
        rebalance=rebalance,
        random_state=random_state,
    )


def make_focc_nn(
    order: Literal["frequency", "random"] = "frequency",
    rebalance: bool = True,
    random_state: int | None = None,
    n_members: int = 50,
    hidden_size: int = 128,
    max_epochs: int = 250,
    device: str | None = None,
) -> FOCCClassifier:
    """FOCC-NN: frequency-ordered chain, bagged-NN-ensemble per link,
    per-link rebalancing -- design §3b, method 6, the **main multi-label
    proposal**. See nn_link_factory's docstring re: the §3a base-learner
    unit pilot dependency."""
    return FOCCClassifier(
        base_learner_factory=nn_link_factory(
            n_members=n_members, hidden_size=hidden_size, max_epochs=max_epochs,
            device=device, base_seed=random_state,
        ),
        order=order,
        rebalance=rebalance,
        random_state=random_state,
    )
