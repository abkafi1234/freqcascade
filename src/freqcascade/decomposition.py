"""The core RFOED mechanism: recursively peel the current majority class
off vs. "the rest," ordered by frequency, recurse on the remainder. Each
node's classifier is only trained on samples not yet resolved by an
earlier node (mirroring SBC's "removed from contention" semantics), and
inference cascades each sample through nodes in the same order until one
fires positive.

Ordering is fixed once from the *initial* training label distribution
(not recomputed from the shrinking remainder), matching Vasudevan et
al. (2024): "classes are first sorted in decreasing order of
class-frequency." `order="random"` reproduces the ND-random ablation
cell for comparison.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Literal

import numpy as np


@dataclass
class _Node:
    class_label: object
    classifier: object


@dataclass
class RFOEDClassifier:
    base_learner_factory: Callable[[int], object]
    order: Literal["frequency", "random"] = "frequency"
    random_state: int | None = None
    decision: Literal["cascade", "argmax_proba"] = "cascade"
    # T2.0 remediation #2 (capped cascade depth): if set, only the top
    # `cascade_cap` classes (by `order`) get their own binary peel node;
    # every remaining class is handled by a single flat multiclass "tail"
    # classifier fit once on whatever's still active after the capped
    # nodes. None (default) reproduces the original uncapped behavior
    # exactly — the last class in `class_order_` is the implicit
    # residual bucket, same as before this option existed.
    cascade_cap: int | None = None
    tail_learner_factory: Callable[[], object] | None = None
    # T2.0 remediation (evidence-driven, from the WOS46985 per-node
    # diagnostic — scripts/diagnose_wos.py): a fixed decision threshold
    # override (default 0.5) applied to each node's P(positive) — RF's
    # nodes turned out systematically *under*-confident on WOS46985
    # (own-class test recall averaging 0.30 despite perfect *training*
    # recall — a generalization/overfitting gap, not a calibration
    # shift), so a *lower* threshold empirically recovers some of that;
    # `threshold` is the blunt, model-agnostic knob for that.
    threshold: float = 0.5
    # `prior_correct=True` instead applies the principled fix for the
    # *opposite*, over-confident failure mode found in RFOED-NN: each
    # node is trained on a `balanced_bootstrap_indices` draw that is
    # always exactly 50/50 by construction (both classes drawn to the
    # same target_n — see rebalance.py), but predictions are thresholded
    # against the raw output as if it reflected the true node-local
    # prior, which is usually << 0.5 (often <1%, e.g. one class vs.
    # "everyone else still active"). The diagnostic found 94% of
    # RFOED-NN's test errors were "early_capture" — an earlier node
    # firing a false positive and stealing a document before it ever
    # reached its own correct node — with cumulative false-positive
    # captures 75% concentrated in just the first 25% of cascade depth,
    # exactly what training-prior/deployment-prior mismatch predicts
    # (Elkan 2001, "The Foundations of Cost-Sensitive Learning"): a
    # classifier trained under prior p_train=0.5 systematically
    # over-predicts the "balanced-training" class once deployed against
    # a much more skewed true prior. The correction re-maps the raw
    # output back to a probability calibrated to the *true* node-local
    # prior (n_positive/n_active from that node's own training data,
    # recorded in node_diagnostics_) before thresholding:
    #   p_true = p_bal*prior_true / (p_bal*prior_true + (1-p_bal)*(1-prior_true))
    # (the general Elkan formula simplifies to this because p_train=0.5
    # cancels out algebraically). Mutually exclusive in practice with a
    # custom `threshold` — combining both is allowed but not the
    # intended use.
    prior_correct: bool = False

    class_order_: list = field(default_factory=list, init=False)
    nodes_: list[_Node] = field(default_factory=list, init=False)
    node_diagnostics_: list[dict] = field(default_factory=list, init=False)
    tail_classifier_: object = field(default=None, init=False)
    label_dtype_: np.dtype = field(default=None, init=False)

    def fit(self, X, y: np.ndarray) -> "RFOEDClassifier":
        y = np.asarray(y)
        self.label_dtype_ = y.dtype
        rng = np.random.default_rng(self.random_state)

        classes, counts = np.unique(y, return_counts=True)
        if self.order == "frequency":
            self.class_order_ = list(classes[np.argsort(-counts)])
        elif self.order == "random":
            order = list(classes)
            rng.shuffle(order)
            self.class_order_ = order
        else:
            raise ValueError(f"unknown order: {self.order}")

        n_peel = len(self.class_order_) - 1
        if self.cascade_cap is not None:
            if self.tail_learner_factory is None:
                raise ValueError("cascade_cap requires tail_learner_factory")
            n_peel = min(n_peel, self.cascade_cap)

        self.nodes_ = []
        self.node_diagnostics_ = []
        self.tail_classifier_ = None
        active_mask = np.ones(len(y), dtype=bool)
        for i, current_class in enumerate(self.class_order_[:n_peel]):
            X_active = X[active_mask]
            y_active = y[active_mask]
            y_binary = (y_active == current_class).astype(int)

            learner = self.base_learner_factory(i)
            learner.fit(X_active, y_binary)
            self.nodes_.append(_Node(class_label=current_class, classifier=learner))

            # Diagnostics: how well each node separates its class from the
            # remainder *on its own training data* — lets us see whether
            # cascade errors are concentrated at particular depths (e.g.
            # the tail, where "rest" shrinks toward a handful of classes)
            # rather than spread uniformly, per the follow-up after the
            # first full-scale run (RFOED-NN's near-zero G-mean).
            train_pred = (learner.predict_proba(X_active)[:, 1] >= 0.5).astype(int)
            # Raw accuracy is misleading here: node targets get more
            # skewed with depth (e.g. 100 positive / 8725 active near the
            # root), so "always predict negative" already scores ~99% —
            # that's what made an earlier (K=50, undertrained — see
            # torch_ensemble.py fix) run look fine on this diagnostic
            # while actually collapsing on real test macro-F1/G-mean.
            # Balanced accuracy (mean of the two per-class recalls) does
            # not have that blind spot.
            pos_mask = y_binary == 1
            recall_pos = float(train_pred[pos_mask].mean()) if pos_mask.any() else float("nan")
            recall_neg = float((1 - train_pred[~pos_mask]).mean()) if (~pos_mask).any() else float("nan")
            self.node_diagnostics_.append(
                {
                    "depth": i,
                    "class_label": current_class,
                    "n_active": int(active_mask.sum()),
                    "n_positive": int(y_binary.sum()),
                    "train_binary_accuracy": float(np.mean(train_pred == y_binary)),
                    "train_balanced_accuracy": float(np.nanmean([recall_pos, recall_neg])),
                    "train_positive_recall": recall_pos,
                }
            )

            resolved = np.zeros(len(y), dtype=bool)
            resolved[active_mask] = y_active == current_class
            active_mask = active_mask & ~resolved

        # Tail: only built when cascade_cap actually truncated the peel
        # (n_peel < len(class_order_) - 1) — i.e. there's more than just
        # the single implicit-residual class left over.
        if self.cascade_cap is not None and n_peel < len(self.class_order_) - 1:
            X_tail = X[active_mask]
            y_tail = y[active_mask]
            tail = self.tail_learner_factory()
            tail.fit(X_tail, y_tail)
            self.tail_classifier_ = tail

        return self

    def _correct_p_positive(self, node_idx: int, p_positive: np.ndarray) -> np.ndarray:
        """Elkan (2001) training-prior -> true-prior recalibration for
        node `node_idx`'s raw P(positive), when `prior_correct=True` (see
        the dataclass field docstring for the derivation/motivation). A
        no-op when `prior_correct=False` (the default)."""
        if not self.prior_correct:
            return p_positive
        diag = self.node_diagnostics_[node_idx]
        prior_true = diag["n_positive"] / diag["n_active"]
        prior_true = min(max(prior_true, 1e-6), 1 - 1e-6)
        num = p_positive * prior_true
        denom = num + (1 - p_positive) * (1 - prior_true)
        denom = np.where(denom <= 0, 1e-12, denom)
        return num / denom

    def predict(self, X) -> np.ndarray:
        if self.decision == "argmax_proba":
            proba = self.predict_proba(X)
            best = np.argmax(proba, axis=1)
            order = np.asarray(self.class_order_, dtype=object)
            return order[best].astype(self.label_dtype_)

        n = X.shape[0]
        predicted = np.empty(n, dtype=object)
        active_mask = np.ones(n, dtype=bool)

        for i, node in enumerate(self.nodes_):
            if not active_mask.any():
                break
            idx = np.flatnonzero(active_mask)
            proba = node.classifier.predict_proba(X[idx])
            p_positive = self._correct_p_positive(i, proba[:, 1])
            positive = p_positive >= self.threshold
            resolved_idx = idx[positive]
            predicted[resolved_idx] = node.class_label
            active_mask[resolved_idx] = False

        if active_mask.any():
            idx = np.flatnonzero(active_mask)
            if self.tail_classifier_ is not None:
                tail_proba = self.tail_classifier_.predict_proba(X[idx])
                best = np.argmax(tail_proba, axis=1)
                tail_classes = np.asarray(self.tail_classifier_.classes_)
                predicted[idx] = tail_classes[best]
            else:
                predicted[idx] = self.class_order_[-1]
        # Predictions are built up in an object array (nodes fire in
        # arbitrary order, one label at a time), but the label type itself
        # is homogeneous -- cast back to the dtype `y` was fit with so
        # downstream sklearn metrics (`type_of_target`) see a proper
        # 'multiclass' array instead of 'unknown' for plain int/str labels.
        return predicted.astype(self.label_dtype_)

    def predict_proba(self, X) -> np.ndarray:
        """Path probability per class, as in nested-dichotomy ensembles:
        P(class_i) = P(node_i fires) * prod_{j<i} P(node_j does not fire).
        When cascade_cap truncated the peel, the remaining_mass left over
        after the capped nodes is distributed across the tail classes in
        proportion to the tail classifier's own predict_proba, rather
        than dumped entirely onto the single last class in class_order_
        (which is what happens, correctly, when there's no cap). Each
        node's p_positive is passed through `_correct_p_positive` first,
        so argmax_proba decisions and path-probabilities are consistent
        with the same prior-correction used by the cascade decision
        rule (`threshold`/`prior_correct` apply identically to both)."""
        n = X.shape[0]
        proba = np.zeros((n, len(self.class_order_)))
        remaining_mass = np.ones(n)

        for i, node in enumerate(self.nodes_):
            node_proba = node.classifier.predict_proba(X)
            p_positive = self._correct_p_positive(i, node_proba[:, 1])
            proba[:, i] = remaining_mass * p_positive
            remaining_mass = remaining_mass * (1 - p_positive)

        if self.tail_classifier_ is not None:
            tail_proba = self.tail_classifier_.predict_proba(X)
            tail_classes = np.asarray(self.tail_classifier_.classes_)
            class_to_col = {c: j for j, c in enumerate(self.class_order_)}
            for tcol, c in enumerate(tail_classes):
                proba[:, class_to_col[c]] = remaining_mass * tail_proba[:, tcol]
        else:
            proba[:, -1] = remaining_mass
        return proba

    @property
    def classes_(self) -> list:
        return self.class_order_


@dataclass
class HierarchicalRFOEDClassifier:
    """Two-level RFOED (T2.0 WOS46985 remediation attempt #1): first peel
    by a coarse *group* label (frequency-ordered RFOEDClassifier over
    e.g. WOS46985's 7 YL1 parent domains), then, independently within
    each group, fit a second frequency-ordered RFOEDClassifier over that
    group's own leaf classes only.

    Motivation (see scripts/diagnose_wos.py / results/TIER1_RESULTS.md
    "WOS46985 investigation"): a single flat 133-node cascade forces
    every document to survive up to 133 sequential binary decisions
    before reaching its own class's node, and the per-node diagnostic
    showed the per-node error, while individually modest, compounds
    across that many nodes. Splitting into a short (<=6-node) group
    stage plus up to 7 independent per-group cascades (each only as deep
    as that group's own leaf-class count, e.g. 8-54 nodes for WOS46985's
    YL1 domains, not 133) bounds the worst-case chain length any single
    document has to survive, without changing the base learner or the
    rebalancing mechanism at all — isolating "does shortening the chain
    help" as its own factor.

    Node random-state/seed namespacing: the group-level cascade uses
    small indices (0, 1, 2, ...) from `base_learner_factory`. Each
    per-group sub-cascade gets its own disjoint block of indices
    (`(group_rank + 1) * 1000 + local_index`) so no two nodes anywhere
    in the whole hierarchical model — group stage or any sub-cascade —
    ever call the factory with the same index and collide on
    `random_state=seed*1000+i`-style seeding used by the base learners.
    """

    base_learner_factory: Callable[[int], object]
    order: Literal["frequency", "random"] = "frequency"
    random_state: int | None = None
    decision: Literal["cascade", "argmax_proba"] = "cascade"
    # Passed straight through to both the group-level cascade and every
    # per-group sub-cascade — see RFOEDClassifier's fields for the
    # motivation (T2.0 WOS46985 remediation: RF nodes were found to be
    # under-confident, NN nodes over-confident relative to their
    # node-local prior).
    threshold: float = 0.5
    prior_correct: bool = False

    group_classifier_: RFOEDClassifier = field(init=False, default=None)
    sub_classifiers_: dict = field(default_factory=dict, init=False)
    group_order_: list = field(default_factory=list, init=False)
    label_dtype_: np.dtype = field(default=None, init=False)

    _INDEX_BLOCK = 1000  # headroom per group; no WOS46985 domain has >55 leaf classes

    def fit(self, X, y, groups) -> "HierarchicalRFOEDClassifier":
        y = np.asarray(y)
        self.label_dtype_ = y.dtype
        groups = np.asarray(groups)

        self.group_classifier_ = RFOEDClassifier(
            base_learner_factory=self.base_learner_factory,
            order=self.order, decision=self.decision, random_state=self.random_state,
            threshold=self.threshold, prior_correct=self.prior_correct,
        )
        self.group_classifier_.fit(X, groups)
        self.group_order_ = list(self.group_classifier_.class_order_)

        self.sub_classifiers_ = {}
        for rank, g in enumerate(self.group_order_):
            mask = groups == g
            offset = (rank + 1) * self._INDEX_BLOCK
            sub = RFOEDClassifier(
                base_learner_factory=lambda i, _offset=offset: self.base_learner_factory(_offset + i),
                order=self.order, decision=self.decision, random_state=self.random_state,
                threshold=self.threshold, prior_correct=self.prior_correct,
            )
            sub.fit(X[mask], y[mask])
            self.sub_classifiers_[g] = sub

        return self

    def predict(self, X) -> np.ndarray:
        pred_groups = np.asarray(self.group_classifier_.predict(X), dtype=object)
        predicted = np.empty(X.shape[0], dtype=object)
        for g in np.unique(pred_groups):
            idx = np.flatnonzero(pred_groups == g)
            sub = self.sub_classifiers_.get(g)
            if sub is None:
                # Group classifier predicted a group value that had no
                # training rows (shouldn't happen — every group value it
                # can emit came from its own training labels — but fail
                # soft rather than crash a long run).
                predicted[idx] = self.group_order_[-1]
                continue
            predicted[idx] = sub.predict(X[idx])
        return predicted.astype(self.label_dtype_)

    def predict_proba(self, X):
        """Not implemented: path-probability composition across two
        independent cascades (group then leaf) isn't a simple product in
        the same way a single RFOEDClassifier's is, since leaf classes
        with the same name never appear in more than one sub-cascade's
        `class_order_` in the common case. Only `predict()` (cascade
        decision) is supported; callers needing argmax_proba should set
        `decision="argmax_proba"`, which applies at both the group and
        leaf stage internally."""
        raise NotImplementedError(
            "HierarchicalRFOEDClassifier.predict_proba is not supported; use predict()."
        )
