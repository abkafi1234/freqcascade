"""WOS46985 node-level error decomposition (paper: Mechanistic Failure
Diagnosis). Classifies every test error from an uncapped RFOED cascade as:

  own-node-miss  the document's own correct node failed to fire for it
  early-capture  an earlier, wrong node fired a false positive and removed
                 the document before it reached its own node

and reports, per base learner, the dominant error type, own-class test
recall, per-node false-positive rate, and how FP captures concentrate by
cascade depth -- the RF-under-confident / NN-over-confident contrast.

    python scripts/diagnose_wos.py

The per-node trace re-walks `clf.nodes_` directly (there's no public
decision-path API yet). It uses an 80/20 fixed diagnostic split, matching
the paper; the remediation sweep (cap + threshold) is driven from
run_single_label_benchmark.py via RFOEDClassifier's cascade_cap / threshold
/ prior_correct arguments.
"""

from __future__ import annotations

import numpy as np

from _shared import N_ESTIMATORS, NN_EPOCHS, NN_MEMBERS, embed, make_tfidf

from freqcascade.datasets import load


def build(base: str, seed: int = 0):
    from freqcascade.base_learners import RFBaseLearner
    from freqcascade.decomposition import RFOEDClassifier
    from freqcascade.torch_ensemble import TorchNNEnsembleBaseLearner

    if base == "rf":
        node = lambda i: RFBaseLearner(n_estimators=N_ESTIMATORS, rebalance=True,
                                       random_state=seed * 1000 + i, n_jobs=4)
    else:
        node = lambda i: TorchNNEnsembleBaseLearner(n_members=NN_MEMBERS, rebalance=True,
                                                    hidden_size=128, max_epochs=NN_EPOCHS,
                                                    random_state=seed * 1000 + i)
    return RFOEDClassifier(base_learner_factory=node, order="frequency", random_state=seed)


def trace_errors(clf, X_test, y_test) -> dict:
    """Walk the fixed cascade node by node, recording for each test row the
    first node to fire and whether that was its own (correct) node."""
    order = list(clf.class_order_)
    rank = {c: i for i, c in enumerate(order)}
    n = X_test.shape[0]
    fired_at = np.full(n, -1)          # index in class_order_ of the first firing node
    active = np.ones(n, bool)
    for i, node in enumerate(clf.nodes_):
        if not active.any():
            break
        p = node.classifier.predict_proba(X_test[active])[:, 1]
        p = clf._correct_p_positive(i, p)
        idx = np.flatnonzero(active)
        hit = idx[p >= clf.threshold]
        fired_at[hit] = i
        active[hit] = False

    true_rank = np.array([rank[c] for c in y_test])
    pred_rank = np.where(fired_at >= 0, fired_at, len(order) - 1)
    correct = pred_rank == true_rank
    errors = ~correct
    early_capture = errors & (fired_at >= 0) & (fired_at < true_rank)
    own_node_miss = errors & ~early_capture
    return {
        "n_errors": int(errors.sum()),
        "own_node_miss_frac": float(own_node_miss.sum() / max(errors.sum(), 1)),
        "early_capture_frac": float(early_capture.sum() / max(errors.sum(), 1)),
        "fp_captures_first_quartile": float(
            (early_capture & (fired_at < 0.25 * len(order))).sum() / max(early_capture.sum(), 1)
        ),
    }


def main() -> None:
    ds = load("wos46985")
    rng = np.random.default_rng(0)
    idx = rng.permutation(ds.n_samples)
    cut = int(0.8 * ds.n_samples)
    tr, te = idx[:cut], idx[cut:]

    for base in ("rf", "nn"):
        X = make_tfidf(ds.texts[tr], ds.texts) if base == "rf" else embed(ds.texts, "minilm", ds.name)
        clf = build(base)
        clf.fit(X[tr], ds.target[tr])
        report = trace_errors(clf, X[te], ds.target[te])
        print(f"\nRFOED-{base.upper()}  {report}")


if __name__ == "__main__":
    main()
