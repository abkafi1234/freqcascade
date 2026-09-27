"""WOS46985 node-level error decomposition (paper: Mechanistic Failure
Diagnosis, Table tab:diagnosis). Classifies every test error from an RFOED
cascade as:

  own-node-miss  the document's own correct node failed to fire for it
  early-capture  an earlier, wrong node fired a false positive and removed
                 the document before it reached its own node

and reports, per base learner, the dominant error type, own-class test
recall, per-node false-positive rate, and how FP captures concentrate by
cascade depth -- the RF-under-confident / NN-over-confident contrast.

    python scripts/diagnose_wos.py --bases rf     # CPU
    python scripts/diagnose_wos.py --bases nn     # GPU, capped and uncapped

Archiving (added 2026-09-17). This script used to print its report and write
nothing, and it computed only three of the seven node-level quantities in the
paper's diagnosis table; the other four (own-class test and train recall, and
the per-node false-positive rate) had no code path in the repository. It also
ran with the base learners' default rebalancing cap of 2000, although both this
docstring and the paper's caption described the cascade as uncapped. It now
computes every quantity under the explicit definitions in `node_rates`, runs
the neural cascade both capped and uncapped (the cap does not reach the Random
Forest base learner, which balances through class weights), and appends one
row per configuration to results/wos46985_diagnosis.jsonl, resumable per
(base, cap). The paper's table should be generated from that file.

The per-node trace re-walks `clf.nodes_` directly (there's no public
decision-path API yet). It uses an 80/20 fixed diagnostic split, matching
the paper; the remediation sweep (cap + threshold) is driven from
run_single_label_benchmark.py via RFOEDClassifier's cascade_cap / threshold
/ prior_correct arguments.
"""

from __future__ import annotations

import json
import time

import numpy as np

from _shared import N_ESTIMATORS, NN_EPOCHS, NN_MEMBERS, RESULTS_DIR, embed, make_tfidf

from freqcascade.datasets import load


OUT = RESULTS_DIR / "wos46985_diagnosis.jsonl"
# (base, cap): cap=2000 is the base learners' default and what this script ran
# before it was archived; None is the uncapped draw the paper's caption names.
CONFIGS = [("rf", 2000), ("nn", 2000), ("nn", None)]


def build(base: str, seed: int = 0, cap: int | None = 2000):
    from freqcascade.base_learners import RFBaseLearner
    from freqcascade.decomposition import RFOEDClassifier
    from freqcascade.torch_ensemble import TorchNNEnsembleBaseLearner

    if base == "rf":
        node = lambda i: RFBaseLearner(n_estimators=N_ESTIMATORS, rebalance=True,
                                       random_state=seed * 1000 + i, n_jobs=4)
    else:
        node = lambda i: TorchNNEnsembleBaseLearner(n_members=NN_MEMBERS, rebalance=True,
                                                    hidden_size=128, max_epochs=NN_EPOCHS,
                                                    random_state=seed * 1000 + i,
                                                    max_bootstrap_per_class=cap)
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


def node_rates(clf, X_test, y_test) -> dict:
    """Standalone node rates, each node evaluated on *every* test document
    rather than only those still active when the cascade reaches it:

      own_class_test_recall_mean   mean over nodes k of Pr[h_k fires | y = c_k]
                                   (one minus the own-node miss rate alpha_k)
      own_class_train_recall_mean  mean over nodes of the recall on its own class
                                   of the node's own training set, as recorded in
                                   RFOEDClassifier.node_diagnostics_
      fpr_other_mean               mean over nodes of Pr[h_k fires | y != c_k]
      fpr_later_mean               mean over nodes of Pr[h_k fires | y later than
                                   c_k in the peel order] (the beta_{k,j} of
                                   Proposition 1, pooled over later classes)
    """
    order = list(clf.class_order_)
    rank = {c: i for i, c in enumerate(order)}
    true_rank = np.array([rank[c] for c in y_test])
    rec, fpr_o, fpr_l = [], [], []
    for i, node in enumerate(clf.nodes_):
        p = clf._correct_p_positive(i, node.classifier.predict_proba(X_test)[:, 1])
        fire = p >= clf.threshold
        own, later = true_rank == i, true_rank > i
        if own.any():
            rec.append(float(fire[own].mean()))
        if (~own).any():
            fpr_o.append(float(fire[~own].mean()))
        if later.any():
            fpr_l.append(float(fire[later].mean()))
    train = [d["train_positive_recall"] for d in clf.node_diagnostics_
             if d.get("train_positive_recall") == d.get("train_positive_recall")]
    return {
        "own_class_test_recall_mean": float(np.mean(rec)),
        "own_class_train_recall_mean": float(np.mean(train)) if train else None,
        "fpr_other_mean": float(np.mean(fpr_o)),
        "fpr_later_mean": float(np.mean(fpr_l)),
        "n_nodes": len(clf.nodes_),
    }


def done() -> set:
    if not OUT.exists():
        return set()
    return {(r["base"], r["cap"]) for r in
            (json.loads(x) for x in OUT.read_text(encoding="utf-8").splitlines() if x.strip())}


def main() -> None:
    import argparse

    from freqcascade.metrics import evaluate

    ap = argparse.ArgumentParser()
    ap.add_argument("--bases", default="rf,nn",
                    help="rf runs on CPU and needs memory; nn runs on the GPU")
    bases = set(ap.parse_args().bases.split(","))

    ds = load("wos46985")
    rng = np.random.default_rng(0)
    idx = rng.permutation(ds.n_samples)
    cut = int(0.8 * ds.n_samples)
    tr, te = idx[:cut], idx[cut:]
    seen = done()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    for base, cap in CONFIGS:
        if base not in bases or (base, cap) in seen:
            continue
        X = make_tfidf(ds.texts[tr], ds.texts) if base == "rf" else embed(ds.texts, "minilm", ds.name)
        t0 = time.perf_counter()
        clf = build(base, cap=cap)
        clf.fit(X[tr], ds.target[tr])
        report = trace_errors(clf, X[te], ds.target[te])
        report.update(node_rates(clf, X[te], ds.target[te]))
        report["macro_f1"] = evaluate(ds.target[te], clf.predict(X[te]), ds.target[tr])["macro_f1"]
        row = dict(report, dataset="wos46985", base=base, cap=cap,
                   split="permutation 80/20, numpy default_rng(0)", seed=0,
                   n_train=int(len(tr)), n_test=int(len(te)),
                   secs=round(time.perf_counter() - t0, 1))
        with OUT.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row) + "\n")
        print(f"\nRFOED-{base.upper()} cap={cap}  {report}", flush=True)


if __name__ == "__main__":
    main()
