"""Cross-machine reproduction check, run before resuming on a new machine.

Recomputes already-stored result cells in memory (nothing is written) and
compares them with the stored values, so a resumed campaign does not silently
mix numbers from machines that disagree.

  CPU cell  R5 weak-learner sweep: RUSBoost, stump, TF-IDF, CLINC150 K=5, seed 0.
            Pure scikit-learn / imbalanced-learn on CPU: must match exactly.
  GPU cell  R8 adaptive-cap sweep: RFOED-NN, fixed n_cap=2000, 20 Newsgroups fold 0.
            Batched neural ensemble on CUDA: may differ slightly across GPU models.

    python scripts/check_reproduction.py

Exit code 0 = CPU exact and GPU within tolerance; 1 otherwise. What to do on a
mismatch is in KBS/CLAUDE.md ("Reproduction check").
"""

from __future__ import annotations

import json
import sys

import numpy as np

from _shared import NN_EPOCHS, NN_MEMBERS, RESULTS_DIR, get_folds, make_tfidf, embed

from freqcascade.datasets import load
from freqcascade.metrics import evaluate

GPU_TOL = 0.002


def stored(fname, pred):
    for x in (RESULTS_DIR / fname).read_text(encoding="utf-8").splitlines():
        if x.strip():
            r = json.loads(x)
            if pred(r):
                return r
    raise LookupError("reference cell not found in %s" % fname)


def cpu_cell():
    from imblearn.ensemble import RUSBoostClassifier
    from sklearn.tree import DecisionTreeClassifier

    ref = stored("weaklearner_sweep.jsonl", lambda r: r["protocol"] == "class-count sweep" and
                 r["learner"] == "stump" and r["ensemble"] == "RUSBoost" and r["features"] == "tfidf" and
                 r["K"] == 5 and r["seed"] == 0 and "macro_f1" in r)["macro_f1"]
    ds = load("clinc150")
    y = np.asarray(ds.target)
    is_test = np.asarray(ds.split) == "test"
    rng = np.random.default_rng(0)
    sub = set(rng.choice(np.unique(y), size=5, replace=False).tolist())
    mask = np.isin(y, list(sub))
    tr, te = np.flatnonzero(mask & ~is_test), np.flatnonzero(mask & is_test)
    X = make_tfidf(ds.texts[tr], ds.texts)
    est = RUSBoostClassifier(estimator=DecisionTreeClassifier(max_depth=1, random_state=0),
                             n_estimators=30, random_state=0)
    est.fit(X[tr], y[tr])
    return ref, evaluate(y[te], est.predict(X[te]), y[tr])["macro_f1"]


def gpu_cell():
    from freqcascade.decomposition import RFOEDClassifier
    from freqcascade.torch_ensemble import TorchNNEnsembleBaseLearner

    ref = stored("ncap_adaptive.jsonl", lambda r: r["dataset"] == "20newsgroups_ir50" and
                 r["rule"] == "fixed" and r["fold"] == 0 and r.get("score") is not None)["score"]
    ds = load("20newsgroups")
    y = np.asarray(ds.target)
    X = embed(ds.texts, "minilm", ds.name)
    f = get_folds(ds)[0]
    fac = lambda node: TorchNNEnsembleBaseLearner(  # noqa: E731
        n_members=NN_MEMBERS, rebalance=True, hidden_size=128, max_epochs=NN_EPOCHS,
        random_state=0 * 1000 + node, max_bootstrap_per_class=2000, cap_rule="fixed")
    est = RFOEDClassifier(base_learner_factory=fac, order="frequency", random_state=0)
    est.fit(X[f.train_idx], y[f.train_idx])
    return ref, evaluate(y[f.test_idx], est.predict(X[f.test_idx]), y[f.train_idx])["macro_f1"]


def main():
    ok = True
    ref, got = cpu_cell()
    exact = abs(ref - got) < 1e-12
    ok &= exact
    print("CPU cell  stored %.10f  recomputed %.10f  -> %s" % (ref, got, "EXACT" if exact else "MISMATCH"))
    try:
        import torch
        dev = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no CUDA"
    except Exception:
        dev = "torch unavailable"
    ref, got = gpu_cell()
    d = abs(ref - got)
    within = d <= GPU_TOL
    ok &= within
    print("GPU cell  stored %.10f  recomputed %.10f  |diff| %.6f  (tolerance %.3f, device: %s)  -> %s"
          % (ref, got, d, GPU_TOL, dev, "EXACT" if d < 1e-12 else ("WITHIN TOLERANCE" if within else "MISMATCH")))
    print("\nREPRODUCTION:", "OK" if ok else "MISMATCH, read KBS/CLAUDE.md before resuming")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
