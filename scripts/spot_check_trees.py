"""100 vs 200 trees on CLINC150 -- is the config's ~0.005 macro-F1 shift real
and uniform? Writes results/spot_trees.jsonl."""
from __future__ import annotations

import json
import time

import numpy as np

from _shared import RESULTS_DIR, RF_NJOBS, get_folds, make_tfidf

from freqcascade.datasets import load
from freqcascade.metrics import evaluate

ds = load("clinc150")
tr = np.flatnonzero(ds.split != "test")
te = np.flatnonzero(ds.split == "test")
X = make_tfidf(ds.texts[tr], ds.texts)
out = []
for n_est in (100, 200):
    from freqcascade.base_learners import RFBaseLearner
    from freqcascade.decomposition import RFOEDClassifier
    from freqcascade.resampling_baselines import ResamplingBaseline

    for name, mk in [
        ("Flat-RF+SMOTE", lambda s: ResamplingBaseline("smote", n_estimators=n_est, n_jobs=RF_NJOBS,
                                                       random_state=s, oversample_cap_mult=4.0)),
        ("RFOED-RF", lambda s: RFOEDClassifier(
            base_learner_factory=lambda i, s=s: RFBaseLearner(n_estimators=n_est, rebalance=True,
                                                             random_state=s * 1000 + i, n_jobs=RF_NJOBS),
            order="frequency", random_state=s)),
    ]:
        for s in range(5):
            est = mk(s); t0 = time.perf_counter()
            est.fit(X[tr], ds.target[tr])
            m = evaluate(ds.target[te], est.predict(X[te]), ds.target[tr])
            out.append(dict(n_estimators=n_est, method=name, seed=s, macro_f1=m["macro_f1"],
                            secs=round(time.perf_counter() - t0, 1)))
            print(f"  {n_est}t {name} seed {s}: {m['macro_f1']:.3f}", flush=True)

(RESULTS_DIR / "spot_trees.jsonl").write_text("\n".join(json.dumps(r) for r in out), encoding="utf-8")
for n_est in (100, 200):
    for name in ("Flat-RF+SMOTE", "RFOED-RF"):
        v = [r["macro_f1"] for r in out if r["n_estimators"] == n_est and r["method"] == name]
        print(f"{n_est}t {name}: {np.mean(v):.4f} +/- {np.std(v):.4f}")
