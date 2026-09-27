"""CLINC150 under pooled 5x2 cross-validation (reviewer response R4).

The main results evaluate CLINC150 on its provided fixed train/val/test split,
repeated over 10 seeds, while every other single-label corpus uses repeated
stratified 5x2 cross-validation. The fixed split is what the CLINC150
literature reports against, so the paper keeps it; but it means the ten
CLINC150 repetitions vary only training stochasticity, not the data partition,
and the paired Wilcoxon therefore estimates a different quantity there than
elsewhere.

This script re-runs the whole single-label method set on CLINC150 with the
protocol used everywhere else -- pooled data, repeated stratified 5x2 CV at the
same CV_SEED -- so the paper can report whether any conclusion depends on the
protocol. It is an addition to the fixed-split results, not a replacement.

Two deviations from the fixed-split run, both deliberate and both making the
comparison stricter rather than weaker:

  * TF-IDF is fit inside each fold on that fold's training rows only. The
    pooled-CV path in _shared.py fits the vectorizer on all rows, which lets
    test-fold vocabulary and document frequencies inform the representation.
    Fitting per fold removes that.
  * The train/test partition genuinely resamples, so the confidence intervals
    reflect sampling variability rather than initialisation noise alone.

    python scripts/run_clinc150_cv.py --methods RFOED-NN,Flat-RF+SMOTE
    python scripts/run_clinc150_cv.py

Writes results/clinc150_cv.jsonl (one row per method x fold, resumable).
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np

from _shared import (
    CV_REPEATS,
    CV_SEED,
    CV_SPLITS,
    DEFAULT_ENCODER,
    RESULTS_DIR,
    SINGLE_LABEL_METHODS,
    embed,
    make_tfidf,
)

from freqcascade.datasets import load
from freqcascade.metrics import evaluate

OUT = RESULTS_DIR / "clinc150_cv.jsonl"


def done():
    if not OUT.exists():
        return set()
    rows = (json.loads(x) for x in OUT.read_text(encoding="utf-8").splitlines() if x.strip())
    return {(r["method"], r["fold"]) for r in rows if "error" not in r}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", default=None, help="comma-separated subset")
    a = ap.parse_args()

    from freqcascade import cv

    ds = load("clinc150")
    y = np.asarray(ds.target)
    print("=== clinc150 pooled 5x2 CV === n=%d K=%d" % (len(y), len(np.unique(y))), flush=True)

    # Pooled folds over ALL rows: the dataset's own split is deliberately ignored.
    folds = cv.repeated_stratified_kfold(y, n_repeats=CV_REPEATS, n_splits=CV_SPLITS, seed=CV_SEED)

    methods = SINGLE_LABEL_METHODS()
    names = a.methods.split(",") if a.methods else list(methods)
    seen = done()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    X_nn = embed(ds.texts, DEFAULT_ENCODER, ds.name)

    for fi, f in enumerate(folds):
        tr, te = f.train_idx, f.test_idx
        # Per-fold TF-IDF, fit on this fold's training rows only.
        X_rf = None
        for mname in names:
            if (mname, fi) in seen:
                continue
            m = methods[mname]
            if m.kind == "rf" and X_rf is None:
                X_rf = make_tfidf(ds.texts[tr], ds.texts)
            X = X_rf if m.kind == "rf" else X_nn
            t0 = time.perf_counter()
            try:
                est = m.build(fi)
                est.fit(X[tr], y[tr])
                met = evaluate(y[te], est.predict(X[te]), y[tr])
            except Exception as e:
                met = dict(error="%s: %s" % (type(e).__name__, e))
            met.update(method=mname, fold=fi, dataset="clinc150_cv",
                       protocol="pooled 5x2 CV, per-fold TF-IDF",
                       fit_predict_s=round(time.perf_counter() - t0, 2))
            with OUT.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(met) + "\n")
            print("  fold %d  %-22s %s" % (
                fi, mname,
                ("macroF1=%.4f" % met["macro_f1"]) if "error" not in met
                else met["error"][:70]), flush=True)


if __name__ == "__main__":
    main()
