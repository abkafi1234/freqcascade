"""Is FOCC-NN's rare-label advantage a threshold artefact? (round-3 follow-up)

The flat multi-output MLP predicts at sklearn's fixed 0.5 threshold; FOCC-NN's
per-link rebalancing shifts its operating point toward recall. This script asks
whether the flat MLP, moved to a comparable operating point, closes the gap.
Pre-registered in KBS/plan.md (2026-09-19, 12:00) before any run.

  harness  refit Flat-MLP per fold; thresholding at 0.5 must reproduce the stored
           Flat-MLP rows, else abort
  (A)      one global threshold tuned on an inner 80/20 split of the training fold
           (label-macro-F1), applied to the full-training-fold model   [deployable]
  (B)      per-label thresholds tuned the same way (per-label F1)       [deployable]
  (C)      one global threshold chosen on the TEST fold so the flat MLP's
           label-macro-recall matches FOCC-NN's on that fold            [diagnostic]

    python scripts/threshold_flat_mlp.py            # all three corpora
    python scripts/threshold_flat_mlp.py hoc

Writes results/threshold_flat_mlp.jsonl (one row per corpus x fold x arm).
"""

from __future__ import annotations

import argparse
import json

import numpy as np
from sklearn.metrics import f1_score
from sklearn.neural_network import MLPClassifier

from _shared import DEFAULT_ENCODER, FLAT_MLP_ML_KW, RESULTS_DIR, embed, get_folds

from freqcascade.cv import _iterative_stratify
from freqcascade.datasets import load
from freqcascade.multilabel_metrics import evaluate_multilabel

DATASETS = ["reuters21578", "hoc", "litcovid"]
GRID_A = np.round(np.arange(0.02, 0.501, 0.02), 3)
GRID_C = np.round(np.arange(0.005, 0.9951, 0.005), 3)
OUT = RESULTS_DIR / "threshold_flat_mlp.jsonl"


def _stored(tag, method):
    rows = [json.loads(x) for x in (RESULTS_DIR / f"{tag}.jsonl").read_text().splitlines() if x.strip()]
    return {r["fold"]: r for r in rows if r.get("method") == method and "error" not in r}


def _proba(X_fit, Y_fit, X_eval, seed):
    est = MLPClassifier(random_state=seed, **FLAT_MLP_ML_KW).fit(X_fit, Y_fit)
    return est.predict_proba(X_eval), est


def _macro_f1(Y, P):
    return f1_score(Y, P, average="macro", zero_division=0)


def run(name):
    ds = load(name)
    Y = np.asarray(ds.target)
    X = embed(ds.texts, DEFAULT_ENCODER, ds.name)
    folds = get_folds(ds)
    flat, focc = _stored(ds.name, "Flat-MLP"), _stored(ds.name, "FOCC-NN")
    rows = []
    for i, f in enumerate(folds):
        tr, te = f.train_idx, f.test_idx
        P_te, est = _proba(X[tr], Y[tr], X[te], i)

        # harness: the stored row is est.predict, i.e. probability > 0.5
        base = evaluate_multilabel(Y[te], est.predict(X[te]), Y[tr])
        for k in ("label_macro_f1", "bottom_quartile_label_recall", "micro_f1"):
            if abs(base[k] - flat[i][k]) > 1e-9:
                raise SystemExit(f"harness mismatch {name} fold {i} {k}: {base[k]} vs stored {flat[i][k]}")
        assert np.array_equal(est.predict(X[te]), (P_te > 0.5).astype(int))

        # inner, label-stratified 80/20 split of the training fold
        parts = _iterative_stratify(Y[tr], [0.8, 0.2], np.random.default_rng(1000 + i))
        itr, iva = tr[np.sort(parts[0])], tr[np.sort(parts[1])]
        P_va, _ = _proba(X[itr], Y[itr], X[iva], i)
        Yva = Y[iva]

        tA = max(GRID_A, key=lambda t: _macro_f1(Yva, (P_va > t).astype(int)))
        tB = np.full(Y.shape[1], 0.5)
        for j in range(Y.shape[1]):
            if Yva[:, j].sum() > 0:
                tB[j] = max(GRID_A, key=lambda t: f1_score(Yva[:, j], (P_va[:, j] > t).astype(int),
                                                           zero_division=0))
        target = focc[i]["label_macro_recall"]
        recalls = {t: evaluate_multilabel(Y[te], (P_te > t).astype(int), Y[tr])["label_macro_recall"]
                   for t in GRID_C}
        tC = min(GRID_C, key=lambda t: abs(recalls[t] - target))

        arms = {"0.5 (stored)": (P_te > 0.5), "A global-tuned": (P_te > tA),
                "B per-label-tuned": (P_te > tB[None, :]), "C recall-matched": (P_te > tC)}
        for arm, pred in arms.items():
            m = evaluate_multilabel(Y[te], pred.astype(int), Y[tr])
            thr = {"0.5 (stored)": 0.5, "A global-tuned": float(tA), "C recall-matched": float(tC)}.get(arm)
            rows.append(dict(dataset=ds.name, fold=i, arm=arm, threshold=thr,
                             focc_label_macro_recall=target, **m))
        print(f"  {name} fold {i}: tA={tA:.2f} tC={tC:.3f} (target recall {target:.3f}, got {recalls[tC]:.3f})",
              flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("datasets", nargs="*", default=DATASETS)
    a = ap.parse_args()
    done = set()
    if OUT.exists():
        done = {json.loads(x)["dataset"] for x in OUT.read_text().splitlines() if x.strip()}
    for name in a.datasets:
        if load(name).name in done:
            print(f"  {name}: done, skipping"); continue
        rows = run(name)
        with OUT.open("a") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
        print(f"wrote {len(rows)} rows -> {OUT}")


if __name__ == "__main__":
    main()
