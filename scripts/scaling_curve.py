"""Class-count scaling curve -- the paper's central-claim figure.

Thesis: purpose-built resampling ensembles collapse as class count K grows,
while local decomposition stays well-posed. Existing evidence shows this at
one K per dataset; this script sweeps K on a single dataset with everything
else held fixed, so the collapse is shown to be a function of K itself.

CLINC150 is subsampled to K in {5,10,25,50,100,150} randomly chosen classes
(several seeds), its fixed train/test split restricted to those classes, and
macro-F1 recorded for Flat-RF+SMOTE / EasyEnsemble / RUSBoost / RFOED-RF /
RFOED-NN -- the same method builders used in the main results. CLINC150's
natural imbalance ratio is only ~4x, so any collapse here is attributable to
cardinality, not to an extreme ratio.

    python scripts/scaling_curve.py
    FREQCASCADE_SCALING_SEEDS=5 python scripts/scaling_curve.py

Writes results/scaling_clinc150.jsonl (resumable: finished (K, seed, method)
cells are skipped on re-run).
"""

from __future__ import annotations

import json
import os
import time

import numpy as np

from _shared import RESULTS_DIR, SINGLE_LABEL_METHODS, embed, make_tfidf

from freqcascade.datasets import load
from freqcascade.metrics import evaluate

DATASET = "clinc150"
K_VALUES = [5, 10, 25, 50, 100, 150]
N_SEEDS = int(os.environ.get("FREQCASCADE_SCALING_SEEDS", "3"))
METHODS = ["Flat-RF+SMOTE", "EasyEnsemble", "RUSBoost", "RFOED-RF", "RFOED-NN"]
OUT = RESULTS_DIR / "scaling_clinc150.jsonl"


def _done() -> set:
    if not OUT.exists():
        return set()
    seen = set()
    for line in OUT.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if "macro_f1" in r:
            seen.add((r["K"], r["seed"], r["method"]))
    return seen


def _append(row: dict) -> None:
    with OUT.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row) + "\n")


def main() -> None:
    ds = load(DATASET)
    y = np.asarray(ds.target)
    is_test = np.asarray(ds.split) == "test"
    classes_all = np.unique(y)
    reg = SINGLE_LABEL_METHODS()
    done = _done()
    emb = None

    for seed in range(N_SEEDS):
        rng = np.random.default_rng(seed)
        for K in K_VALUES:
            # sample the same K classes for every method at this (seed, K)
            sub = set(rng.choice(classes_all, size=K, replace=False).tolist())
            mask = np.isin(y, list(sub))
            tr = np.flatnonzero(mask & ~is_test)
            te = np.flatnonzero(mask & is_test)
            X_tf = None
            for mname in METHODS:
                if (K, seed, mname) in done:
                    continue
                meth = reg[mname]
                if meth.kind == "nn":
                    if emb is None:
                        emb = embed(ds.texts, "minilm", ds.name)
                    X = emb
                else:
                    if X_tf is None:
                        X_tf = make_tfidf(ds.texts[tr], ds.texts)
                    X = X_tf
                try:
                    est = meth.build(seed)
                    t0 = time.perf_counter()
                    est.fit(X[tr], y[tr])
                    pred = est.predict(X[te])
                    m = evaluate(y[te], pred, y[tr])
                    row = dict(dataset=DATASET, K=K, seed=seed, method=mname,
                               macro_f1=m["macro_f1"], gmean=m.get("gmean"),
                               n_train=int(len(tr)), n_test=int(len(te)),
                               secs=round(time.perf_counter() - t0, 1))
                    print(f"  K={K:3d} seed={seed} {mname:16s} macro_f1={m['macro_f1']:.3f}", flush=True)
                except Exception as e:  # noqa: BLE001
                    row = dict(dataset=DATASET, K=K, seed=seed, method=mname, error=str(e)[:200])
                    print(f"  K={K:3d} seed={seed} {mname:16s} ERROR {str(e)[:80]}", flush=True)
                _append(row)

    # summary
    import collections
    agg = collections.defaultdict(list)
    for line in OUT.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if "macro_f1" in r:
            agg[(r["method"], r["K"])].append(r["macro_f1"])
    print("\n=== mean macro-F1 by (method, K) ===")
    for mname in METHODS:
        cells = " ".join(f"K{K}={np.mean(agg[(mname, K)]):.3f}" if agg[(mname, K)] else f"K{K}=--"
                         for K in K_VALUES)
        print(f"{mname:16s} {cells}")


if __name__ == "__main__":
    main()
