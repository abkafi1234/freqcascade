"""Controlled ablation.

Single-label: the 2x2x2 factorial -- ordering {frequency, random} x base
learner {rf, nn} x rebalance {on, off} -- response macro-F1, one value per
CV fold per cell. Analyse with freqcascade.stats.anova_or_srh.

Multi-label (FOCC): the 2x2 -- ordering x rebalance.

    python scripts/run_factorial_ablation.py wos46985
    python scripts/run_factorial_ablation.py --multilabel reuters21578
"""

from __future__ import annotations

import argparse
import itertools
import json
import time

import os

import numpy as np

from _shared import (
    NN_EPOCHS,
    NN_MEMBERS,
    RESULTS_DIR,
    RF_NJOBS,
    embed,
    get_folds,
    make_tfidf,
)

from freqcascade.datasets import load
from freqcascade.metrics import evaluate
from freqcascade.multilabel_metrics import evaluate_multilabel

# The factorial isolates main effects and interactions, not absolute accuracy:
# a lighter forest (100 trees) and 6 of the 10 CV folds per cell (still a
# well-powered 3-way ANOVA -- the effects the paper reports are large,
# F=46-930). Env FREQCASCADE_FACT_TREES / FREQCASCADE_FACT_FOLDS to restore.
FACT_TREES = int(os.environ.get("FREQCASCADE_FACT_TREES", "100"))
FACT_FOLDS = int(os.environ.get("FREQCASCADE_FACT_FOLDS", "6"))


def _rfoed(order, base, rebalance, seed):
    from freqcascade.base_learners import RFBaseLearner
    from freqcascade.decomposition import RFOEDClassifier
    from freqcascade.torch_ensemble import TorchNNEnsembleBaseLearner

    if base == "rf":
        node = lambda i: RFBaseLearner(n_estimators=FACT_TREES, rebalance=rebalance,
                                       random_state=seed * 1000 + i, n_jobs=RF_NJOBS)
    else:
        node = lambda i: TorchNNEnsembleBaseLearner(n_members=NN_MEMBERS, rebalance=rebalance,
                                                    hidden_size=128, max_epochs=NN_EPOCHS,
                                                    random_state=seed * 1000 + i)
    return RFOEDClassifier(base_learner_factory=node, order=order, random_state=seed)


def _focc(order, rebalance, base, seed):
    from freqcascade.focc import make_focc_nn, make_focc_rf

    if base == "rf":
        return make_focc_rf(order=order, rebalance=rebalance, random_state=seed,
                            n_estimators=FACT_TREES, n_jobs=RF_NJOBS)
    return make_focc_nn(order=order, rebalance=rebalance, random_state=seed,
                        n_members=NN_MEMBERS, hidden_size=128, max_epochs=NN_EPOCHS)


def _cell_done(name: str, cell: dict, n_folds: int) -> bool:
    out = RESULTS_DIR / f"ablation_{name}.jsonl"
    if not out.exists():
        return False
    k = sum(
        1 for line in out.read_text(encoding="utf-8").splitlines() if line.strip()
        and all(json.loads(line).get(k2) == v for k2, v in cell.items())
    )
    return k >= n_folds


def _append(rows, name):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"ablation_{name}.jsonl"
    with out.open("a", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    print(f"  +{len(rows)} rows -> {out.name}")


def run_single_label(name: str) -> None:
    ds = load(name)
    folds = get_folds(ds)[:FACT_FOLDS]
    feats = {"rf": None, "nn": None}
    for order, base, reb in itertools.product(("frequency", "random"), ("rf", "nn"), (True, False)):
        cell = dict(ordering=order, base_learner=base, rebalance=reb)
        if _cell_done(ds.name, cell, len(folds)):
            print(f"  skip {order}/{base}/reb={int(reb)} (done)")
            continue
        if feats[base] is None:
            feats[base] = make_tfidf(ds.texts, ds.texts) if base == "rf" else embed(ds.texts, "minilm", ds.name)
        X = feats[base]
        rows = []
        for i, f in enumerate(folds):
            est = _rfoed(order, base, reb, i)
            t0 = time.perf_counter()
            est.fit(X[f.train_idx], ds.target[f.train_idx])
            m = evaluate(ds.target[f.test_idx], est.predict(X[f.test_idx]), ds.target[f.train_idx])
            rows.append(dict(**cell, fold=i, macro_f1=m["macro_f1"], gmean=m["gmean"],
                             secs=round(time.perf_counter() - t0, 1)))
            print(f"  {order:9} {base} reb={int(reb)} fold {i}: macro_f1={m['macro_f1']:.3f}")
        _append(rows, ds.name)


def run_multilabel(name: str, base: str = "nn") -> None:
    ds = load(name)
    folds = get_folds(ds)[:FACT_FOLDS]
    X = None
    for order, reb in itertools.product(("frequency", "random"), (True, False)):
        cell = dict(ordering=order, rebalance=reb, base_learner=base)
        if _cell_done(ds.name, cell, len(folds)):
            print(f"  skip {order}/reb={int(reb)} (done)")
            continue
        if X is None:
            X = embed(ds.texts, "minilm", ds.name) if base == "nn" else make_tfidf(ds.texts, ds.texts)
        rows = []
        for i, f in enumerate(folds):
            est = _focc(order, reb, base, i)
            est.fit(X[f.train_idx], ds.target[f.train_idx])
            m = evaluate_multilabel(ds.target[f.test_idx], est.predict(X[f.test_idx]), ds.target[f.train_idx])
            rows.append(dict(**cell, fold=i, label_macro_f1=m["label_macro_f1"]))
            print(f"  {order:9} reb={int(reb)} fold {i}: label_macro_f1={m['label_macro_f1']:.3f}")
        _append(rows, ds.name)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("datasets", nargs="+")
    ap.add_argument("--multilabel", action="store_true")
    ap.add_argument("--base", default="nn", help="multi-label ablation base learner (nn|rf)")
    args = ap.parse_args()
    for name in args.datasets:
        (run_multilabel(name, args.base) if args.multilabel else run_single_label(name))


if __name__ == "__main__":
    main()
