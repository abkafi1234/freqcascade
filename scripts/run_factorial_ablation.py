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

import numpy as np

from _shared import (
    N_ESTIMATORS,
    NN_EPOCHS,
    NN_MEMBERS,
    RESULTS_DIR,
    embed,
    get_folds,
    make_tfidf,
)

from freqcascade.datasets import load
from freqcascade.metrics import evaluate
from freqcascade.multilabel_metrics import evaluate_multilabel


def _rfoed(order, base, rebalance, seed):
    from freqcascade.base_learners import RFBaseLearner
    from freqcascade.decomposition import RFOEDClassifier
    from freqcascade.torch_ensemble import TorchNNEnsembleBaseLearner

    if base == "rf":
        node = lambda i: RFBaseLearner(n_estimators=N_ESTIMATORS, rebalance=rebalance,
                                       random_state=seed * 1000 + i, n_jobs=4)
    else:
        node = lambda i: TorchNNEnsembleBaseLearner(n_members=NN_MEMBERS, rebalance=rebalance,
                                                    hidden_size=128, max_epochs=NN_EPOCHS,
                                                    random_state=seed * 1000 + i)
    return RFOEDClassifier(base_learner_factory=node, order=order, random_state=seed)


def _focc(order, rebalance, base, seed):
    from freqcascade.focc import make_focc_nn, make_focc_rf

    if base == "rf":
        return make_focc_rf(order=order, rebalance=rebalance, random_state=seed, n_estimators=N_ESTIMATORS)
    return make_focc_nn(order=order, rebalance=rebalance, random_state=seed,
                        n_members=NN_MEMBERS, hidden_size=128, max_epochs=NN_EPOCHS)


def run_single_label(name: str) -> None:
    ds = load(name)
    folds = get_folds(ds)
    tfidf = make_tfidf(ds.texts, ds.texts)
    emb = embed(ds.texts, "minilm", ds.name)
    rows = []
    for order, base, reb in itertools.product(("frequency", "random"), ("rf", "nn"), (True, False)):
        X = tfidf if base == "rf" else emb
        for i, f in enumerate(folds):
            est = _rfoed(order, base, reb, i)
            t0 = time.perf_counter()
            est.fit(X[f.train_idx], ds.target[f.train_idx])
            pred = est.predict(X[f.test_idx])
            m = evaluate(ds.target[f.test_idx], pred, ds.target[f.train_idx])
            rows.append(dict(ordering=order, base_learner=base, rebalance=reb, fold=i,
                             macro_f1=m["macro_f1"], gmean=m["gmean"],
                             secs=round(time.perf_counter() - t0, 1)))
            print(f"  {order:9} {base} reb={int(reb)} fold {i}: macro_f1={m['macro_f1']:.3f}")
    _write(rows, f"ablation_{ds.name}")


def run_multilabel(name: str, base: str = "nn") -> None:
    ds = load(name)
    folds = get_folds(ds)
    X = embed(ds.texts, "minilm", ds.name) if base == "nn" else make_tfidf(ds.texts, ds.texts)
    rows = []
    for order, reb in itertools.product(("frequency", "random"), (True, False)):
        for i, f in enumerate(folds):
            est = _focc(order, reb, base, i)
            est.fit(X[f.train_idx], ds.target[f.train_idx])
            pred = est.predict(X[f.test_idx])
            m = evaluate_multilabel(ds.target[f.test_idx], pred, ds.target[f.train_idx])
            rows.append(dict(ordering=order, rebalance=reb, base_learner=base, fold=i,
                             label_macro_f1=m["label_macro_f1"]))
            print(f"  {order:9} reb={int(reb)} fold {i}: label_macro_f1={m['label_macro_f1']:.3f}")
    _write(rows, f"ablation_{ds.name}")


def _write(rows, name):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"{name}.jsonl"
    out.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    print(f"wrote {len(rows)} rows -> {out}")


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
