"""Per-class test recall for the proposed method vs the strongest baseline,
so Figure 6 (rarest-quartile per-class recall) can be drawn.

    python scripts/dump_per_class_recall.py clinc150
    python scripts/dump_per_class_recall.py --proposed RFOED-NN --baseline Flat-RF+SMOTE clinc150

Writes results/perclass_<dataset>.jsonl: one row per (method, class) with the
class's pooled test recall and its training frequency.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
from sklearn.metrics import recall_score

from _shared import (
    DEFAULT_ENCODER,
    RESULTS_DIR,
    SINGLE_LABEL_METHODS,
    embed,
    get_folds,
    make_tfidf,
)

from freqcascade.datasets import load


def per_class_recall(name: str, proposed: str, baseline: str, encoder: str) -> None:
    ds = load(name)
    methods = SINGLE_LABEL_METHODS()
    fixed = ds.split if ds.split is not None else None
    folds = None if fixed is not None else get_folds(ds)
    train_mask = (ds.split != "test") if fixed is not None else np.ones(ds.n_samples, bool)

    classes = np.array(sorted(set(ds.target.tolist())))
    _, train_counts = np.unique(ds.target[train_mask], return_counts=True)
    count_of = dict(zip(*np.unique(ds.target[train_mask], return_counts=True)))

    rows = []
    for mname in (proposed, baseline):
        m = methods[mname]
        X = make_tfidf(ds.texts[train_mask], ds.texts) if m.kind == "rf" else embed(ds.texts, encoder, ds.name)
        y_true_all, y_pred_all = [], []
        iters = (
            [(np.flatnonzero(ds.split != "test"), np.flatnonzero(ds.split == "test"))] * 3
            if fixed is not None
            else [(f.train_idx, f.test_idx) for f in folds]
        )
        for s, (tr, te) in enumerate(iters):
            est = m.build(s)
            est.fit(X[tr], ds.target[tr])
            y_true_all.append(ds.target[te])
            y_pred_all.append(est.predict(X[te]))
        yt, yp = np.concatenate(y_true_all), np.concatenate(y_pred_all)
        rec = recall_score(yt, yp, labels=list(classes), average=None, zero_division=0)
        for c, r in zip(classes, rec):
            rows.append(dict(method=mname, dataset=name, **{"class": str(c)},
                             recall=float(r), train_count=int(count_of.get(c, 0))))

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"perclass_{ds.name}.jsonl"
    out.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    print(f"wrote {len(rows)} rows -> {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("datasets", nargs="+")
    ap.add_argument("--proposed", default="RFOED-NN")
    ap.add_argument("--baseline", default="Flat-RF+SMOTE")
    ap.add_argument("--encoder", default=DEFAULT_ENCODER)
    args = ap.parse_args()
    for name in args.datasets:
        per_class_recall(name, args.proposed, args.baseline, args.encoder)


if __name__ == "__main__":
    main()
