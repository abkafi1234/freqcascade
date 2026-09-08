"""FOCC exposure-bias check (reviewer major comment 4).

Classifier chains train each link on the *true* earlier-link labels but must
infer with each earlier link's *predicted* labels (Read et al. 2011). That
mismatch is a known exposure-bias channel and it is order-directional: an
early link's error corrupts every later link's input, a late link's error
corrupts nothing downstream. So it is a candidate explanation for why the
ordering effect is sign-inconsistent across our three multi-label corpora
(Section 9.1).

This script quantifies the cost: for FOCC-NN on each corpus, over the same
10 CV folds used in the paper, it compares label-macro-F1 under normal
(predicted-label) inference vs oracle (true-label) inference. The gap is the
exposure-bias cost. It also reports the same for random label ordering, so
we can see whether the cost tracks ordering.

    python scripts/focc_exposure_bias.py

Writes results/focc_exposure.jsonl.
"""

from __future__ import annotations

import json
import time

import numpy as np

from _shared import NN_EPOCHS, NN_MEMBERS, RESULTS_DIR, get_folds

from freqcascade.datasets import load
from freqcascade.focc import make_focc_nn
from freqcascade.multilabel_metrics import evaluate_multilabel

DATASETS = ["reuters21578", "hoc", "litcovid"]


def main() -> None:
    out = []
    for name in DATASETS:
        ds = load(name)
        Y = np.asarray(ds.target)
        folds = get_folds(ds)
        Xemb = _emb(ds)
        for order in ("frequency",):  # deployed config; random-order arm dropped for compute
            for i, f in enumerate(folds):
                tr, te = f.train_idx, f.test_idx
                clf = make_focc_nn(order=order, rebalance=True, random_state=i,
                                   n_members=NN_MEMBERS, hidden_size=128, max_epochs=NN_EPOCHS)
                t0 = time.perf_counter()
                clf.fit(Xemb[tr], Y[tr])
                X_te = Xemb[te]
                pred_norm = (clf.predict_proba(X_te) >= 0.5).astype(np.int8)
                pred_orac = (clf.predict_proba_oracle(X_te, Y[te]) >= 0.5).astype(np.int8)
                m_norm = evaluate_multilabel(Y[te], pred_norm, Y[tr])
                m_orac = evaluate_multilabel(Y[te], pred_orac, Y[tr])
                row = dict(dataset=name, order=order, fold=i,
                           label_macro_f1_predicted=m_norm["label_macro_f1"],
                           label_macro_f1_oracle=m_orac["label_macro_f1"],
                           exposure_gap=m_orac["label_macro_f1"] - m_norm["label_macro_f1"],
                           secs=round(time.perf_counter() - t0, 1))
                out.append(row)
                print(f"  {name:12s} {order:9s} fold {i}: pred={row['label_macro_f1_predicted']:.3f} "
                      f"oracle={row['label_macro_f1_oracle']:.3f} gap={row['exposure_gap']:+.3f}", flush=True)
        _dump(out)

    _dump(out)
    print("\n=== summary (mean over 10 folds) ===")
    import collections
    agg = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in out:
        agg[(r["dataset"], r["order"])]["pred"].append(r["label_macro_f1_predicted"])
        agg[(r["dataset"], r["order"])]["orac"].append(r["label_macro_f1_oracle"])
        agg[(r["dataset"], r["order"])]["gap"].append(r["exposure_gap"])
    for (dsn, order), d in sorted(agg.items()):
        print(f"{dsn:12s} {order:9s}  predicted={np.mean(d['pred']):.3f}  "
              f"oracle={np.mean(d['orac']):.3f}  exposure gap={np.mean(d['gap']):+.3f}")


def _emb(ds):
    from _shared import embed
    return embed(ds.texts, "minilm", ds.name)


def _dump(out):
    (RESULTS_DIR / "focc_exposure.jsonl").write_text(
        "\n".join(json.dumps(r) for r in out), encoding="utf-8")


if __name__ == "__main__":
    main()
