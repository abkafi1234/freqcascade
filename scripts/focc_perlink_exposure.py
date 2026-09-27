"""Per-link exposure bias down the FOCC chain (round-3 R9).

focc_exposure_bias.py reports one number per fold: the label-macro-F1 gap
between deployed inference (each link augmented with earlier links'
*predicted* labels) and oracle inference (earlier links' *true* labels). The
reviewer asks where along the chain that cost accrues, and in particular
whether late links -- which on Reuters-21578 sit up to 89 steps downstream --
are degraded by accumulated upstream errors.

Under frequency ordering, chain position and label rarity are the same axis:
the last links are the rarest labels. A late-link drop could therefore be
exposure bias or simply tail-label difficulty. Random ordering breaks that
coupling, so both orders are run on the identical folds.

For every fold, order and link this records, under deployed and oracle
conditioning: precision, recall, F1 and false-positive rate on the link's own
label; the mean number of erroneous upstream labels fed into that link's
augmented feature vector (the injected error), and the share of documents with
at least one; the label's training frequency and test support.

    python scripts/focc_perlink_exposure.py --probe
    python scripts/focc_perlink_exposure.py

Writes results/focc_perlink.jsonl (one row per dataset x order x fold x link;
resumable per dataset x order x fold).
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np

from _shared import NN_EPOCHS, NN_MEMBERS, RESULTS_DIR, embed, get_folds

from freqcascade.datasets import load
from freqcascade.focc import make_focc_nn

DATASETS = ["reuters21578", "hoc", "litcovid"]
ORDERS = ["frequency", "random"]
OUT = RESULTS_DIR / "focc_perlink.jsonl"


def prf(y, p):
    tp = int(((p == 1) & (y == 1)).sum())
    fp = int(((p == 1) & (y == 0)).sum())
    fn = int(((p == 0) & (y == 1)).sum())
    tn = int(((p == 0) & (y == 0)).sum())
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    return dict(precision=round(prec, 5), recall=round(rec, 5), f1=round(f1, 5), fpr=round(fpr, 6))


def done():
    if not OUT.exists():
        return set()
    return {(r["dataset"], r["order"], r["fold"]) for r in
            (json.loads(x) for x in OUT.read_text(encoding="utf-8").splitlines() if x.strip())}


def run(datasets, orders, probe=False, max_folds=None):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    seen = done()
    for name in datasets:
        ds = load(name)
        Y = np.asarray(ds.target).astype(np.int8)
        X = embed(ds.texts, "minilm", ds.name)
        folds = get_folds(ds)
        for order in orders:
            for i, f in enumerate(folds):
                if max_folds is not None and i >= max_folds:
                    break
                if (ds.name, order, i) in seen:
                    continue
                tr, te = f.train_idx, f.test_idx
                t0 = time.perf_counter()
                clf = make_focc_nn(order=order, rebalance=True, random_state=i,
                                   n_members=NN_MEMBERS, hidden_size=128, max_epochs=NN_EPOCHS)
                clf.fit(X[tr], Y[tr])
                Yte = Y[te]
                dep = (clf.predict_proba(X[te]) >= 0.5).astype(np.int8)
                orc = (clf.predict_proba_oracle(X[te], Yte) >= 0.5).astype(np.int8)
                train_freq = Y[tr].mean(axis=0)
                wrong = (dep != Yte)                         # docs x labels, deployed errors
                order_idx = clf.label_order_
                cum = np.zeros(len(te), dtype=np.int32)      # upstream errors seen by the current link
                rows = []
                for rank, lab in enumerate(order_idx):
                    rows.append(dict(
                        dataset=ds.name, order=order, fold=i, rank=rank, n_links=len(order_idx),
                        label=int(lab), train_freq=round(float(train_freq[lab]), 6),
                        test_support=int(Yte[:, lab].sum()),
                        upstream_err_mean=round(float(cum.mean()), 4),
                        upstream_err_any=round(float((cum > 0).mean()), 5),
                        deployed=prf(Yte[:, lab], dep[:, lab]),
                        oracle=prf(Yte[:, lab], orc[:, lab]),
                    ))
                    cum += wrong[:, lab]
                secs = round(time.perf_counter() - t0, 1)
                with OUT.open("a", encoding="utf-8") as fh:
                    for r in rows:
                        r["fold_secs"] = secs
                        fh.write(json.dumps(r) + "\n")
                gap = np.mean([r["oracle"]["f1"] - r["deployed"]["f1"] for r in rows])
                last = rows[-1]
                print("  %-12s %-9s fold %d  links=%d  mean per-link F1 gap=%+.4f  "
                      "upstream errors at last link=%.2f  %.0fs"
                      % (ds.name, order, i, len(rows), gap, last["upstream_err_mean"], secs), flush=True)
                if probe:
                    return


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("datasets", nargs="*", default=None)
    ap.add_argument("--orders", default=",".join(ORDERS))
    ap.add_argument("--max-folds", type=int, default=None)
    ap.add_argument("--probe", action="store_true")
    a = ap.parse_args()
    run(a.datasets or DATASETS, a.orders.split(","), probe=a.probe, max_folds=a.max_folds)


if __name__ == "__main__":
    main()
