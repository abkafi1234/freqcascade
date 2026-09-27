"""Fixed versus data-scaled rebalancing cap (round-3 R8).

The review objects that a single n_cap = 2000 across corpora from 5k to 50k
documents is arbitrary, and proposes scaling it with the data:

    n_cap(n, K) = max(n_pos, min(n_neg, gamma * n / K))

implemented as cap_rule="adaptive" in rebalance.balanced_bootstrap_indices
with max_per_class = gamma * n_train / K (n_train and K of the fold being
fitted; K = label count for the multi-label chain). Both strata are still
drawn to one common size, so the training prior stays 0.5 and the Elkan
correction is unaffected. gamma -> infinity recovers the uncapped draw.

Arms per corpus: fixed-2000 (the paper) | uncapped | adaptive gamma in
{2, 8, 32, 128}. The cap was originally justified by GPU memory, so memory is
measured here rather than asserted: peak allocated CUDA memory over the whole
fit+predict of each cell, alongside wall time and macro-F1 (label-macro-F1 for
the chains). Fold-to-fold dispersion is the stability measure.

Selection note: these runs use the paper's own test folds, so they are a
sensitivity analysis. Any finite gamma adopted as a new default must be chosen
on validation data carved from the training folds, not read off this table.

    python scripts/run_ncap_adaptive.py --probe
    python scripts/run_ncap_adaptive.py --folds 5

Writes results/ncap_adaptive.jsonl (resumable per corpus x arm x fold).
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch

from _shared import NN_EPOCHS, NN_MEMBERS, RESULTS_DIR, embed, get_folds

from freqcascade.datasets import load
from freqcascade.decomposition import RFOEDClassifier
from freqcascade.focc import make_focc_nn
from freqcascade.metrics import evaluate
from freqcascade.multilabel_metrics import evaluate_multilabel

SINGLE = ["20newsgroups", "ohsumed_23", "clinc150", "wos46985", "drug_reviews"]
MULTI = ["hoc", "litcovid", "reuters21578"]
GAMMAS = [2, 8, 32, 128]
OUT = RESULTS_DIR / "ncap_adaptive.jsonl"


def arms():
    yield "fixed", None, 2000
    yield "uncapped", None, None
    for g in GAMMAS:
        yield "adaptive", g, None


def partitions(ds, n_folds):
    if ds.split is not None:
        is_test = np.asarray(ds.split) == "test"
        tr, te = np.flatnonzero(~is_test), np.flatnonzero(is_test)
        return [(s, tr, te) for s in range(n_folds)]
    return [(i, f.train_idx, f.test_idx) for i, f in enumerate(get_folds(ds))][:n_folds]


def done():
    if not OUT.exists():
        return set()
    return {(r["dataset"], r["rule"], r["gamma"], r["fold"]) for r in
            (json.loads(x) for x in OUT.read_text(encoding="utf-8").splitlines() if x.strip())}


def run(names, n_folds, probe):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    seen = done()
    for name in names:
        ds = load(name)
        X = embed(ds.texts, "minilm", ds.name)
        Y = np.asarray(ds.target)
        multi = ds.is_multilabel
        for fold, tr, te in partitions(ds, n_folds):
            K = Y.shape[1] if multi else int(len(np.unique(Y[tr])))
            for rule, gamma, fixed_cap in arms():
                if (ds.name, rule, gamma, fold) in seen:
                    continue
                if rule == "fixed":
                    cap, cap_rule = fixed_cap, "fixed"
                elif rule == "uncapped":
                    cap, cap_rule = None, "fixed"
                else:
                    cap, cap_rule = int(round(gamma * len(tr) / K)), "adaptive"
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                t0 = time.perf_counter()
                try:
                    if multi:
                        est = make_focc_nn(order="frequency", rebalance=True, random_state=fold,
                                           n_members=NN_MEMBERS, hidden_size=128, max_epochs=NN_EPOCHS,
                                           max_bootstrap_per_class=cap, cap_rule=cap_rule)
                        est.fit(X[tr], Y[tr])
                        m = evaluate_multilabel(Y[te], est.predict(X[te]), Y[tr])
                        score = m["label_macro_f1"]
                    else:
                        from freqcascade.torch_ensemble import TorchNNEnsembleBaseLearner
                        fac = lambda node, s=fold, c=cap, r=cap_rule: TorchNNEnsembleBaseLearner(  # noqa: E731
                            n_members=NN_MEMBERS, rebalance=True, hidden_size=128, max_epochs=NN_EPOCHS,
                            random_state=s * 1000 + node, max_bootstrap_per_class=c, cap_rule=r)
                        est = RFOEDClassifier(base_learner_factory=fac, order="frequency", random_state=fold)
                        est.fit(X[tr], Y[tr])
                        m = evaluate(Y[te], est.predict(X[te]), Y[tr])
                        score = m["macro_f1"]
                    row = dict(m, score=score)
                except Exception as e:
                    row = dict(error="%s: %s" % (type(e).__name__, str(e)[:200]), score=None)
                row.update(dataset=ds.name, multilabel=bool(multi), rule=rule, gamma=gamma, cap=cap,
                           fold=fold, K=K, n_train=int(len(tr)), n_members=NN_MEMBERS, epochs=NN_EPOCHS,
                           secs=round(time.perf_counter() - t0, 1),
                           peak_gpu_mem_gb=round(torch.cuda.max_memory_allocated() / 1e9, 3))
                with OUT.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(row) + "\n")
                print("  %-18s fold %d %-9s gamma=%-4s cap=%-6s score=%s  %.0fs  peak %.2f GB"
                      % (ds.name, fold, rule, gamma, cap,
                         ("%.4f" % row["score"]) if row["score"] is not None else row["error"][:50],
                         row["secs"], row["peak_gpu_mem_gb"]), flush=True)
                if probe:
                    return


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("datasets", nargs="*", default=None)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--probe", action="store_true")
    a = ap.parse_args()
    run(a.datasets or (SINGLE + MULTI), a.folds, a.probe)


if __name__ == "__main__":
    main()
