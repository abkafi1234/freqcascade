"""Select the data-scaled rebalancing cap on validation data (round-3 D2).

R8 compared fixed-2000, uncapped and adaptive gamma in {2, 8, 32, 128} on the
paper's own test folds. Uncapped tied or beat every gamma on 7 of 8 corpora but
lost to gamma=8 on HoC by 0.0009, so under the rule fixed before the data were
seen a finite gamma is required, chosen on validation data carved from the
*training* folds. This script does that, and nothing here touches a test
partition. The protocol was written into plan.md before this script first ran:

  for each corpus and folds 0-2, split the fold's training partition 80/20 into
  inner-train / inner-validation (stratified for single-label, random for
  multi-label; seed = fold index); fit RFOED-NN (FOCC-NN for multi-label) on
  inner-train with cap = gamma * n_inner_train / K under cap_rule="adaptive";
  score macro-F1 (label-macro-F1) on inner-validation. Per corpus, average over
  folds; a gamma's regret is the best gamma's mean minus its own. Select the
  smallest mean regret over the 8 corpora, preferring the larger gamma when two
  are within 0.001.

    python scripts/select_ncap_gamma.py            # run (resumable per corpus x fold x gamma)
    python scripts/select_ncap_gamma.py --report   # apply the criterion to what is on disk

Writes results/ncap_gamma_selection.jsonl.
"""

from __future__ import annotations

import argparse
import collections
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

CORPORA = ["20newsgroups", "ohsumed_23", "clinc150", "wos46985", "drug_reviews", "hoc", "litcovid", "reuters21578"]
GAMMAS = [2, 8, 32, 128]
FOLDS = [0, 1, 2]
VAL_FRAC = 0.2
TIE = 0.001
OUT = RESULTS_DIR / "ncap_gamma_selection.jsonl"


def training_partition(ds, fold):
    if ds.split is not None:                      # CLINC150: fixed split, vary the inner split by fold
        return np.flatnonzero(np.asarray(ds.split) != "test")
    return np.asarray(get_folds(ds)[fold].train_idx)


def inner_split(y, tr, multi, seed):
    from sklearn.model_selection import train_test_split
    if multi:
        rng = np.random.default_rng(seed)
        perm = rng.permutation(tr)
        k = int(round(VAL_FRAC * len(perm)))
        return perm[k:], perm[:k]
    a, b = train_test_split(tr, test_size=VAL_FRAC, random_state=seed, stratify=np.asarray(y)[tr])
    return np.asarray(a), np.asarray(b)


def done():
    if not OUT.exists():
        return set()
    return {(r["dataset"], r["fold"], r["gamma"]) for r in
            (json.loads(x) for x in OUT.read_text(encoding="utf-8").splitlines() if x.strip())}


def run():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    seen = done()
    for name in CORPORA:
        ds = load(name)
        X = embed(ds.texts, "minilm", ds.name)
        Y = np.asarray(ds.target)
        multi = bool(ds.is_multilabel)
        for fold in FOLDS:
            tr_all = training_partition(ds, fold)
            itr, iva = inner_split(Y, tr_all, multi, seed=fold)
            K = Y.shape[1] if multi else int(len(np.unique(Y[itr])))
            for gamma in GAMMAS:
                if (ds.name, fold, gamma) in seen:
                    continue
                cap = int(round(gamma * len(itr) / K))
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                t0 = time.perf_counter()
                try:
                    if multi:
                        est = make_focc_nn(order="frequency", rebalance=True, random_state=fold,
                                           n_members=NN_MEMBERS, hidden_size=128, max_epochs=NN_EPOCHS,
                                           max_bootstrap_per_class=cap, cap_rule="adaptive")
                        est.fit(X[itr], Y[itr])
                        score = evaluate_multilabel(Y[iva], est.predict(X[iva]), Y[itr])["label_macro_f1"]
                    else:
                        from freqcascade.torch_ensemble import TorchNNEnsembleBaseLearner
                        fac = lambda node, s=fold, c=cap: TorchNNEnsembleBaseLearner(  # noqa: E731
                            n_members=NN_MEMBERS, rebalance=True, hidden_size=128, max_epochs=NN_EPOCHS,
                            random_state=s * 1000 + node, max_bootstrap_per_class=c, cap_rule="adaptive")
                        est = RFOEDClassifier(base_learner_factory=fac, order="frequency", random_state=fold)
                        est.fit(X[itr], Y[itr])
                        score = evaluate(Y[iva], est.predict(X[iva]), Y[itr])["macro_f1"]
                    row = dict(score=float(score))
                except Exception as e:
                    row = dict(score=None, error="%s: %s" % (type(e).__name__, str(e)[:200]))
                row.update(dataset=ds.name, multilabel=multi, fold=fold, gamma=gamma, cap=cap, K=K,
                           n_inner_train=int(len(itr)), n_inner_val=int(len(iva)),
                           secs=round(time.perf_counter() - t0, 1))
                with OUT.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(row) + "\n")
                print("  %-18s fold %d gamma=%-4d cap=%-6d val=%s  %.0fs" % (
                    ds.name, fold, gamma, cap,
                    ("%.4f" % row["score"]) if row["score"] is not None else row["error"][:50],
                    row["secs"]), flush=True)


def report():
    rows = [json.loads(x) for x in OUT.read_text(encoding="utf-8").splitlines() if x.strip()]
    g = collections.defaultdict(list)
    for r in rows:
        if r.get("score") is not None:
            g[(r["dataset"], r["gamma"])].append(r["score"])
    datasets = sorted({d for d, _ in g})
    complete = all(len(g.get((d, gm), [])) == len(FOLDS) for d in datasets for gm in GAMMAS)
    regret = collections.defaultdict(list)
    print("%-18s " % "corpus" + "".join("gamma=%-8d" % gm for gm in GAMMAS))
    for d in datasets:
        means = {gm: float(np.mean(g[(d, gm)])) for gm in GAMMAS if g.get((d, gm))}
        best = max(means.values())
        for gm, m in means.items():
            regret[gm].append(best - m)
        print("%-18s " % d + "".join("%-14s" % ("%.4f" % means[gm] if gm in means else "--") for gm in GAMMAS))
    mean_regret = {gm: float(np.mean(v)) for gm, v in regret.items()}
    print("mean regret: " + ", ".join("gamma=%d %.4f" % (gm, mean_regret[gm]) for gm in sorted(mean_regret)))
    lo = min(mean_regret.values())
    chosen = max(gm for gm, v in mean_regret.items() if v - lo <= TIE)
    print("%s: gamma = %d" % ("SELECTED" if complete and len(datasets) == len(CORPORA) else
                              "PROVISIONAL (incomplete)", chosen))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    if a.report:
        report()
    else:
        run()
        report()


if __name__ == "__main__":
    main()
