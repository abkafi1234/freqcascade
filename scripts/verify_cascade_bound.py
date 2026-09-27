"""Empirical check of the cascade-admissibility proposition (paper Proposition 1).

The proposition bounds the recall of the class at position j in the peel by

    1 - alpha_j - sum_{k<j} beta_{k,j}  <=  R_j  <=  min(1-alpha_j, min_k 1-beta_{k,j})

with the identity R_j = (1-alpha_j) * prod_{k<j} (1-beta_{k,j}) under conditional
independence of the node decisions given the true class. alpha_j is the own-node
miss rate of class c_j and beta_{k,j} is the rate at which an earlier node k
falsely fires on class-c_j documents.

Measuring alpha and beta needs every node evaluated on every test document, not
just on the documents still active when the cascade reaches it -- the cascade
itself short-circuits, which is exactly the effect the proposition is about. So
this script re-runs each fitted node over the whole test set to recover the full
fire matrix, then compares the three predictions against observed per-class
recall.

    python scripts/verify_cascade_bound.py --probe clinc150
    python scripts/verify_cascade_bound.py

Writes results/cascade_bound.jsonl (one row per dataset/base/class) and prints
the macro-level summary the paper quotes.
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np

from _shared import (
    DEFAULT_ENCODER,
    N_ESTIMATORS,
    NN_EPOCHS,
    NN_MEMBERS,
    RESULTS_DIR,
    RF_NJOBS,
    embed,
    make_tfidf,
)

from freqcascade.datasets import load
from freqcascade.decomposition import RFOEDClassifier

SPLIT_SEED = 20260914
DATASETS = ["clinc150", "20newsgroups", "wos46985"]
OUT = RESULTS_DIR / "cascade_bound.jsonl"


def split_of(ds):
    y = np.asarray(ds.target)
    if ds.split is not None and "val" in set(np.asarray(ds.split).tolist()):
        s = np.asarray(ds.split)
        return np.flatnonzero(s == "train"), np.flatnonzero(s == "val"), "native val split"
    from sklearn.model_selection import train_test_split

    tr, te = train_test_split(np.arange(len(y)), test_size=0.2,
                              random_state=SPLIT_SEED, stratify=y)
    return tr, te, "stratified 80/20 @ %d" % SPLIT_SEED


def fire_matrix(est, X):
    """(n_nodes, n_test) boolean: would node k fire on this document, judged
    independently of whether the cascade would ever route it there."""
    F = np.zeros((len(est.nodes_), X.shape[0]), dtype=bool)
    for i, node in enumerate(est.nodes_):
        p = node.classifier.predict_proba(X)[:, 1]
        F[i] = est._correct_p_positive(i, p) >= est.threshold
    return F


def analyse(est, X_te, y_te):
    """alpha, beta and the three recall predictions, per position in the peel."""
    order = list(est.class_order_)
    F = fire_matrix(est, X_te)
    n_nodes = F.shape[0]
    rows = []
    for j, cls in enumerate(order):
        mask = (y_te == cls)
        if not mask.any():
            continue
        Fj = F[:, mask]                       # nodes x (docs of class cls)
        if j < n_nodes:
            alpha = float(1.0 - Fj[j].mean())  # own node fails to fire
        else:
            alpha = 0.0                        # residual bucket: no own node
        betas = Fj[:j].mean(axis=1) if j > 0 else np.zeros(0)
        lower = 1.0 - alpha - float(betas.sum())
        upper = min([1.0 - alpha] + [1.0 - float(b) for b in betas]) if j > 0 else 1.0 - alpha
        indep = (1.0 - alpha) * float(np.prod(1.0 - betas)) if j > 0 else 1.0 - alpha
        rows.append(dict(position=j, cls=str(cls), n_test=int(mask.sum()),
                         alpha=round(alpha, 5),
                         beta_mean=round(float(betas.mean()), 6) if j else 0.0,
                         beta_max=round(float(betas.max()), 6) if j else 0.0,
                         beta_sum=round(float(betas.sum()), 5) if j else 0.0,
                         pred_lower=round(lower, 5), pred_upper=round(upper, 5),
                         pred_indep=round(indep, 5)))
    return rows


def run(names, bases=("nn", "rf"), probe=False, seed=0):
    from freqcascade.base_learners import RFBaseLearner
    from freqcascade.torch_ensemble import TorchNNEnsembleBaseLearner

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    for name in names:
        ds = load(name)
        tr, te, how = split_of(ds)
        y = np.asarray(ds.target)
        print("\n=== %s === K=%d (%s)" % (name, len(np.unique(y)), how), flush=True)
        for base in bases:
            if base == "nn":
                X = embed(ds.texts, DEFAULT_ENCODER, ds.name)
                fac = lambda node: TorchNNEnsembleBaseLearner(  # noqa: E731
                    n_members=NN_MEMBERS, rebalance=True, hidden_size=128,
                    max_epochs=NN_EPOCHS, random_state=seed * 1000 + node)
            else:
                mask = (ds.split != "test") if ds.split is not None else np.ones(ds.n_samples, bool)
                X = make_tfidf(ds.texts[mask], ds.texts)
                fac = lambda node: RFBaseLearner(  # noqa: E731
                    n_estimators=N_ESTIMATORS, rebalance=True,
                    random_state=seed * 1000 + node, n_jobs=RF_NJOBS)

            t0 = time.perf_counter()
            est = RFOEDClassifier(base_learner_factory=fac, order="frequency", random_state=seed)
            est.fit(X[tr], y[tr])
            pred = est.predict(X[te])
            rows = analyse(est, X[te], y[te])

            # observed per-class recall from the cascade's own predictions
            for r in rows:
                m = (y[te] == np.array(r["cls"], dtype=y.dtype)) if y.dtype.kind in "iu" \
                    else (y[te].astype(str) == r["cls"])
                r["observed"] = round(float((pred[m].astype(str) == r["cls"]).mean()), 5)
                r.update(dataset=ds.name, base=base, fold=seed, split=how)

            obs = np.array([r["observed"] for r in rows])
            lo = np.array([r["pred_lower"] for r in rows])
            up = np.array([r["pred_upper"] for r in rows])
            ind = np.array([r["pred_indep"] for r in rows])
            K = len(rows)
            a_bar = max(r["alpha"] for r in rows)
            b_bar = max(r["beta_max"] for r in rows)
            summary = dict(
                dataset=ds.name, base=base, kind="summary", K=K,
                alpha_bar=round(a_bar, 5), beta_bar=round(b_bar, 6),
                beta_mean_overall=round(float(np.mean([r["beta_mean"] for r in rows])), 6),
                macro_recall_observed=round(float(obs.mean()), 5),
                # Worst-case form of Eq. (macrobound): a_bar and b_bar are maxima
                # over classes/nodes, so a single pathological node makes it
                # vacuous. Reported for completeness.
                macro_recall_bound=round(1.0 - a_bar - 0.5 * (K - 1) * b_bar, 5),
                # Averaging the per-class lower bounds instead is equally valid
                # and far tighter -- this is the usable form.
                macro_recall_bound_mean=round(float(lo.mean()), 5),
                within_bounds_frac=round(float(((obs >= lo - 1e-9) & (obs <= up + 1e-9)).mean()), 4),
                indep_mae=round(float(np.abs(obs - ind).mean()), 5),
                indep_corr=round(float(np.corrcoef(obs, ind)[0, 1]), 4),
                fit_s=round(time.perf_counter() - t0, 1))
            with OUT.open("a", encoding="utf-8") as fh:
                for r in rows:
                    fh.write(json.dumps(r) + "\n")
                fh.write(json.dumps(summary) + "\n")
            print("  %s/%s  K=%d  alpha_bar=%.3f beta_bar=%.4f  observed macro-recall=%.3f"
                  % (name, base, K, a_bar, b_bar, summary["macro_recall_observed"]), flush=True)
            print("      within [lower,upper]: %.1f%%   independence-form MAE=%.4f  r=%.3f"
                  % (100 * summary["within_bounds_frac"], summary["indep_mae"],
                     summary["indep_corr"]), flush=True)
            if probe:
                return


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("datasets", nargs="*", default=None)
    ap.add_argument("--base", default="nn,rf")
    ap.add_argument("--probe", action="store_true")
    a = ap.parse_args()
    run(a.datasets or DATASETS, bases=tuple(a.base.split(",")), probe=a.probe)


if __name__ == "__main__":
    main()
