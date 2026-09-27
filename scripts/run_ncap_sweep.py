"""n_cap sensitivity across every single-label dataset (reviewer response R3).

The paper reported an n_cap sweep on WOS46985 alone, on a post-hoc diagnostic
split. That is not enough to justify a fixed n_cap=2000 applied everywhere, so
this sweeps the cap on all five single-label corpora under the *same* protocol
as the main results -- identical committed folds for the cross-validated
corpora, the identical fixed split for CLINC150 -- so the numbers are directly
comparable with Table 3 rather than being a separate diagnostic.

n_cap enters only through balanced_bootstrap_indices (rebalance.py), which the
neural base learners call and RFBaseLearner does not: RF rebalances via
class_weight instead. The sweep is therefore over RFOED-NN, and RFOED-RF is
included at two cap settings on one dataset as a control that verifies the
cap really is inert for RF rather than merely believed to be.

    python scripts/run_ncap_sweep.py --probe          # time one cell first
    python scripts/run_ncap_sweep.py                  # full sweep, resumable
    python scripts/run_ncap_sweep.py clinc150 ohsumed_23

Writes results/ncap_sweep.jsonl (finished cells skipped on re-run).
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np

from _shared import (
    DEFAULT_ENCODER,
    NN_EPOCHS,
    NN_MEMBERS,
    RESULTS_DIR,
    embed,
    get_folds,
    make_tfidf,
)

from freqcascade.datasets import load
from freqcascade.decomposition import RFOEDClassifier
from freqcascade.metrics import evaluate

# None = uncapped (draw to the size of the larger stratum, the textbook
# balanced bootstrap). 2000 is the value the paper fixes everywhere.
NCAPS = [250, 500, 1000, 2000, 4000, None]
DATASETS = ["clinc150", "20newsgroups", "ohsumed_23", "wos46985", "drug_reviews"]
RF_CONTROL_DATASET = "wos46985"
RF_CONTROL_CAPS = [500, 4000]
OUT = RESULTS_DIR / "ncap_sweep.jsonl"


def nn_factory(seed, n_cap):
    from freqcascade.torch_ensemble import TorchNNEnsembleBaseLearner
    return lambda node: TorchNNEnsembleBaseLearner(
        n_members=NN_MEMBERS, rebalance=True, hidden_size=128, max_epochs=NN_EPOCHS,
        random_state=seed * 1000 + node, max_bootstrap_per_class=n_cap)


def rf_factory(seed, n_cap):
    from freqcascade.base_learners import RFBaseLearner
    from _shared import N_ESTIMATORS, RF_NJOBS
    # RFBaseLearner takes no cap argument: it rebalances with class_weight.
    # Accepting and ignoring n_cap here is the point of the control arm.
    return lambda node: RFBaseLearner(
        n_estimators=N_ESTIMATORS, rebalance=True,
        random_state=seed * 1000 + node, n_jobs=RF_NJOBS)


def done_cells() -> set:
    if not OUT.exists():
        return set()
    seen = set()
    for line in OUT.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            seen.add((r["dataset"], r["base"], r["n_cap"], r["fold"]))
    return seen


def iter_splits(ds):
    """Same partitions the main results use: CLINC150's fixed split repeated
    over seeds, committed 5x2 folds everywhere else."""
    if ds.split is not None:
        tr = np.flatnonzero(ds.split != "test")
        te = np.flatnonzero(ds.split == "test")
        for s in range(10):
            yield s, tr, te
    else:
        for i, f in enumerate(get_folds(ds)):
            yield i, f.train_idx, f.test_idx


def run(names, probe=False, max_folds=None):
    seen = done_cells()
    for name in names:
        ds = load(name)
        print(f"\n=== {name} === {ds.summary()}", flush=True)
        X_nn = embed(ds.texts, DEFAULT_ENCODER, ds.name)
        y = np.asarray(ds.target)
        X_rf = None

        arms = [("nn", c) for c in NCAPS]
        if name == RF_CONTROL_DATASET:
            arms += [("rf", c) for c in RF_CONTROL_CAPS]

        for base, n_cap in arms:
            for fold, tr, te in iter_splits(ds):
                if max_folds is not None and fold >= max_folds:
                    break
                if (ds.name, base, n_cap, fold) in seen:
                    continue
                if base == "rf" and X_rf is None:
                    mask = (ds.split != "test") if ds.split is not None else np.ones(ds.n_samples, bool)
                    X_rf = make_tfidf(ds.texts[mask], ds.texts)
                X = X_nn if base == "nn" else X_rf
                fac = nn_factory(fold, n_cap) if base == "nn" else rf_factory(fold, n_cap)
                est = RFOEDClassifier(base_learner_factory=fac, order="frequency", random_state=fold)
                t0 = time.perf_counter()
                est.fit(X[tr], y[tr])
                pred = est.predict(X[te])
                dt = time.perf_counter() - t0
                met = evaluate(y[te], pred, y[tr])
                met.update(dataset=ds.name, base=base, n_cap=n_cap, fold=fold,
                           fit_predict_s=round(dt, 2), n_members=NN_MEMBERS, epochs=NN_EPOCHS)
                with OUT.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(met) + "\n")
                print("  %-14s base=%s n_cap=%-5s fold=%d  macroF1=%.3f  %.1fs"
                      % (ds.name, base, n_cap, fold, met["macro_f1"], dt), flush=True)
                if probe:
                    return


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("datasets", nargs="*", default=None)
    ap.add_argument("--probe", action="store_true", help="run one cell and stop, to time it")
    ap.add_argument("--max-folds", type=int, default=None)
    a = ap.parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    run(a.datasets or DATASETS, probe=a.probe, max_folds=a.max_folds)


if __name__ == "__main__":
    main()
