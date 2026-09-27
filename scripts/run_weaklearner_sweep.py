"""Is the high-K collapse an artifact of decision-stump weak learners? (round-3 R5)

EasyEnsemble and RUSBoost both default to depth-1 stumps: imbalanced-learn's
EasyEnsembleClassifier boosts AdaBoost (stumps) inside each balanced bag, and
RUSBoostClassifier boosts stumps directly. So the class-count sweep, as run
for the paper, cannot separate "class count breaks global resampling" from
"a stump cannot represent K=150 classes". This script separates them.

Design (all on the paper's CLINC150 class-count sweep, same K values and the
same random class subsets as scaling_curve.py, so the stump rows must
reproduce scaling_clinc150.jsonl -- a built-in harness check):

  weak learner   stump | tree depth 3 | tree depth 10 | unbounded tree | MLP | XGBoost
  ensemble       RUSBoost | EasyEnsemble | flat (same learner, no resampling,
                 no boosting) -- the flat arm isolates what the resampling
                 ensemble costs from what learner capacity buys
  features       TF-IDF (10k) | frozen all-MiniLM-L6-v2 (384-d)

Trees keep the canonical budget (RUSBoost 30 rounds; EasyEnsemble 50 bags of
AdaBoost with 50 rounds). MLP and XGBoost use a reduced budget (RUSBoost 10
rounds; EasyEnsemble 10 bags x 10 rounds), recorded in every row; if a reduced
ensemble collapses, --fullbudget re-runs that cell at the canonical size.

    python scripts/run_weaklearner_sweep.py --probe
    python scripts/run_weaklearner_sweep.py --learners stump,tree3,tree10,treefull
    python scripts/run_weaklearner_sweep.py --learners mlp,xgb --features emb
    python scripts/run_weaklearner_sweep.py --native wos46985 --learners treefull --folds 5

Writes results/weaklearner_sweep.jsonl (resumable per cell).
"""

from __future__ import annotations

import argparse
import json
import fcntl
import os
import time

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

from _shared import RESULTS_DIR, RF_NJOBS, embed, get_folds, make_tfidf

from freqcascade.datasets import load
from freqcascade.metrics import evaluate

K_VALUES = [5, 10, 25, 50, 100, 150]
N_SEEDS = int(os.environ.get("FREQCASCADE_SCALING_SEEDS", "3"))
LEARNERS = ["stump", "tree3", "tree10", "treefull", "mlp", "xgb"]
ENSEMBLES = ["RUSBoost", "EasyEnsemble", "flat"]
FEATURES = ["tfidf", "emb"]
OUT = RESULTS_DIR / "weaklearner_sweep.jsonl"
ATTEMPTS = RESULTS_DIR / "weaklearner_attempts.json"
LOCK = RESULTS_DIR / "weaklearner_sweep.lock"
# The rung can run as several processes at once (one per ensemble/feature slice, see
# run/run_r5_treefull.sh). Slices are disjoint, so no two processes ever compute the same
# cell, but they share the results file and the attempt counter; the counter is rewritten
# whole on every bump, so it takes an exclusive lock or one process's update is lost.
# run_cell catches Python-level failures (an aborted boosting loop, a MemoryError) and
# writes an error row, which done() then skips -- so those need no guard. A *kernel* OOM
# kill writes nothing at all, and a restart would re-enter the same cell forever and never
# reach the rest of the rung. EasyEnsemble over unbounded trees is the case this is for:
# 50 bags x 50 AdaBoost rounds on 10k-dimensional TF-IDF exceeded this machine's 15 GB at
# K=100 and left tfidf/treefull/EasyEnsemble/K=100/seed=0 with no row.
# One attempt, not two: on 2026-09-24 the first two attempts at tfidf/treefull/EasyEnsemble/K=100
# both died at the same point (~4 min, cgroup at its 6 GB cap, a 1.9 GB loky worker killed),
# so the second cost four minutes and told us nothing. Each unbounded-tree EasyEnsemble bag is
# larger than the cap, so a kill here is deterministic rather than a transient.
MAX_CELL_ATTEMPTS = 1
# Canonical budget = the paper's configuration in _shared.py: RUSBoost 30 rounds;
# EasyEnsemble 50 bags of imbalanced-learn's default inner AdaBoost (50 rounds).
CANONICAL = dict(rus_rounds=30, ee_bags=50, ee_rounds=50)
REDUCED = dict(rus_rounds=10, ee_bags=10, ee_rounds=10)


class EncodedXGB(BaseEstimator, ClassifierMixin):
    """XGBoost as an AdaBoost weak learner: label-encodes internally (XGBoost
    requires labels 0..K-1; a class subset of CLINC150 is not contiguous) and
    honours sample_weight, which SAMME relies on."""

    def __init__(self, max_depth=6, n_estimators=50, learning_rate=0.3, random_state=0):
        self.max_depth = max_depth
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.random_state = random_state

    def fit(self, X, y, sample_weight=None):
        import xgboost as xgb
        from sklearn.preprocessing import LabelEncoder

        self._le = LabelEncoder().fit(y)
        self.classes_ = self._le.classes_
        yi = self._le.transform(y)
        if len(self.classes_) == 1:
            self._const = True
            return self
        self._const = False
        obj = "binary:logistic" if len(self.classes_) == 2 else "multi:softprob"
        self._m = xgb.XGBClassifier(max_depth=self.max_depth, n_estimators=self.n_estimators,
                                    learning_rate=self.learning_rate, tree_method="hist",
                                    device="cuda", objective=obj, random_state=self.random_state,
                                    verbosity=0)
        self._m.fit(X, yi, sample_weight=sample_weight)
        return self

    def predict_proba(self, X):
        if self._const:
            return np.ones((X.shape[0], 1))
        return self._m.predict_proba(X)

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


def make_learner(name, seed):
    from sklearn.neural_network import MLPClassifier
    from sklearn.tree import DecisionTreeClassifier

    if name == "stump":
        return DecisionTreeClassifier(max_depth=1, random_state=seed)
    if name == "tree3":
        return DecisionTreeClassifier(max_depth=3, random_state=seed)
    if name == "tree10":
        return DecisionTreeClassifier(max_depth=10, random_state=seed)
    if name == "treefull":
        return DecisionTreeClassifier(max_depth=None, random_state=seed)
    if name == "mlp":
        return MLPClassifier(hidden_layer_sizes=(128,), max_iter=60, random_state=seed)
    if name == "xgb":
        return EncodedXGB(random_state=seed)
    raise ValueError(name)


def budget_for(learner, full):
    return CANONICAL if (full or learner.startswith("tree") or learner == "stump") else REDUCED


def make_estimator(ensemble, learner, seed, full):
    from imblearn.ensemble import EasyEnsembleClassifier, RUSBoostClassifier
    from sklearn.ensemble import AdaBoostClassifier

    b = budget_for(learner, full)
    parallel = 1 if learner in ("mlp", "xgb") else RF_NJOBS
    if ensemble == "flat":
        return make_learner(learner, seed), b
    if ensemble == "RUSBoost":
        return RUSBoostClassifier(estimator=make_learner(learner, seed), n_estimators=b["rus_rounds"],
                                  random_state=seed), b
    if ensemble == "EasyEnsemble":
        inner = AdaBoostClassifier(estimator=make_learner(learner, seed), n_estimators=b["ee_rounds"],
                                   random_state=seed)
        return EasyEnsembleClassifier(n_estimators=b["ee_bags"], estimator=inner,
                                      n_jobs=parallel, random_state=seed), b
    raise ValueError(ensemble)


def done():
    if not OUT.exists():
        return set()
    seen = set()
    for x in OUT.read_text(encoding="utf-8").splitlines():
        if x.strip():
            r = json.loads(x)
            seen.add((r["dataset"], r["protocol"], r["features"], r["learner"], r["ensemble"],
                      r["K"], r["seed"], r["full_budget"]))
    return seen


class _locked:
    def __enter__(self):
        LOCK.parent.mkdir(parents=True, exist_ok=True)
        self.fh = open(LOCK, "a+")
        fcntl.flock(self.fh, fcntl.LOCK_EX)

    def __exit__(self, *exc):
        fcntl.flock(self.fh, fcntl.LOCK_UN)
        self.fh.close()


def emit(row):
    with _locked(), OUT.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row) + "\n")


def _bump_attempt(key):
    """Consecutive attempts at `key`, persisted so they survive the process being killed."""
    k = "|".join(str(x) for x in key)
    with _locked():
        try:
            d = json.loads(ATTEMPTS.read_text(encoding="utf-8"))
        except Exception:
            d = {}
        d[k] = d.get(k, 0) + 1
        ATTEMPTS.parent.mkdir(parents=True, exist_ok=True)
        tmp = ATTEMPTS.with_suffix(".tmp")
        tmp.write_text(json.dumps(d, indent=0, sort_keys=True), encoding="utf-8")
        os.replace(tmp, ATTEMPTS)
    return d[k]


def run_cell(X, y, tr, te, ensemble, learner, seed, full, base_row):
    tries = _bump_attempt((base_row["dataset"], base_row["protocol"], base_row["features"],
                           learner, ensemble, base_row["K"], seed, full))
    if tries > MAX_CELL_ATTEMPTS:
        emit(dict(base_row, error="abandoned: killed by the OS on %d attempts" % (tries - 1),
                  **budget_for(learner, full)))
        print("  %-8s %-9s %-12s K=%-4s seed=%s  ABANDONED after %d kills -- error row written, "
              "continuing" % (base_row["features"], learner, ensemble, base_row["K"], seed,
                              tries - 1), flush=True)
        return
    try:
        est, b = make_estimator(ensemble, learner, seed, full)
        t0 = time.perf_counter()
        est.fit(X[tr], y[tr])
        m = evaluate(y[te], est.predict(X[te]), y[tr])
        row = dict(base_row, macro_f1=m["macro_f1"], gmean=m.get("gmean"),
                   macro_recall=m.get("macro_recall"), secs=round(time.perf_counter() - t0, 1), **b)
        msg = "macro_f1=%.3f  %.0fs" % (m["macro_f1"], row["secs"])
    except Exception as e:  # an aborted boosting loop is a result, not a crash
        row = dict(base_row, error="%s: %s" % (type(e).__name__, str(e)[:200]),
                   **budget_for(learner, full))
        msg = "ERROR " + row["error"][:70]
    emit(row)
    print("  %-8s %-9s %-12s K=%-4s seed=%s  %s" % (base_row["features"], learner, ensemble,
                                                     base_row["K"], seed, msg), flush=True)


def sweep(learners, ensembles, features, full, probe):
    ds = load("clinc150")
    y = np.asarray(ds.target)
    is_test = np.asarray(ds.split) == "test"
    classes_all = np.unique(y)
    seen = done()
    emb = None
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    only = os.environ.get("FREQCASCADE_ONLY_SEEDS")
    for seed in range(N_SEEDS):
        if only and str(seed) not in only.split(","):
            continue
        rng = np.random.default_rng(seed)
        for K in K_VALUES:
            # identical subset draw to scaling_curve.py (one rng.choice per K, in order)
            sub = set(rng.choice(classes_all, size=K, replace=False).tolist())
            mask = np.isin(y, list(sub))
            tr, te = np.flatnonzero(mask & ~is_test), np.flatnonzero(mask & is_test)
            X_tf = None
            for feat in features:
                if feat == "tfidf":
                    if X_tf is None:
                        X_tf = make_tfidf(ds.texts[tr], ds.texts)
                    X = X_tf
                else:
                    if emb is None:
                        emb = embed(ds.texts, "minilm", ds.name)
                    X = emb
                for learner in learners:
                    for ens in ensembles:
                        key = ("clinc150", "class-count sweep", feat, learner, ens, K, seed, full)
                        if key in seen:
                            continue
                        base_row = dict(dataset="clinc150", protocol="class-count sweep", features=feat,
                                        learner=learner, ensemble=ens, K=K, seed=seed, full_budget=full,
                                        n_train=int(len(tr)), n_test=int(len(te)))
                        run_cell(X, y, tr, te, ens, learner, seed, full, base_row)
                        if probe:
                            return


def native(name, learners, ensembles, features, full, n_folds):
    """Best rungs at native K on the main results' own partitions."""
    ds = load(name)
    y = np.asarray(ds.target)
    seen = done()
    if ds.split is not None:
        is_test = np.asarray(ds.split) == "test"
        parts = [(s, np.flatnonzero(~is_test), np.flatnonzero(is_test)) for s in range(n_folds)]
        proto = "native fixed split"
    else:
        parts = [(i, f.train_idx, f.test_idx) for i, f in enumerate(get_folds(ds))][:n_folds]
        proto = "native 5x2 CV folds"
    K = int(len(np.unique(y)))
    emb = None
    for seed, tr, te in parts:
        X_tf = None
        for feat in features:
            if feat == "tfidf":
                if X_tf is None:
                    mask = (ds.split != "test") if ds.split is not None else np.ones(ds.n_samples, bool)
                    X_tf = make_tfidf(ds.texts[mask], ds.texts)
                X = X_tf
            else:
                if emb is None:
                    emb = embed(ds.texts, "minilm", ds.name)
                X = emb
            for learner in learners:
                for ens in ensembles:
                    key = (ds.name, proto, feat, learner, ens, K, seed, full)
                    if key in seen:
                        continue
                    base_row = dict(dataset=ds.name, protocol=proto, features=feat, learner=learner,
                                    ensemble=ens, K=K, seed=seed, full_budget=full,
                                    n_train=int(len(tr)), n_test=int(len(te)))
                    run_cell(X, y, tr, te, ens, learner, seed, full, base_row)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--learners", default=",".join(LEARNERS))
    ap.add_argument("--ensembles", default=",".join(ENSEMBLES))
    ap.add_argument("--features", default=",".join(FEATURES))
    ap.add_argument("--fullbudget", action="store_true")
    ap.add_argument("--native", default=None, help="dataset name: run at native K on main partitions")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--probe", action="store_true")
    a = ap.parse_args()
    L, E, F = a.learners.split(","), a.ensembles.split(","), a.features.split(",")
    if a.native:
        native(a.native, L, E, F, a.fullbudget, a.folds)
    else:
        sweep(L, E, F, a.fullbudget, a.probe)


if __name__ == "__main__":
    main()
