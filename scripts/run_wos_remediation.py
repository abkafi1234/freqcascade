"""WOS46985 remediated RFOED -- the config behind the paper's Table 1c
(RFOED-RF 0.689, RFOED-NN 0.552): cap the cascade at the 20 most frequent
classes with a regularized flat tail classifier for the rest, and correct
the decision threshold per base learner (RF under-confident -> lower tau;
NN over-confident -> higher tau).

    python scripts/run_wos_remediation.py

Writes results/wos46985_remediated.jsonl (method names RFOED-RF (remediated),
RFOED-NN (remediated)) so it merges alongside the plain wos46985 results.
"""

from __future__ import annotations

import time

from _shared import (
    N_ESTIMATORS,
    NN_EPOCHS,
    NN_MEMBERS,
    RF_NJOBS,
    already_done,
    embed,
    get_folds,
    make_tfidf,
    write_results,
)

from freqcascade.datasets import load
from freqcascade.metrics import evaluate

CAP = 20
TAU_RF, TAU_NN = 0.2, 0.95


def _rf_tail():
    from sklearn.ensemble import RandomForestClassifier
    return RandomForestClassifier(
        n_estimators=N_ESTIMATORS, max_depth=20, min_samples_leaf=3,
        class_weight="balanced_subsample", n_jobs=RF_NJOBS, random_state=0,
    )


def build(base: str, seed: int):
    from freqcascade.base_learners import RFBaseLearner
    from freqcascade.decomposition import RFOEDClassifier
    from freqcascade.torch_ensemble import TorchNNEnsembleBaseLearner

    if base == "rf":
        node = lambda i: RFBaseLearner(n_estimators=N_ESTIMATORS, rebalance=True,
                                       random_state=seed * 1000 + i, n_jobs=RF_NJOBS)
        return RFOEDClassifier(base_learner_factory=node, order="frequency", random_state=seed,
                               cascade_cap=CAP, tail_learner_factory=_rf_tail, threshold=TAU_RF)
    node = lambda i: TorchNNEnsembleBaseLearner(n_members=NN_MEMBERS, rebalance=True, hidden_size=128,
                                                max_epochs=NN_EPOCHS, random_state=seed * 1000 + i)
    from sklearn.neural_network import MLPClassifier
    tail = lambda: MLPClassifier(hidden_layer_sizes=(128,), max_iter=NN_EPOCHS, early_stopping=True, random_state=0)
    return RFOEDClassifier(base_learner_factory=node, order="frequency", random_state=seed,
                           cascade_cap=CAP, tail_learner_factory=tail, threshold=TAU_NN)


def main() -> None:
    ds = load("wos46985")
    folds = get_folds(ds)
    tag = "wos46985_remediated"
    for base, label in (("rf", "RFOED-RF (remediated)"), ("nn", "RFOED-NN (remediated)")):
        if already_done(tag, label):
            print(f"skip {label} (done)"); continue
        X = make_tfidf(ds.texts, ds.texts) if base == "rf" else embed(ds.texts, "minilm", ds.name)
        rows = []
        for i, f in enumerate(folds):
            est = build(base, i)
            t0 = time.perf_counter()
            est.fit(X[f.train_idx], ds.target[f.train_idx])
            m = evaluate(ds.target[f.test_idx], est.predict(X[f.test_idx]), ds.target[f.train_idx])
            m.update(method=label, fold=i, fit_predict_s=round(time.perf_counter() - t0, 1))
            rows.append(m)
            print(f"  {label} fold {i}: macro_f1={m['macro_f1']:.3f}")
        write_results(rows, tag)


if __name__ == "__main__":
    main()
