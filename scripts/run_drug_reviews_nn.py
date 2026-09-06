"""RFOED-NN on drug_reviews (K=356) with a capped cascade -- the plain
355-node cascade crashed the GPU during nn_sl (rc 1073807364 after ~350
sequential torch fits). Capping the cascade to the top `CAP` classes with
a flat neural tail for the rest is the same remedy the paper applies to
WOS46985, and it is the sensible config at K in the hundreds anyway.

    python scripts/run_drug_reviews_nn.py

Writes RFOED-NN (capped) into results/drug_reviews.jsonl.
"""

from __future__ import annotations

import time

from _shared import NN_EPOCHS, NN_MEMBERS, already_done, embed, get_folds, write_results

from freqcascade.datasets import load
from freqcascade.metrics import evaluate

CAP = 50


def build(seed: int):
    from freqcascade.decomposition import RFOEDClassifier
    from freqcascade.torch_ensemble import TorchNNEnsembleBaseLearner
    from sklearn.neural_network import MLPClassifier

    node = lambda i: TorchNNEnsembleBaseLearner(
        n_members=NN_MEMBERS, rebalance=True, hidden_size=128,
        max_epochs=NN_EPOCHS, random_state=seed * 1000 + i,
    )
    tail = lambda: MLPClassifier(hidden_layer_sizes=(128,), max_iter=NN_EPOCHS,
                                 early_stopping=True, random_state=0)
    return RFOEDClassifier(
        base_learner_factory=node, order="frequency", random_state=seed,
        cascade_cap=CAP, tail_learner_factory=tail,
    )


def main() -> None:
    label = "RFOED-NN (capped)"
    if already_done("drug_reviews", label):
        print(f"{label} already done"); return
    ds = load("drug_reviews")
    folds = get_folds(ds)
    X = embed(ds.texts, "minilm", ds.name)
    rows = []
    for i, f in enumerate(folds):
        est = build(i)
        t0 = time.perf_counter()
        est.fit(X[f.train_idx], ds.target[f.train_idx])
        m = evaluate(ds.target[f.test_idx], est.predict(X[f.test_idx]), ds.target[f.train_idx])
        m.update(method=label, fold=i, fit_predict_s=round(time.perf_counter() - t0, 1))
        rows.append(m)
        print(f"  fold {i}: macro_f1={m['macro_f1']:.3f}")
        write_results([m], "drug_reviews")


if __name__ == "__main__":
    main()
