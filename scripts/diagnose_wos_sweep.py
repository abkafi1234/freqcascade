"""WOS46985 remediation sweep on the fixed 80/20 diagnostic split (same
rng(0) permutation as scripts/diagnose_wos.py). Produces the numbers behind
the paper's "Remediation Sweeps and Structural Limits" section:

  * plain RFOED-RF / RFOED-NN (tau=0.5, no cap)               -> starting point
  * threshold-corrected only (tau=0.2 RF / tau=0.95 NN)       -> calibration effect
  * cap in {10,15,20,30} + threshold correction               -> the plateau
  * cap=20 + threshold + extra peel-node regularization        -> "more reg = worse"
  * two-level HierarchicalRFOEDClassifier over YL1 domains,
    with and without threshold correction                      -> structural negative result

Writes results/wos46985_sweep.jsonl (one row per config).

    python scripts/diagnose_wos_sweep.py
"""

from __future__ import annotations

import json
import time

import numpy as np

from _shared import N_ESTIMATORS, NN_EPOCHS, NN_MEMBERS, RESULTS_DIR, RF_NJOBS, embed, make_tfidf

from freqcascade.datasets import load
from freqcascade.metrics import evaluate

TAU = {"rf": 0.2, "nn": 0.95}
SEED = 0


def _yl1_groups(n: int) -> np.ndarray:
    from huggingface_hub import hf_hub_download

    p = hf_hub_download("bakirgrbic/web-of-science", "YL1.txt", repo_type="dataset")
    g = np.array([f"D{int(v)}" for v in open(p, encoding="utf-8").read().split()], dtype=object)
    assert len(g) == n, f"YL1 length {len(g)} != {n}"
    return g


def _rf_node(seed_mult=1000, min_samples_leaf=1):
    from freqcascade.base_learners import RFBaseLearner

    return lambda i: RFBaseLearner(
        n_estimators=N_ESTIMATORS, rebalance=True, random_state=SEED * seed_mult + i,
        n_jobs=RF_NJOBS, min_samples_leaf=min_samples_leaf,
    )


def _nn_node(seed_mult=1000):
    from freqcascade.torch_ensemble import TorchNNEnsembleBaseLearner

    return lambda i: TorchNNEnsembleBaseLearner(
        n_members=NN_MEMBERS, rebalance=True, hidden_size=128,
        max_epochs=NN_EPOCHS, random_state=SEED * seed_mult + i,
    )


def _rf_tail(min_samples_leaf=3):
    from sklearn.ensemble import RandomForestClassifier

    return RandomForestClassifier(
        n_estimators=N_ESTIMATORS, max_depth=20, min_samples_leaf=min_samples_leaf,
        class_weight="balanced_subsample", n_jobs=RF_NJOBS, random_state=0,
    )


def _nn_tail():
    from sklearn.neural_network import MLPClassifier

    return MLPClassifier(hidden_layer_sizes=(128,), max_iter=NN_EPOCHS,
                         early_stopping=True, random_state=0)


def _flat(base, cap, threshold, node_factory=None, tail_factory=None):
    from freqcascade.decomposition import RFOEDClassifier

    node = node_factory or (_rf_node() if base == "rf" else _nn_node())
    kw = dict(base_learner_factory=node, order="frequency", random_state=SEED, threshold=threshold)
    if cap is not None:
        kw["cascade_cap"] = cap
        kw["tail_learner_factory"] = tail_factory or (_rf_tail if base == "rf" else _nn_tail)
    return RFOEDClassifier(**kw)


def _hier(base, threshold):
    from freqcascade.decomposition import HierarchicalRFOEDClassifier

    node = _rf_node() if base == "rf" else _nn_node()
    return HierarchicalRFOEDClassifier(
        base_learner_factory=node, order="frequency", random_state=SEED, threshold=threshold,
    )


def main() -> None:
    ds = load("wos46985")
    n = ds.n_samples
    rng = np.random.default_rng(0)
    idx = rng.permutation(n)
    cut = int(0.8 * n)
    tr, te = idx[:cut], idx[cut:]
    groups = _yl1_groups(n)

    out = []
    for base in ("rf", "nn"):
        X = make_tfidf(ds.texts[tr], ds.texts) if base == "rf" else embed(ds.texts, "minilm", ds.name)
        ytr, yte = ds.target[tr], ds.target[te]

        configs = [
            ("plain", dict(cap=None, threshold=0.5)),
            ("threshold_only", dict(cap=None, threshold=TAU[base])),
            ("cap10+thr", dict(cap=10, threshold=TAU[base])),
            ("cap15+thr", dict(cap=15, threshold=TAU[base])),
            ("cap20+thr", dict(cap=20, threshold=TAU[base])),
            ("cap30+thr", dict(cap=30, threshold=TAU[base])),
        ]
        for tag, kw in configs:
            est = _flat(base, **kw)
            t0 = time.perf_counter()
            est.fit(X[tr], ytr)
            m = evaluate(yte, est.predict(X[te]), ytr)
            row = dict(base=base, config=tag, macro_f1=m["macro_f1"],
                       macro_recall=m.get("macro_recall"), secs=round(time.perf_counter() - t0, 1))
            out.append(row)
            print(f"  {base} {tag}: macro_f1={m['macro_f1']:.3f}", flush=True)

        # cap=20 + threshold + heavier peel-node regularization
        if base == "rf":
            est = _flat("rf", cap=20, threshold=TAU["rf"],
                        node_factory=_rf_node(min_samples_leaf=5),
                        tail_factory=lambda: _rf_tail(min_samples_leaf=8))
        else:
            from freqcascade.torch_ensemble import TorchNNEnsembleBaseLearner
            est = _flat("nn", cap=20, threshold=TAU["nn"],
                        node_factory=lambda i: TorchNNEnsembleBaseLearner(
                            n_members=NN_MEMBERS, rebalance=True, hidden_size=64,
                            max_epochs=NN_EPOCHS, weight_decay=1e-3, random_state=SEED * 1000 + i))
        t0 = time.perf_counter()
        est.fit(X[tr], ytr)
        m = evaluate(yte, est.predict(X[te]), ytr)
        out.append(dict(base=base, config="cap20+thr+morereg", macro_f1=m["macro_f1"],
                        macro_recall=m.get("macro_recall"), secs=round(time.perf_counter() - t0, 1)))
        print(f"  {base} cap20+thr+morereg: macro_f1={m['macro_f1']:.3f}", flush=True)

        # two-level hierarchical
        for tag, thr in (("hier_plain", 0.5), ("hier+thr", TAU[base])):
            est = _hier(base, thr)
            t0 = time.perf_counter()
            est.fit(X[tr], ytr, groups[tr])
            m = evaluate(yte, est.predict(X[te]), ytr)
            out.append(dict(base=base, config=tag, macro_f1=m["macro_f1"],
                            macro_recall=m.get("macro_recall"), secs=round(time.perf_counter() - t0, 1)))
            print(f"  {base} {tag}: macro_f1={m['macro_f1']:.3f}", flush=True)

    (RESULTS_DIR / "wos46985_sweep.jsonl").write_text(
        "\n".join(json.dumps(r) for r in out), encoding="utf-8")
    print("\n=== summary ===")
    for r in out:
        print(f"{r['base']:3s} {r['config']:20s} {r['macro_f1']:.3f}")


if __name__ == "__main__":
    main()
