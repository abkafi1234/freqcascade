"""n_cap (max_bootstrap_per_class) sensitivity on WOS46985 -- the dataset
whose shallow-node "rest"-class size motivated the cap, and the one whose
failure is diagnosed as a balanced-bootstrap prior/threshold miscalibration
(Section 7). Reviewer major comment 5: the cap was only ever verified on
CLINC150.

Runs on the SAME fixed 80/20 diagnostic split as scripts/diagnose_wos.py,
sweeping n_cap in {500, 1000, 2000, 4000, 8000, None} and reporting, per
base learner, macro-F1 plus the node-level error decomposition
(own-node-miss / early-capture / FP-first-quartile) so we can see whether
the Section 7 diagnosis is stable across the cap.

    python scripts/diagnose_ncap_wos.py

Writes results/wos46985_ncap.jsonl.
"""

from __future__ import annotations

import json
import time

import numpy as np

from _shared import N_ESTIMATORS, NN_EPOCHS, NN_MEMBERS, RESULTS_DIR, RF_NJOBS
from diagnose_wos import trace_errors

from freqcascade.datasets import load
from freqcascade.metrics import evaluate

CAPS = [500, 1000, 2000, 4000, 8000]  # uncapped omitted: exhausts 12 GB VRAM on WOS shallow nodes
SEED = 0


def build(base: str, n_cap):
    from freqcascade.base_learners import RFBaseLearner
    from freqcascade.decomposition import RFOEDClassifier
    from freqcascade.torch_ensemble import TorchNNEnsembleBaseLearner

    if base == "rf":
        # RF rebalances via class_weight, not the capped bootstrap, so it is a
        # control: n_cap should not move RF at all.
        node = lambda i: RFBaseLearner(n_estimators=N_ESTIMATORS, rebalance=True,
                                       random_state=SEED * 1000 + i, n_jobs=RF_NJOBS)
    else:
        node = lambda i: TorchNNEnsembleBaseLearner(
            n_members=NN_MEMBERS, rebalance=True, hidden_size=128, max_epochs=NN_EPOCHS,
            random_state=SEED * 1000 + i, max_bootstrap_per_class=(n_cap if n_cap is not None else 10**9),
        )
    return RFOEDClassifier(base_learner_factory=node, order="frequency", random_state=SEED)


def main() -> None:
    ds = load("wos46985")
    rng = np.random.default_rng(0)
    idx = rng.permutation(ds.n_samples)
    cut = int(0.8 * ds.n_samples)
    tr, te = idx[:cut], idx[cut:]

    from _shared import embed, make_tfidf

    out = []
    for base in ("nn", "rf"):
        X = make_tfidf(ds.texts[tr], ds.texts) if base == "rf" else embed(ds.texts, "minilm", ds.name)
        for n_cap in CAPS:
            if base == "rf" and n_cap != 2000:
                continue  # RF is cap-independent by construction; one run is enough as the control
            clf = build(base, n_cap)
            t0 = time.perf_counter()
            clf.fit(X[tr], ds.target[tr])
            m = evaluate(ds.target[te], clf.predict(X[te]), ds.target[tr])
            diag = trace_errors(clf, X[te], ds.target[te])
            row = dict(base=base, n_cap=(n_cap if n_cap is not None else "none"),
                       macro_f1=m["macro_f1"], macro_recall=m.get("macro_recall"),
                       own_node_miss=diag["own_node_miss_frac"],
                       early_capture=diag["early_capture_frac"],
                       fp_first_quartile=diag["fp_captures_first_quartile"],
                       secs=round(time.perf_counter() - t0, 1))
            out.append(row)
            (RESULTS_DIR / "wos46985_ncap.jsonl").write_text(
                "\n".join(json.dumps(r) for r in out), encoding="utf-8")
            print(f"  {base} n_cap={row['n_cap']:>5}: macro_f1={m['macro_f1']:.3f} "
                  f"own_node_miss={diag['own_node_miss_frac']:.3f} "
                  f"early_capture={diag['early_capture_frac']:.3f}", flush=True)
    print("\n=== summary ===")
    for r in out:
        print(f"{r['base']:3s} n_cap={str(r['n_cap']):>5}  macro_f1={r['macro_f1']:.3f}  "
              f"own-node-miss={r['own_node_miss']:.3f}  early-capture={r['early_capture']:.3f}  "
              f"fp-q1={r['fp_first_quartile']:.3f}")


if __name__ == "__main__":
    main()
