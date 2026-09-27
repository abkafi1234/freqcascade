"""Base-learner unit pilot and ensemble-size sweep (paper Table 8).

The original pilot was run once during the first study design and its inputs
were never archived: no script, no results file, and its reported values are
mutually inconsistent (the unit-comparison half and the sweep half disagree
at the same M on the same dataset). This script re-runs both arms under the
current, leakage-controlled pipeline so the table is reproducible.

Two arms, both on a single deterministic validation split per dataset so the
cross-validation test folds are never touched:

  unit   frozen-sentence-embedding + MLP ensemble  vs  trainable TextCNN,
         at M=10 members, on CLINC150 / 20 Newsgroups / WOS46985.
  sweep  the frozen-embedding unit at M in {5,10,25,50,100} on CLINC150 and
         WOS46985, recording macro-F1 and training cost.

By construction the sweep's M=10 rows and the unit arm's frozen-embedding
rows are the same computation on the same split, so they must agree -- the
internal-consistency check the original table failed.

    python scripts/run_unit_pilot.py            # both arms, resumable
    python scripts/run_unit_pilot.py unit       # one arm

Writes results/unit_pilot.jsonl (finished cells are skipped on re-run).
"""

from __future__ import annotations

import json
import sys
import time

import numpy as np

from _shared import EMB_CACHE, NN_EPOCHS, RESULTS_DIR, embed  # noqa: F401

from freqcascade.datasets import load
from freqcascade.decomposition import RFOEDClassifier
from freqcascade.metrics import evaluate

PILOT_SEED = 20260914          # split seed for datasets without a native val split
UNIT_M = 10                    # ensemble size for the unit comparison
SWEEP_M = [5, 10, 25, 50, 100]
UNIT_DATASETS = ["clinc150", "20newsgroups", "wos46985"]
SWEEP_DATASETS = ["clinc150", "wos46985"]
# The TextCNN unit caps its balanced-bootstrap draw at 1000 per class (see
# text_cnn.py); the frozen-embedding unit uses the pipeline default.
TEXTCNN_NCAP = 1000
OUT = RESULTS_DIR / "unit_pilot.jsonl"


def pilot_split(ds):
    """Deterministic validation split. Uses the dataset's own validation
    partition where one exists, otherwise a stratified 80/20 draw at
    PILOT_SEED. The cross-validation *test* partitions are never used."""
    y = np.asarray(ds.target)
    if ds.split is not None:
        s = np.asarray(ds.split)
        if "val" in set(s.tolist()):
            return np.flatnonzero(s == "train"), np.flatnonzero(s == "val"), "native val split"
    from sklearn.model_selection import train_test_split
    tr, te = train_test_split(
        np.arange(len(y)), test_size=0.2, random_state=PILOT_SEED, stratify=y
    )
    return np.sort(tr), np.sort(te), f"stratified 80/20, seed {PILOT_SEED}"


def nn_factory(seed, n_members):
    from freqcascade.torch_ensemble import TorchNNEnsembleBaseLearner
    return lambda node: TorchNNEnsembleBaseLearner(
        n_members=n_members, rebalance=True, hidden_size=128,
        max_epochs=NN_EPOCHS, random_state=seed * 1000 + node,
    )


def textcnn_factory(seed, n_members):
    from freqcascade.text_cnn import TextCNNEnsembleBaseLearner
    return lambda node: TextCNNEnsembleBaseLearner(
        n_members=n_members, rebalance=True, random_state=seed * 1000 + node,
        max_bootstrap_per_class=TEXTCNN_NCAP,
    )


def _done() -> set:
    if not OUT.exists():
        return set()
    seen = set()
    for line in OUT.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            if "macro_f1" in r:
                seen.add((r["arm"], r["dataset"], r["unit"], r["n_members"]))
    return seen


def _append(row: dict) -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row) + "\n")


def run_cell(arm, ds, unit, n_members, tr, te, split_desc, emb_cache):
    y = np.asarray(ds.target)
    if unit == "frozen_embed_mlp":
        if ds.name not in emb_cache:
            emb_cache[ds.name] = embed(ds.texts, "minilm", ds.name)
        X = emb_cache[ds.name]
        factory = nn_factory(PILOT_SEED, n_members)
    else:
        X = np.asarray(ds.texts)
        factory = textcnn_factory(PILOT_SEED, n_members)

    est = RFOEDClassifier(base_learner_factory=factory, order="frequency",
                          random_state=PILOT_SEED)
    t0 = time.perf_counter()
    est.fit(X[tr], y[tr])
    fit_s = time.perf_counter() - t0
    t1 = time.perf_counter()
    pred = est.predict(X[te])
    infer_s = time.perf_counter() - t1

    m = evaluate(y[te], pred, y[tr])
    n_nodes = len(est.nodes_)
    return dict(
        arm=arm, dataset=ds.name, unit=unit, n_members=n_members,
        macro_f1=m["macro_f1"], macro_recall=m["macro_recall"],
        gmean=m.get("gmean"), fit_s=round(fit_s, 1), infer_s=round(infer_s, 2),
        ms_per_member=round(fit_s * 1000.0 / (n_nodes * n_members), 1),
        n_nodes=n_nodes, n_classes=int(len(np.unique(y[tr]))),
        n_train=int(len(tr)), n_test=int(len(te)), split=split_desc,
        nn_epochs=NN_EPOCHS, seed=PILOT_SEED,
    )


def main(arms) -> None:
    done = _done()
    emb_cache: dict = {}
    plan = []
    if "unit" in arms:
        for name in UNIT_DATASETS:
            for unit in ("frozen_embed_mlp", "textcnn"):
                plan.append(("unit", name, unit, UNIT_M))
    if "sweep" in arms:
        for name in SWEEP_DATASETS:
            for M in SWEEP_M:
                plan.append(("sweep", name, "frozen_embed_mlp", M))

    loaded: dict = {}
    for arm, name, unit, M in plan:
        if name not in loaded:
            loaded[name] = load(name)
        ds = loaded[name]
        # key on ds.name, not the registry key: the loader may rename the
        # dataset (20newsgroups -> 20newsgroups_ir50) and the results rows
        # record ds.name, so keying on `name` never matched and re-ran the cell.
        key = (arm, ds.name, unit, M)
        if key in done:
            print(f"  skip {arm:5s} {ds.name:18s} {unit:18s} M={M:3d} (done)", flush=True)
            continue
        tr, te, split_desc = pilot_split(ds)
        try:
            row = run_cell(arm, ds, unit, M, tr, te, split_desc, emb_cache)
            print(f"  {arm:5s} {name:14s} {unit:18s} M={M:3d} "
                  f"macro_f1={row['macro_f1']:.3f} fit={row['fit_s']:.1f}s "
                  f"({row['ms_per_member']:.1f} ms/member) infer={row['infer_s']:.2f}s",
                  flush=True)
        except Exception as e:  # noqa: BLE001
            row = dict(arm=arm, dataset=name, unit=unit, n_members=M, error=str(e)[:300])
            print(f"  {arm:5s} {name:14s} {unit:18s} M={M:3d} ERROR {str(e)[:100]}", flush=True)
        _append(row)

    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    a = [x for x in sys.argv[1:] if x in ("unit", "sweep")] or ["unit", "sweep"]
    main(a)
