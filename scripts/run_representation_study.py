"""RQ5: does the decomposition benefit depend on the frozen encoder?

Runs the neural methods (RFOED-NN / FOCC-NN) and the strongest flat baseline
across every encoder in _shared.ENCODERS, on the medical datasets plus one
general-domain control, so "representation" enters the analysis as a factor
alongside ordering / rebalancing / base learner.

    python scripts/run_representation_study.py
    python scripts/run_representation_study.py --encoders minilm,pubmedbert --datasets drug_reviews,hoc
"""

from __future__ import annotations

import argparse

import numpy as np

from _shared import (
    ENCODERS,
    MULTI_LABEL_METHODS,
    SINGLE_LABEL_METHODS,
    already_done,
    embed,
    get_folds,
    make_tfidf,
    score,
    write_results,
)

from freqcascade.datasets import MULTI_LABEL, load

# medical first, then one well-separated general-domain control (CLINC150)
DEFAULT_DATASETS = ["drug_reviews", "ohsumed_23", "hoc", "litcovid", "clinc150"]
SL_METHODS = ["RFOED-NN", "Flat-RF+SMOTE"]      # proposed + strongest baseline
ML_METHODS = ["FOCC-NN", "Binary Relevance-RF"]


def run(dataset: str, encoders: list[str]) -> None:
    ds = load(dataset)
    multilabel = dataset in MULTI_LABEL
    registry = MULTI_LABEL_METHODS() if multilabel else SINGLE_LABEL_METHODS()
    want = ML_METHODS if multilabel else SL_METHODS
    methods = [registry[m] for m in want]

    fixed_split = ds.split if ds.split is not None else None
    folds = None if fixed_split is not None else get_folds(ds)
    train_mask = (ds.split != "test") if fixed_split is not None else np.ones(ds.n_samples, bool)

    tag = f"rq5_{ds.name}"
    for m in methods:
        combos = [("tfidf", None)] if m.kind == "rf" else [(e, e) for e in encoders]
        for enc_label, enc_arg in combos:
            if already_done(tag, m.name, encoder=enc_label):
                print(f"  skip {ds.name} / {m.name} / {enc_label} (done)")
                continue
            X = make_tfidf(ds.texts[train_mask], ds.texts) if m.kind == "rf" else embed(ds.texts, enc_arg, ds.name)
            print(f"  {ds.name} / {m.name} / {enc_label}")
            rows = [{**r, "encoder": enc_label} for r in score(m, X, ds.target, folds, fixed_split)]
            write_results(rows, tag)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default=",".join(DEFAULT_DATASETS))
    ap.add_argument("--encoders", default=",".join(ENCODERS))
    args = ap.parse_args()
    encoders = [e for e in args.encoders.split(",") if e in ENCODERS]
    for name in args.datasets.split(","):
        run(name.strip(), encoders)


if __name__ == "__main__":
    main()
