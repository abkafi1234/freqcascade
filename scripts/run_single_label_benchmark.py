"""Single-label benchmark: RFOED-RF / RFOED-NN against every baseline, on
CLINC150 (fixed split, 10 seeds), 20 Newsgroups, WOS46985, and Drug Reviews
and OHSUMED-23 (5x2 repeated stratified CV).

    python scripts/run_single_label_benchmark.py wos46985
    python scripts/run_single_label_benchmark.py --methods RFOED-NN,Flat-RF+SMOTE clinc150
    python scripts/run_single_label_benchmark.py --encoder pubmedbert drug_reviews

Produces results/<dataset>.jsonl (one row per method x fold). Feed those to
scripts/analyze.py for the significance tables.
"""

from __future__ import annotations

import argparse

import numpy as np

from _shared import (
    DEFAULT_ENCODER,
    SINGLE_LABEL_METHODS,
    already_done,
    embed,
    get_folds,
    make_tfidf,
    score,
    write_results,
)

from freqcascade.datasets import load

DEFAULT_DATASETS = ["clinc150", "20newsgroups", "wos46985", "drug_reviews", "ohsumed_23"]


def run_dataset(name: str, method_names: list[str], encoder: str, kind: str = "all") -> None:
    ds = load(name)
    print(f"\n=== {name} === {ds.summary()}")
    methods = SINGLE_LABEL_METHODS()
    selected = [methods[m] for m in (method_names or methods)]
    if kind != "all":
        selected = [m for m in selected if m.kind == kind]
    enc_key = None if encoder == DEFAULT_ENCODER else encoder
    tag = ds.name if encoder == DEFAULT_ENCODER else f"{ds.name}.{encoder}"
    selected = [m for m in selected if not already_done(tag, m.name, encoder=enc_key)]
    if not selected:
        print("  (all methods already complete -- skipping)")
        return

    fixed_split = ds.split if ds.split is not None else None
    folds = None if fixed_split is not None else get_folds(ds)

    # Featurize once per representation and reuse across folds/methods.
    train_mask = (ds.split != "test") if fixed_split is not None else np.ones(ds.n_samples, bool)
    feats: dict[str, object] = {}
    if any(m.kind == "rf" for m in selected):
        feats["rf"] = make_tfidf(ds.texts[train_mask], ds.texts)
    if any(m.kind == "nn" for m in selected):
        feats["nn"] = embed(ds.texts, encoder, ds.name)

    rows = []
    for m in selected:
        print(f"  {m.name} ({m.kind})...")
        try:
            rows += score(m, feats[m.kind], ds.target, folds, fixed_split)
        except Exception as e:  # a baseline that collapses is a result, not a crash
            print(f"    !! {m.name} failed: {type(e).__name__}: {e}")
            rows.append(dict(method=m.name, fold=-1, error=f"{type(e).__name__}: {e}"))
    write_results(rows, tag)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("datasets", nargs="*", default=DEFAULT_DATASETS)
    ap.add_argument("--methods", default="", help="comma-separated subset of the method registry")
    ap.add_argument("--kind", default="all", choices=("all", "rf", "nn"), help="restrict to TF-IDF (rf) or embedding (nn) methods")
    ap.add_argument("--encoder", default=DEFAULT_ENCODER, help="minilm | pubmedbert | mpnet")
    args = ap.parse_args()
    method_names = [s for s in args.methods.split(",") if s]
    for name in (args.datasets or DEFAULT_DATASETS):
        run_dataset(name, method_names, args.encoder, args.kind)


if __name__ == "__main__":
    main()
