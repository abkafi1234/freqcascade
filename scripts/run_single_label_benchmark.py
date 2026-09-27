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


def run_dataset(name: str, method_names: list[str], encoder: str, kind: str = "all",
                per_fold_features: bool = False) -> None:
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

    # Featurize once per representation and reuse across folds/methods. On a
    # dataset with a fixed split that is fit on the training partition only; on
    # a cross-validated dataset it is fit over all rows, which lets test-fold
    # vocabulary and document frequencies inform TF-IDF. Pass per_fold_features
    # to refit the vectoriser inside each fold instead. Embeddings are computed
    # per document by a pretrained encoder and are unaffected either way.
    train_mask = (ds.split != "test") if fixed_split is not None else np.ones(ds.n_samples, bool)
    per_fold_tfidf = per_fold_features and fixed_split is None
    feats: dict[str, object] = {}
    if any(m.kind == "rf" for m in selected) and not per_fold_tfidf:
        feats["rf"] = make_tfidf(ds.texts[train_mask], ds.texts)
    if any(m.kind == "nn" for m in selected):
        feats["nn"] = embed(ds.texts, encoder, ds.name)

    for m in selected:
        print(f"  {m.name} ({m.kind})...")
        fz = None
        if m.kind == "rf" and per_fold_tfidf:
            def fz(tr, _ds=ds):  # noqa: F811 -- refit on this fold's training rows
                return make_tfidf(_ds.texts[tr], _ds.texts)
        try:
            rows = score(m, feats.get(m.kind), ds.target, folds, fixed_split, featurize=fz)
        except Exception as e:  # a baseline that collapses is a result, not a crash
            print(f"    !! {m.name} failed: {type(e).__name__}: {e}")
            rows = [dict(method=m.name, fold=-1, error=f"{type(e).__name__}: {e}")]
        write_results(rows, tag)  # write per method so a kill mid-dataset keeps finished cells


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("datasets", nargs="*", default=DEFAULT_DATASETS)
    ap.add_argument("--methods", default="", help="comma-separated subset of the method registry")
    ap.add_argument("--kind", default="all", choices=("all", "rf", "nn"), help="restrict to TF-IDF (rf) or embedding (nn) methods")
    ap.add_argument("--encoder", default=DEFAULT_ENCODER, help="minilm | pubmedbert | mpnet")
    ap.add_argument("--per-fold-features", action="store_true",
                    help="refit TF-IDF inside each CV fold on that fold's training rows "
                         "instead of once over all rows (no effect on fixed-split datasets "
                         "or on embedding methods)")
    args = ap.parse_args()
    method_names = [s for s in args.methods.split(",") if s]
    for name in (args.datasets or DEFAULT_DATASETS):
        run_dataset(name, method_names, args.encoder, args.kind,
                    per_fold_features=args.per_fold_features)


if __name__ == "__main__":
    main()
