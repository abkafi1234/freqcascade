"""Multi-label benchmark: FOCC-RF / FOCC-NN against Binary Relevance,
Balanced-BR, Classifier Chains and Ensemble of Classifier Chains, on
Reuters-21578, Hallmarks of Cancer, and LitCovid (5x2 repeated
iterative-stratified CV).

    python scripts/run_multilabel_benchmark.py reuters21578
    python scripts/run_multilabel_benchmark.py --methods FOCC-NN,Binary Relevance-RF hoc
"""

from __future__ import annotations

import argparse

import numpy as np

from _shared import (
    DEFAULT_ENCODER,
    MULTI_LABEL_METHODS,
    already_done,
    embed,
    get_folds,
    make_tfidf,
    score,
    write_results,
)

from freqcascade.datasets import load

DEFAULT_DATASETS = ["reuters21578", "hoc", "litcovid"]


def run_dataset(name: str, method_names: list[str], encoder: str, kind: str = "all") -> None:
    ds = load(name)
    print(f"\n=== {name} === {ds.summary()}")
    methods = MULTI_LABEL_METHODS()
    selected = [methods[m] for m in (method_names or methods)]
    if kind != "all":
        selected = [m for m in selected if m.kind == kind]
    enc_key = None if encoder == DEFAULT_ENCODER else encoder
    tag = ds.name if encoder == DEFAULT_ENCODER else f"{ds.name}.{encoder}"
    selected = [m for m in selected if not already_done(tag, m.name, encoder=enc_key)]
    if not selected:
        print("  (all methods already complete -- skipping)")
        return
    folds = get_folds(ds)

    feats: dict[str, object] = {}
    if any(m.kind == "rf" for m in selected):
        feats["rf"] = make_tfidf(ds.texts, ds.texts)  # multi-label: pooled, fit on all
    if any(m.kind == "nn" for m in selected):
        feats["nn"] = embed(ds.texts, encoder, ds.name)

    for m in selected:
        print(f"  {m.name} ({m.kind})...")
        try:
            rows = score(m, feats[m.kind], ds.target, folds)
        except Exception as e:
            print(f"    !! {m.name} failed: {type(e).__name__}: {e}")
            rows = [dict(method=m.name, fold=-1, error=f"{type(e).__name__}: {e}")]
        write_results(rows, tag)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("datasets", nargs="*", default=DEFAULT_DATASETS)
    ap.add_argument("--methods", default="")
    ap.add_argument("--kind", default="all", choices=("all", "rf", "nn"))
    ap.add_argument("--encoder", default=DEFAULT_ENCODER)
    args = ap.parse_args()
    method_names = [s for s in args.methods.split(",") if s]
    for name in (args.datasets or DEFAULT_DATASETS):
        run_dataset(name, method_names, args.encoder, args.kind)


if __name__ == "__main__":
    main()
