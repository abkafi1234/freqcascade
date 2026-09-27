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
    RESULTS_DIR,
    already_done,
    embed,
    get_folds,
    make_tfidf,
    score,
    write_results,
)

from freqcascade.datasets import load

DEFAULT_DATASETS = ["reuters21578", "hoc", "litcovid"]


def _perfold_done(tag: str, method: str) -> bool:
    """True if results/<tag>.jsonl already holds >= 6 per-fold-TF-IDF rows for method."""
    import json
    out = RESULTS_DIR / f"{tag}.jsonl"
    if not out.exists():
        return False
    rows = (json.loads(x) for x in out.read_text(encoding="utf-8").splitlines() if x.strip())
    return sum(1 for r in rows if r.get("method") == method and r.get("features") == "per-fold TF-IDF"
               and "error" not in r) >= 6


def run_dataset(name: str, method_names: list[str], encoder: str, kind: str = "all",
                per_fold_features: bool = False) -> None:
    ds = load(name)
    print(f"\n=== {name} === {ds.summary()}")
    methods = MULTI_LABEL_METHODS()
    selected = [methods[m] for m in (method_names or methods)]
    if kind != "all":
        selected = [m for m in selected if m.kind == kind]
    enc_key = None if encoder == DEFAULT_ENCODER else encoder
    tag = ds.name if encoder == DEFAULT_ENCODER else f"{ds.name}.{encoder}"
    if per_fold_features:
        # replaces the pooled-TF-IDF rows of the TF-IDF methods; archive them first
        selected = [m for m in selected if m.kind == "rf" and not _perfold_done(tag, m.name)]
        import shutil
        arch = RESULTS_DIR / "_archive_global_tfidf"
        arch.mkdir(parents=True, exist_ok=True)
        src = RESULTS_DIR / f"{tag}.jsonl"
        if src.exists() and not (arch / src.name).exists():
            shutil.copy2(src, arch / src.name)
    else:
        selected = [m for m in selected if not already_done(tag, m.name, encoder=enc_key)]
    if not selected:
        print("  (all methods already complete -- skipping)")
        return
    folds = get_folds(ds)

    feats: dict[str, object] = {}
    if any(m.kind == "rf" for m in selected) and not per_fold_features:
        feats["rf"] = make_tfidf(ds.texts, ds.texts)  # multi-label: pooled, fit on all
    if any(m.kind == "nn" for m in selected):
        feats["nn"] = embed(ds.texts, encoder, ds.name)

    for m in selected:
        print(f"  {m.name} ({m.kind})...")
        fz = None
        if m.kind == "rf" and per_fold_features:
            def fz(tr, _ds=ds):  # refit the vectoriser on this fold's training rows only
                return make_tfidf(_ds.texts[tr], _ds.texts)
        try:
            rows = score(m, feats.get(m.kind), ds.target, folds, featurize=fz)
            if fz is not None:
                for r in rows:
                    r["features"] = "per-fold TF-IDF"
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
    ap.add_argument("--per-fold-features", action="store_true",
                    help="refit TF-IDF inside each fold and replace the pooled-TF-IDF rows (B6)")
    args = ap.parse_args()
    method_names = [s for s in args.methods.split(",") if s]
    for name in (args.datasets or DEFAULT_DATASETS):
        run_dataset(name, method_names, args.encoder, args.kind, args.per_fold_features)


if __name__ == "__main__":
    main()
