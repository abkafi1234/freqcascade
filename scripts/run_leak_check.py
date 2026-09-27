"""How much does fitting TF-IDF over all rows inflate the cross-validated results?

The cross-validated path in run_single_label_benchmark.py fits the vectoriser
once over the whole corpus, so test-fold vocabulary and document frequencies
inform the representation. That is transductive leakage, not label leakage, and
it is symmetric across every method sharing the features -- but it makes the
absolute numbers optimistic, and a protocol-conscious reader is entitled to know
by how much.

This re-runs the TF-IDF methods on the identical folds with the vectoriser refit
inside each fold on that fold's training rows only (the `featurize` hook added
to _shared.score), and writes the results alongside the originals so the two can
be differenced. Embedding methods are excluded: frozen sentence embeddings are
computed per document by a pretrained encoder, with no corpus statistic fitted,
so they cannot leak this way.

    python scripts/run_leak_check.py 20newsgroups ohsumed_23
    python scripts/run_leak_check.py --compare

Writes results/leak_check.jsonl (resumable per dataset/method/fold).
"""

from __future__ import annotations

import argparse
import collections
import json
import statistics as st

import numpy as np

from _shared import (
    RESULTS_DIR,
    SINGLE_LABEL_METHODS,
    get_folds,
    make_tfidf,
    score,
)

from freqcascade.datasets import load

DATASETS = ["20newsgroups", "ohsumed_23", "wos46985", "drug_reviews"]
OUT = RESULTS_DIR / "leak_check.jsonl"


def done():
    if not OUT.exists():
        return set()
    rows = (json.loads(x) for x in OUT.read_text(encoding="utf-8").splitlines() if x.strip())
    return {(r["dataset"], r["method"]) for r in rows if "error" not in r}


def run(names):
    seen = done()
    methods = SINGLE_LABEL_METHODS()
    rf_methods = [m for m in methods.values() if m.kind == "rf"]
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    for name in names:
        ds = load(name)
        if ds.split is not None:
            print("  skipping %s: fixed split already fits TF-IDF on train only" % name)
            continue
        folds = get_folds(ds)
        y = np.asarray(ds.target)
        print("\n=== %s === %s" % (name, ds.summary()), flush=True)

        # Only methods that have a main-results counterpart on this corpus can be
        # differenced; running the others costs hours and measures nothing.
        main_methods = {json.loads(x).get("method") for x in
                        (RESULTS_DIR / ("%s.jsonl" % ds.name)).read_text(encoding="utf-8").splitlines()
                        if x.strip()} if (RESULTS_DIR / ("%s.jsonl" % ds.name)).exists() else set()
        for m in rf_methods:
            if (ds.name, m.name) in seen or m.name not in main_methods:
                continue

            def fz(tr, _ds=ds):
                return make_tfidf(_ds.texts[tr], _ds.texts)

            print("  %s ..." % m.name, flush=True)
            try:
                rows = score(m, None, y, folds, None, featurize=fz)
            except Exception as e:
                rows = [dict(method=m.name, fold=-1,
                             error="%s: %s" % (type(e).__name__, e))]
            for r in rows:
                r.update(dataset=ds.name, features="per-fold TF-IDF")
            with OUT.open("a", encoding="utf-8") as fh:
                for r in rows:
                    fh.write(json.dumps(r) + "\n")
            ok = [r["macro_f1"] for r in rows if "error" not in r]
            print("    macro-F1 %.4f" % st.mean(ok) if ok else "    failed", flush=True)


def compare():
    """Difference the per-fold run against the pooled-TF-IDF originals."""
    if not OUT.exists():
        print("no leak_check.jsonl yet")
        return
    new = collections.defaultdict(list)
    for line in OUT.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            if "error" not in r:
                new[(r["dataset"], r["method"])].append(r["macro_f1"])

    # The pooled-vectoriser originals were archived when the main TF-IDF results were
    # re-run per fold, so results/<ds>.jsonl now holds per-fold numbers and differencing
    # against it would compare the per-fold run with itself. Prefer the archive.
    old = collections.defaultdict(list)
    for ds in {k[0] for k in new}:
        arch = RESULTS_DIR / "_archive_global_tfidf" / ("%s.jsonl" % ds)
        p = arch if arch.exists() else RESULTS_DIR / ("%s.jsonl" % ds)
        if not p.exists():
            continue
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                if "error" not in r and r.get("encoder") is None:
                    old[(ds, r["method"])].append(r["macro_f1"])

    print("%-18s %-24s %8s %8s %8s" % ("dataset", "method", "pooled", "per-fold", "delta"))
    deltas = []
    for k in sorted(new):
        if k not in old:
            continue
        a, b = st.mean(old[k]), st.mean(new[k])
        deltas.append(b - a)
        print("%-18s %-24s %8.4f %8.4f %+8.4f" % (k[0], k[1], a, b, b - a))
    # Ranking flips matter more than the size of any single delta: the paper's claims are
    # comparative, so a change that preserves every within-corpus ordering is harmless.
    flips = []
    for ds in sorted({k[0] for k in new}):
        ms = [m for (d, m) in new if d == ds and (d, m) in old]
        ra = sorted(ms, key=lambda m: -st.mean(old[(ds, m)]))
        rb = sorted(ms, key=lambda m: -st.mean(new[(ds, m)]))
        if ra != rb:
            flips.append("%s: %s -> %s" % (ds, " > ".join(ra), " > ".join(rb)))
    if deltas:
        print("\nmean delta %+.4f   max |delta| %.4f   (negative = the pooled "
              "vectoriser was optimistic)" % (st.mean(deltas), max(abs(d) for d in deltas)))
        print("ranking flips: %s" % ("none" if not flips else "\n  " + "\n  ".join(flips)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("datasets", nargs="*", default=None)
    ap.add_argument("--compare", action="store_true")
    a = ap.parse_args()
    if a.compare:
        compare()
    else:
        run(a.datasets or DATASETS)


if __name__ == "__main__":
    main()
