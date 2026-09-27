"""Replace the cross-validated single-label TF-IDF results with per-fold TF-IDF (B6).

The leak check (run_leak_check.py) re-ran every TF-IDF method in the main
results on the identical folds with the vectoriser refit inside each fold, using
the same method registry and score() path that
`run_single_label_benchmark.py --per-fold-features` uses; one cell recomputed
through the benchmark path matched its leak-check row to 1e-10. Under the rule
fixed before the leak check ran (a ranking flip requires the per-fold results),
those rows become the main results. The global-TF-IDF rows they replace are
archived, not deleted.

    python scripts/apply_perfold_tfidf.py            # dry run: report what would change
    python scripts/apply_perfold_tfidf.py --apply

Only rows keyed (method, encoder=None, fold) with a leak-check counterpart are
replaced; embedding methods, error rows and CLINC150 (fixed split, whose
vectoriser is already fit on the training partition only) are untouched.
"""

from __future__ import annotations

import argparse
import json
import shutil

from _shared import RESULTS_DIR

CORPORA = ["20newsgroups_ir50", "ohsumed_23", "wos46985", "drug_reviews"]
ARCHIVE = RESULTS_DIR / "_archive_global_tfidf"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()
    leak = {}
    for line in (RESULTS_DIR / "leak_check.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            if r.get("macro_f1") is not None:
                leak[(r["dataset"], r["method"], r["fold"])] = r
    for ds in CORPORA:
        path = RESULTS_DIR / ("%s.jsonl" % ds)
        rows = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
        out, replaced = [], 0
        for r in rows:
            k = (ds, r.get("method"), r.get("fold"))
            if r.get("encoder") is None and "error" not in r and k in leak:
                new = dict(leak[k])
                new.pop("dataset", None)
                new["features"] = "per-fold TF-IDF"
                out.append(new)
                replaced += 1
            else:
                out.append(r)
        methods = sorted({k[1] for k in leak if k[0] == ds})
        print("%-18s replace %3d of %3d rows (%d methods)%s" % (ds, replaced, len(rows), len(methods),
                                                              "" if a.apply else "  [dry run]"))
        if a.apply:
            ARCHIVE.mkdir(parents=True, exist_ok=True)
            if not (ARCHIVE / path.name).exists():
                shutil.copy2(path, ARCHIVE / path.name)
            path.write_text("".join(json.dumps(r) + "\n" for r in out), encoding="utf-8")


if __name__ == "__main__":
    main()
