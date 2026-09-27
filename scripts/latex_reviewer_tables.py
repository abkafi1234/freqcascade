"""LaTeX table bodies for the three new result sets, generated from the jsonl
rather than transcribed by hand (same discipline as latex_pilot_table.py, and
for the same reason: a hand-copied table is how the original Table 8 came to
disagree with itself).

    python scripts/latex_reviewer_tables.py ncap
    python scripts/latex_reviewer_tables.py finetune
    python scripts/latex_reviewer_tables.py cascade
    python scripts/latex_reviewer_tables.py all --summary
"""

from __future__ import annotations

import argparse
import collections
import json
import statistics as st

from _shared import RESULTS_DIR

B = chr(92)
PRETTY = {
    "clinc150": "CLINC150",
    "20newsgroups_ir50": "20 Newsgroups",
    "wos46985": "WOS46985",
    "ohsumed_23": "OHSUMED-23",
    "drug_reviews": "Drug Reviews",
}
NCAP_ORDER = [250, 500, 1000, 2000, 4000, None]


def load(name):
    p = RESULTS_DIR / name
    if not p.exists():
        return []
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]


def fmt(v, nd=3):
    return "--" if v is None else ("%." + str(nd) + "f") % v


# --------------------------------------------------------------------------- #

def ncap_table(summary=False):
    rows = load("ncap_sweep.jsonl")
    g = collections.defaultdict(list)
    for r in rows:
        g[(r["dataset"], r["base"], r["n_cap"])].append(r["macro_f1"])

    datasets = [d for d in ["clinc150", "20newsgroups_ir50", "ohsumed_23",
                            "wos46985", "drug_reviews"]
                if any(k[0] == d and k[1] == "nn" for k in g)]
    out = []
    for ds in datasets:
        cells = []
        for c in NCAP_ORDER:
            v = g.get((ds, "nn", c))
            cells.append(fmt(st.mean(v)) if v and len(v) >= 3 else "--")
        # relative cost of the default versus the best setting on this dataset
        vals = [(c, st.mean(g[(ds, "nn", c)])) for c in NCAP_ORDER
                if g.get((ds, "nn", c)) and len(g[(ds, "nn", c)]) >= 3]
        best = max(vals, key=lambda t: t[1]) if vals else None
        dflt = dict(vals).get(2000)
        delta = ("%+.3f" % (dflt - best[1])) if (best and dflt is not None) else "--"
        out.append("%s & %s & %s %s" % (PRETTY.get(ds, ds), " & ".join(cells), delta, B + B))
    body = "\n".join(out)
    if summary:
        print("# n_cap sweep: mean macro-F1 over folds, RFOED-NN")
        for ds in datasets:
            line = []
            for c in NCAP_ORDER:
                v = g.get((ds, "nn", c))
                line.append("%s=%s(n=%d)" % (c, fmt(st.mean(v)) if v else "--", len(v or [])))
            print("  %-16s %s" % (PRETTY.get(ds, ds), "  ".join(line)))
        rf = {k: st.mean(v) for k, v in g.items() if k[1] == "rf"}
        if rf:
            print("# RF control (cap must be inert: RFBaseLearner rebalances by class_weight)")
            for k, v in sorted(rf.items()):
                print("  %-16s n_cap=%-5s macro-F1=%.4f" % (PRETTY.get(k[0], k[0]), k[2], v))
    return body


# --------------------------------------------------------------------------- #

FT_METHODS = ["RFOED-NN", "Flat-RF", "Flat-RF+SMOTE", "Flat Balanced-RF",
              "EasyEnsemble", "RUSBoost"]


def finetune_table(summary=False):
    rows = load("finetune_study.jsonl")
    idx = {(r["dataset"], r["arm"], r["method"]): r for r in rows}
    datasets = sorted({r["dataset"] for r in rows},
                      key=lambda d: list(PRETTY).index(d) if d in PRETTY else 99)
    out = []
    for ds in datasets:
        e2e = idx.get((ds, "ft-endtoend", "FineTuned-MiniLM"))
        e2ecw = idx.get((ds, "ft-endtoend-classweighted", "FineTuned-MiniLM"))
        out.append("%s%s{%s} %s" % (B + "multicolumn{4}{l}{" + B, "textit",
                                    PRETTY.get(ds, ds), "}" + B + B))
        for m in FT_METHODS:
            fr = idx.get((ds, "frozen-features", m))
            ft = idx.get((ds, "finetuned-features", m))
            fv = fr.get("macro_f1") if fr else None
            tv = ft.get("macro_f1") if ft else None
            d = ("%+.3f" % (tv - fv)) if (fv is not None and tv is not None) else "--"
            out.append("%s & %s & %s & %s %s" % (m, fmt(fv), fmt(tv), d, B + B))
        if e2e:
            out.append("%sFine-tuned MiniLM (end-to-end)%s & -- & %s & -- %s"
                       % (B + "quad ", "", fmt(e2e.get("macro_f1")), B + B))
        if e2ecw:
            out.append("%sFine-tuned MiniLM (class-weighted)%s & -- & %s & -- %s"
                       % (B + "quad ", "", fmt(e2ecw.get("macro_f1")), B + B))
    body = "\n".join(out)
    if summary:
        print("# fine-tuning study: macro-F1 on one held-out split")
        for ds in datasets:
            print("  %s" % PRETTY.get(ds, ds))
            for arm in ["ft-endtoend", "ft-endtoend-classweighted",
                        "frozen-features", "finetuned-features"]:
                for m in ["FineTuned-MiniLM"] + FT_METHODS:
                    r = idx.get((ds, arm, m))
                    if r:
                        print("    %-26s %-18s %s" % (arm, m, fmt(r.get("macro_f1"))))
    return body


# --------------------------------------------------------------------------- #

def cascade_table(summary=False):
    rows = [r for r in load("cascade_bound.jsonl") if r.get("kind") == "summary"]
    seen, uniq = set(), []
    for r in reversed(rows):                      # keep the most recent per cell
        k = (r["dataset"], r["base"])
        if k not in seen:
            seen.add(k)
            uniq.append(r)
    uniq.reverse()
    out = []
    for r in uniq:
        admiss = 0.5 * (r["K"] - 1) * r["beta_mean_overall"]
        out.append("%s & %s & %d & %.4f & %.5f & %.3f & %s & %s & %.3f %s" % (
            PRETTY.get(r["dataset"], r["dataset"]), r["base"].upper(), r["K"],
            r["alpha_bar"], r["beta_mean_overall"], admiss,
            fmt(r["macro_recall_observed"]),
            fmt(r.get("macro_recall_bound_mean")),
            r["indep_corr"], B + B))
    body = "\n".join(out)
    if summary:
        print("# cascade-admissibility check (Proposition 1)")
        for r in uniq:
            print("  %-14s %-3s K=%-4d alpha_bar=%.3f beta_mean=%.5f  "
                  "half(K-1)beta=%.3f  observed=%.3f bound=%s  indep MAE=%.4f r=%.3f"
                  % (PRETTY.get(r["dataset"], r["dataset"]), r["base"], r["K"],
                     r["alpha_bar"], r["beta_mean_overall"],
                     0.5 * (r["K"] - 1) * r["beta_mean_overall"],
                     r["macro_recall_observed"], fmt(r.get("macro_recall_bound_mean")),
                     r["indep_mae"], r["indep_corr"]))
    return body


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("which", choices=["ncap", "finetune", "cascade", "all"])
    ap.add_argument("--summary", action="store_true")
    a = ap.parse_args()
    for name, fn in [("ncap", ncap_table), ("finetune", finetune_table),
                     ("cascade", cascade_table)]:
        if a.which in (name, "all"):
            body = fn(summary=a.summary)
            if not a.summary:
                print("%% ---- %s ----" % name)
                print(body)
                print()


if __name__ == "__main__":
    main()
