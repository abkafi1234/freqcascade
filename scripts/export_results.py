"""Export every result behind the paper into a self-describing bundle.

Copies the raw per-fold jsonl files into one folder per research question,
writes a README for each folder whose summary tables are computed from the
copied data (never transcribed), regenerates the full statistical summary, and
records provenance: repository revision, the uncommitted diff, library
versions, the committed cross-validation folds, run logs, and a SHA-256
manifest of every file.

    python scripts/export_results.py D:/Research/Imb_learn/Imb_Learn/TKDD/Result

Safe to re-run: the destination is rebuilt from the current results/ each time,
so a run still in progress (e.g. the TF-IDF leak check) is picked up on refresh.
"""

from __future__ import annotations

import collections
import datetime as dt
import hashlib
import io
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

from freqcascade import stats

REPO = Path(__file__).resolve().parent.parent
RES = REPO / "results"

PRETTY = {
    "clinc150": "CLINC150", "20newsgroups_ir50": "20 Newsgroups", "wos46985": "WOS46985",
    "ohsumed_23": "OHSUMED-23", "drug_reviews": "Drug Reviews", "reuters21578": "Reuters-21578",
    "hoc": "Hallmarks of Cancer", "litcovid": "LitCovid",
}


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #

def rows_of(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def ci(v) -> str:
    v = [x for x in v if x is not None and x == x]
    if not v:
        return "--"
    if len(v) == 1:
        return "%.3f" % v[0]
    c = stats.t_confidence_interval(np.asarray(v, dtype=float))
    return "%.3f ± %.3f" % (c.mean, c.half_width)


def msd(v) -> str:
    v = [x for x in v if x is not None and x == x]
    if not v:
        return "--"
    if len(v) == 1:
        return "%.3f" % v[0]
    return "%.3f ± %.3f" % (float(np.mean(v)), float(np.std(v, ddof=1)))


def mean(v):
    v = [x for x in v if x is not None and x == x]
    return float(np.mean(v)) if v else float("nan")


def md_table(header: list[str], body: list[list]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in body]
    return "\n".join(out)


def errors_by_method(rows, key="method"):
    e = collections.defaultdict(set)
    for r in rows:
        if "error" in r:
            e[r.get(key, "?")].add(str(r["error"]).split(":")[0])
    return e


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------------------- #
# per-section table builders (all computed from the jsonl)
# --------------------------------------------------------------------------- #

SL_METRICS = [("macro_f1", "macro-F1"), ("gmean", "G-mean"), ("mcc", "MCC"),
              ("bottom_quartile_recall", "bottom-quartile recall")]
ML_METRICS = [("label_macro_f1", "label-macro-F1"),
              ("bottom_quartile_label_recall", "bottom-quartile label recall"),
              ("example_f1", "example-F1"), ("micro_f1", "micro-F1")]


def tbl_benchmark(rows, metrics, primary):
    by = collections.defaultdict(lambda: collections.defaultdict(list))
    nfold = collections.Counter()
    for r in rows:
        if "error" in r:
            continue
        nfold[r["method"]] += 1
        for k, _ in metrics:
            if k in r:
                by[r["method"]][k].append(r[k])
    order = sorted(by, key=lambda m: -mean(by[m][primary]))
    body = [[m] + [ci(by[m][k]) for k, _ in metrics] + [nfold[m]] for m in order]
    t = md_table(["method"] + [n for _, n in metrics] + ["n"], body)
    err = errors_by_method(rows)
    if err:
        t += "\n\nDid not produce a result: " + "; ".join(
            "**%s** (%s)" % (m, ", ".join(sorted(v))) for m, v in sorted(err.items()))
    return t


def sec_main(dst):
    parts = []
    for ds in ["clinc150", "20newsgroups_ir50", "wos46985", "ohsumed_23", "drug_reviews"]:
        rows = rows_of(dst / "single_label" / (ds + ".jsonl"))
        if rows:
            parts.append("#### %s (`single_label/%s.jsonl`)\n\n%s"
                         % (PRETTY[ds], ds, tbl_benchmark(rows, SL_METRICS, "macro_f1")))
    for ds in ["reuters21578", "hoc", "litcovid"]:
        rows = rows_of(dst / "multi_label" / (ds + ".jsonl"))
        if rows:
            parts.append("#### %s (`multi_label/%s.jsonl`)\n\n%s"
                         % (PRETTY[ds], ds, tbl_benchmark(rows, ML_METRICS, "label_macro_f1")))
    return "\n\n".join(parts)


def sec_scaling(dst):
    rows = rows_of(dst / "scaling_clinc150.jsonl")
    g = collections.defaultdict(list)
    for r in rows:
        if "error" not in r:
            g[(int(r["K"]), r["method"])].append(r["macro_f1"])
    Ks = sorted({k for k, _ in g})
    methods = ["RUSBoost", "EasyEnsemble", "Flat-RF+SMOTE", "RFOED-RF", "RFOED-NN"]
    body = [["K = %d" % K] + ["%s (n=%d)" % (msd(g[(K, m)]), len(g[(K, m)])) for m in methods]
            for K in Ks]
    t = md_table(["class count"] + methods, body)
    err = [r for r in rows if "error" in r]
    if err:
        t += "\n\nAborted runs: " + "; ".join(
            "%s at K=%s, seed %s (%s)" % (r.get("method"), r.get("K"), r.get("seed"),
                                          str(r["error"]).split(":")[0]) for r in err)
    return t


def sec_ablation(dst):
    parts = []
    for ds in ["clinc150", "20newsgroups_ir50", "wos46985"]:
        rows = rows_of(dst / ("ablation_%s.jsonl" % ds))
        g = collections.defaultdict(list)
        for r in rows:
            g[(r["base_learner"], r["ordering"], r["rebalance"])].append(r["macro_f1"])
        body = [[b.upper(), o, "on" if rb else "off", msd(v), len(v)]
                for (b, o, rb), v in sorted(g.items())]
        parts.append("#### %s — single-label peel, 2×2×2 (`ablation_%s.jsonl`)\n\n%s"
                     % (PRETTY[ds], ds, md_table(["base learner", "ordering", "rebalancing",
                                                  "macro-F1 (mean ± sd)", "folds"], body)))
    for ds in ["reuters21578", "hoc", "litcovid"]:
        rows = rows_of(dst / ("ablation_%s.jsonl" % ds))
        g = collections.defaultdict(list)
        for r in rows:
            g[(r["ordering"], r["rebalance"])].append(r["label_macro_f1"])
        body = [[o, "on" if rb else "off", msd(v), len(v)] for (o, rb), v in sorted(g.items())]
        parts.append("#### %s — multi-label chain, 2×2, neural base learner (`ablation_%s.jsonl`)\n\n%s"
                     % (PRETTY[ds], ds, md_table(["ordering", "rebalancing",
                                                  "label-macro-F1 (mean ± sd)", "folds"], body)))
    return "\n\n".join(parts)


def sec_rq5(dst):
    parts = []
    enc_order = ["tfidf", "minilm", "mpnet", "pubmedbert", "pubmedbert_neuml"]
    for ds, metric in [("clinc150", "macro_f1"), ("ohsumed_23", "macro_f1"),
                       ("hoc", "label_macro_f1"), ("litcovid", "label_macro_f1")]:
        rows = rows_of(dst / ("rq5_%s.jsonl" % ds))
        g = collections.defaultdict(list)
        for r in rows:
            if "error" not in r:
                g[(r["encoder"], r["method"])].append(r[metric])
        methods = sorted({m for _, m in g})
        body = [[e] + [ci(g[(e, m)]) for m in methods] for e in enc_order if any((e, m) in g for m in methods)]
        parts.append("#### %s — %s (`rq5_%s.jsonl`)\n\n%s"
                     % (PRETTY[ds], metric.replace("_", "-"), ds,
                        md_table(["encoder"] + methods, body)))
    return "\n\n".join(parts)


def sec_diagnosis(dst):
    parts = []
    rows = rows_of(dst / "wos46985_ncap_diagnostic_split.jsonl")
    if rows:
        body = [[r["base"].upper(), r["n_cap"], r["macro_f1"], r.get("macro_recall") or "--",
                 r.get("own_node_miss"), r.get("early_capture"), r.get("fp_first_quartile") or "--"]
                for r in sorted(rows, key=lambda r: r["n_cap"])]
        parts.append("#### Cap sweep with error decomposition, fixed 80/20 diagnostic split "
                     "(`wos46985_ncap_diagnostic_split.jsonl`)\n\n" + md_table(
                         ["base", "n_cap", "macro-F1", "macro-recall", "own-node miss share",
                          "early-capture share", "FP captures in first depth quartile"], body))
    rows = rows_of(dst / "wos46985_remediation_sweep.jsonl")
    if rows:
        body = [[r["base"].upper(), r["config"], "%.3f" % r["macro_f1"],
                 "%.3f" % r["macro_recall"] if r.get("macro_recall") is not None else "--",
                 r.get("secs", "--")] for r in rows]
        parts.append("#### Remediation configurations, fixed 80/20 diagnostic split "
                     "(`wos46985_remediation_sweep.jsonl`)\n\n" + md_table(
                         ["base", "configuration", "macro-F1", "macro-recall", "seconds"], body))
    rows = rows_of(dst / "wos46985_remediated_cv.jsonl")
    if rows:
        parts.append("#### Best remediation per base learner under full 5×2 CV "
                     "(`wos46985_remediated_cv.jsonl`)\n\n" + tbl_benchmark(rows, SL_METRICS, "macro_f1"))
    rows = rows_of(dst / "perclass_clinc150.jsonl")
    if rows:
        g = collections.defaultdict(list)
        for r in rows:
            g[r["method"]].append(r)
        body = []
        for m, rs in sorted(g.items()):
            rs = sorted(rs, key=lambda r: r["train_count"])
            rare = rs[:37]
            body.append([m, len(rs), "%.3f" % mean([r["recall"] for r in rs]),
                         "%.3f" % mean([r["recall"] for r in rare]),
                         sum(1 for r in rs if r["recall"] == 0)])
        parts.append("#### CLINC150 per-class test recall (`perclass_clinc150.jsonl`)\n\n" + md_table(
            ["method", "classes", "mean recall, all classes", "mean recall, 37 rarest",
             "classes never recalled"], body))
    rows = rows_of(dst / "focc_exposure.jsonl")
    if rows:
        g = collections.defaultdict(lambda: collections.defaultdict(list))
        for r in rows:
            for k in ("label_macro_f1_predicted", "label_macro_f1_oracle", "exposure_gap"):
                g[r["dataset"]][k].append(r[k])
        body = [[PRETTY[d], msd(v["label_macro_f1_predicted"]), msd(v["label_macro_f1_oracle"]),
                 msd(v["exposure_gap"]), len(v["exposure_gap"])] for d, v in sorted(g.items())]
        parts.append("#### FOCC-NN exposure bias (`focc_exposure.jsonl`)\n\n" + md_table(
            ["corpus", "label-macro-F1, deployed", "label-macro-F1, oracle upstream labels",
             "exposure gap", "folds"], body))
    return "\n\n".join(parts)


def sec_pilot(dst):
    rows = rows_of(dst / "unit_pilot.jsonl")
    body = [[r["arm"], PRETTY.get(r["dataset"], r["dataset"]), r["unit"], r["n_members"],
             "%.3f" % r["macro_f1"], "%.1f" % r["fit_s"], "%.3f" % r["infer_s"],
             "%.2f" % r["ms_per_member"], r["split"]] for r in rows]
    return md_table(["arm", "dataset", "unit", "M", "macro-F1", "fit s", "infer s",
                     "ms/member/node", "split"], body)


def sec_ncap(dst):
    rows = rows_of(dst / "ncap_sweep.jsonl")
    caps = [250, 500, 1000, 2000, 4000, None]
    g = collections.defaultdict(list)
    for r in rows:
        g[(r["dataset"], r["base"], r["n_cap"])].append(r["macro_f1"])
    body = []
    for ds in ["clinc150", "20newsgroups_ir50", "ohsumed_23", "wos46985", "drug_reviews"]:
        cells = [msd(g[(ds, "nn", c)]) for c in caps]
        if g[(ds, "nn", 2000)] and g[(ds, "nn", None)]:
            cost = "%+.3f" % (mean(g[(ds, "nn", 2000)]) - mean(g[(ds, "nn", None)]))
        else:
            cost = "--"
        body.append([PRETTY[ds]] + cells + [cost, len(g[(ds, "nn", 2000)])])
    t = md_table(["dataset", "250", "500", "1000", "**2000 (default)**", "4000", "uncapped",
                  "cost of default", "folds"], body)
    rf = sorted((k, v) for k, v in g.items() if k[1] == "rf")
    if rf:
        t += "\n\n**Random-Forest control** (the cap should be inert because `RFBaseLearner` " \
             "rebalances through class weights and never draws the capped bootstrap):\n\n" + md_table(
                 ["dataset", "n_cap", "RFOED-RF macro-F1 (mean ± sd)", "folds"],
                 [[PRETTY[k[0]], k[2], msd(v), len(v)] for k, v in rf])
    return t


def sec_finetune(dst):
    rows = rows_of(dst / "finetune_study.jsonl")
    idx = {(r["dataset"], r["arm"], r["method"]): r for r in rows}
    methods = ["RFOED-NN", "Flat-RF", "Flat-RF+SMOTE", "Flat Balanced-RF", "EasyEnsemble", "RUSBoost"]
    parts = []
    for ds in ["clinc150", "20newsgroups_ir50", "wos46985"]:
        def val(arm, m):
            r = idx.get((ds, arm, m))
            if r is None:
                return None, "--"
            if "error" in r:
                return None, "did not fit"
            return r["macro_f1"], "%.3f" % r["macro_f1"]
        body, best = [], {}
        for arm in ("frozen-features", "finetuned-features"):
            best[arm] = max((val(arm, m)[0] for m in methods[1:] if val(arm, m)[0] is not None), default=None)
        for m in methods:
            f, fs = val("frozen-features", m)
            t, ts = val("finetuned-features", m)
            body.append([m, fs, ts, ("%+.3f" % (t - f)) if (f is not None and t is not None) else "--"])
        rf = val("frozen-features", "RFOED-NN")[0]
        rt = val("finetuned-features", "RFOED-NN")[0]
        body.append(["*RFOED-NN margin over best flat baseline*",
                     "%+.3f" % (rf - best["frozen-features"]) if rf is not None and best["frozen-features"] is not None else "--",
                     "%+.3f" % (rt - best["finetuned-features"]) if rt is not None and best["finetuned-features"] is not None else "--", ""])
        for arm, lab in [("ft-endtoend", "fine-tuned MiniLM, end-to-end, plain CE"),
                         ("ft-endtoend-classweighted", "fine-tuned MiniLM, end-to-end, class-weighted CE")]:
            r = idx.get((ds, arm, "FineTuned-MiniLM"))
            body.append(["*%s*" % lab, "", "%.3f" % r["macro_f1"] if r else "--", ""])
        split = next((r.get("split") for r in rows if r["dataset"] == ds), "?")
        parts.append("#### %s (K=%s; %s)\n\n%s" % (
            PRETTY[ds], next((r.get("K") for r in rows if r["dataset"] == ds), "?"), split,
            md_table(["method", "frozen encoder", "fine-tuned encoder", "change"], body)))
    return "\n\n".join(parts)


def sec_cascade(dst):
    rows = [r for r in rows_of(dst / "cascade_bound.jsonl") if r.get("kind") == "summary"]
    latest = {}
    for r in rows:
        latest[(r["dataset"], r["base"])] = r
    body = []
    for (ds, b), r in sorted(latest.items(), key=lambda kv: (kv[0][1], kv[1]["K"])):
        body.append([PRETTY[ds], b.upper(), r["K"], "%.3f" % r["alpha_bar"],
                     "%.5f" % r["beta_mean_overall"], "%.3f" % (0.5 * (r["K"] - 1) * r["beta_mean_overall"]),
                     "%.3f" % r["macro_recall_observed"],
                     ("%.3f" % r["macro_recall_bound_mean"]) if r.get("macro_recall_bound_mean") is not None else "--",
                     "%.1f%%" % (100 * r["within_bounds_frac"]), "%.4f" % r["indep_mae"], "%.3f" % r["indep_corr"]])
    return md_table(["dataset", "base", "K", "ᾱ (max)", "β̄ (mean)", "½(K−1)β̄",
                     "observed macro-recall", "averaged lower bound", "classes within bounds",
                     "independence-form MAE", "rank corr."], body) + \
        "\n\nPer-class rows (one per class and configuration: `position`, `cls`, `alpha`, " \
        "`beta_sum`, `pred_lower`, `pred_upper`, `pred_indep`, `observed`) are in the same file."


def bundle_root(d: Path) -> Path:
    """The export root, found from any section folder however deeply nested."""
    for a in [d, *d.parents]:
        if (a / "01_main_benchmark").is_dir():
            return a
    raise FileNotFoundError("01_main_benchmark not found above %s" % d)


def sec_protocol(dst):
    cv = rows_of(dst / "clinc150_cv.jsonl")
    fx = rows_of(bundle_root(dst) / "01_main_benchmark" / "single_label" / "clinc150.jsonl")
    g_cv, g_fx = collections.defaultdict(list), collections.defaultdict(list)
    for r in cv:
        if "error" not in r:
            g_cv[r["method"]].append(r["macro_f1"])
    for r in fx:
        if "error" not in r:
            g_fx[r["method"]].append(r["macro_f1"])
    methods = sorted(set(g_cv) | set(g_fx), key=lambda m: -mean(g_cv.get(m) or g_fx.get(m)))
    methods += [m for m in errors_by_method(cv) if m not in methods]
    body = [[m, ci(g_fx.get(m, [])), ci(g_cv.get(m, [])) if g_cv.get(m) else "did not run"] for m in methods]
    return md_table(["method", "fixed split × 10 seeds (main results)", "pooled 5×2 CV, per-fold TF-IDF"], body)


def sec_leak(dst):
    new = collections.defaultdict(list)
    for r in rows_of(dst / "leak_check.jsonl"):
        if "error" not in r:
            new[(r["dataset"], r["method"])].append(r["macro_f1"])
    body, deltas = [], []
    for (ds, m) in sorted(new):
        old = [r["macro_f1"] for r in rows_of(bundle_root(dst) / "01_main_benchmark" / "single_label" / (ds + ".jsonl"))
               if "error" not in r and r.get("method") == m]
        if not old:
            continue
        d = mean(new[(ds, m)]) - mean(old)
        deltas.append(d)
        body.append([PRETTY[ds], m, "%.4f" % mean(old), "%.4f" % mean(new[(ds, m)]), "%+.4f" % d,
                     "%d / %d" % (len(new[(ds, m)]), len(old))])
    t = md_table(["dataset", "method", "vectoriser fit on all rows (main results)",
                  "vectoriser refit per fold", "change", "folds (new / main)"], body)
    if deltas:
        t += "\n\nAcross %d cells: mean change %+.4f, largest absolute change %.4f." % (
            len(deltas), float(np.mean(deltas)), max(abs(x) for x in deltas))
    done = sorted({ds for ds, _ in new})
    t += "\n\nCorpora covered so far: %s." % ", ".join(PRETTY[d] for d in done)
    return t


def sec_spot(dst):
    rows = rows_of(dst / "spot_trees.jsonl")
    g = collections.defaultdict(list)
    for r in rows:
        g[(r["method"], r["n_estimators"])].append(r["macro_f1"])
    body = [[m, n, msd(v), len(v)] for (m, n), v in sorted(g.items())]
    return md_table(["method", "trees", "macro-F1 (mean ± sd)", "seeds"], body)


# --------------------------------------------------------------------------- #
# section registry
# --------------------------------------------------------------------------- #

SECTIONS = [
    dict(
        folder="01_main_benchmark",
        title="Main benchmark: all methods on all eight corpora",
        files={"single_label/clinc150.jsonl": "clinc150.jsonl",
               "single_label/20newsgroups_ir50.jsonl": "20newsgroups_ir50.jsonl",
               "single_label/wos46985.jsonl": "wos46985.jsonl",
               "single_label/ohsumed_23.jsonl": "ohsumed_23.jsonl",
               "single_label/drug_reviews.jsonl": "drug_reviews.jsonl",
               "multi_label/reuters21578.jsonl": "reuters21578.jsonl",
               "multi_label/hoc.jsonl": "hoc.jsonl",
               "multi_label/litcovid.jsonl": "litcovid.jsonl"},
        question="RQ1 and RQ2. Do the standard resampling remedies remain viable as the class count "
                 "grows into the hundreds, and how does local frequency-ordered decomposition (RFOED "
                 "single-label, FOCC multi-label) compare with every baseline on every corpus?",
        protocol="Single-label: CLINC150 on its provided fixed train/val/test split, repeated over 10 "
                 "seeds that vary only bootstrap draws and network initialisation; every other corpus "
                 "by repeated stratified 5×2 cross-validation (seed 20260905). Multi-label: repeated "
                 "iterative-stratified 5×2 CV (Sechidis et al., order 1, reimplemented because the "
                 "scikit-multilearn random_state is non-functional). Every method within a corpus sees "
                 "identical folds, which is what licenses the paired tests. TF-IDF (10,000 uni+bigram "
                 "features, df ≥ 2) for Random-Forest methods; frozen all-MiniLM-L6-v2 embeddings, "
                 "chunked and mean-pooled, for neural methods. 100 trees; neural ensemble M=25 members, "
                 "150 epochs, n_cap=2000; oversamplers raise minorities only to min(majority, 4×median).",
        backs="Tables `tab:mainresults`, `tab:mainresultsmed`, `tab:focc`; Figures "
              "`fig:mainresults_sl`, `fig:mainresults_ml`, `fig:collapsebars`, `fig:cd` "
              "(Friedman χ²=14.04, p=0.029, CD=4.50), `fig:margin`.",
        schema="One row per method × fold. Single-label: `macro_f1`, `macro_recall`, `gmean`, "
               "`mcc`, `bottom_quartile_recall`, `fit_predict_s`. Multi-label: `label_macro_f1`, "
               "`label_macro_recall`, `bottom_quartile_label_recall`, `example_f1`, "
               "`hamming_loss`, `subset_accuracy`, `micro_f1`, `fit_predict_s`. A row carrying "
               "`error` records a method that did not produce a result on that corpus (reported as "
               "a result, not engineered around).",
        caveats=["Drug Reviews runs a reduced method set: RFOED-NN is depth-capped "
                 "(`RFOED-NN (capped)`) and RFOED-RF is omitted for compute.",
                 "For the cross-validated corpora the TF-IDF vectoriser was fitted once over all "
                 "rows (transductive; no labels). Its effect is measured in "
                 "`07_reviewer_response/tfidf_leak_check`. Embedding methods are unaffected.",
                 "Every RFOED-NN number is capped at n_cap=2000, which understates it on four of "
                 "five corpora; see `07_reviewer_response/ncap_sensitivity`."],
        table=sec_main,
    ),
    dict(
        folder="02_class_count_sweep",
        title="Controlled class-count sweep on CLINC150",
        files={"scaling_clinc150.jsonl": "scaling_clinc150.jsonl"},
        question="RQ1. Is the resampling-ensemble collapse caused by the class count itself, or by "
                 "something else that differs between corpora?",
        protocol="CLINC150 subsampled to K ∈ {5, 10, 25, 50, 100, 150} randomly drawn classes, three "
                 "independent class subsets per K, restricted to CLINC150's fixed split. Features, "
                 "base learners and protocol identical to the main results. CLINC150's imbalance "
                 "ratio (≈4×) does not change with K, so K is the only factor varied.",
        backs="Figure `fig:scaling`; the collapse paragraphs of the results section.",
        schema="One row per method × K × subset seed: `K`, `seed`, `macro_f1`, `gmean`, `n_train`, "
               "`n_test`, `secs`.",
        caveats=["The single aborted RUSBoost boosting loop at K=50 is kept as an `error` row; the "
                 "paper's K=50 RUSBoost point averages the two subsets on which it fitted."],
        table=sec_scaling,
    ),
    dict(
        folder="03_factorial_ablation",
        title="Factorial ablation: frequency ordering × rebalancing × base learner",
        files={"ablation_clinc150.jsonl": "ablation_clinc150.jsonl",
               "ablation_20newsgroups_ir50.jsonl": "ablation_20newsgroups_ir50.jsonl",
               "ablation_wos46985.jsonl": "ablation_wos46985.jsonl",
               "ablation_reuters21578.jsonl": "ablation_reuters21578.jsonl",
               "ablation_hoc.jsonl": "ablation_hoc.jsonl",
               "ablation_litcovid.jsonl": "ablation_litcovid.jsonl"},
        question="RQ3. Which design factor carries the effect, and is it the same factor in a "
                 "single-label peel as in a multi-label chain?",
        protocol="Single-label: full 2×2×2 (ordering frequency/random × base learner RF/NN × "
                 "rebalancing on/off) on three corpora. Multi-label: 2×2 (ordering × rebalancing) with "
                 "the neural base learner on three corpora. First 6 of the 10 committed CV folds per "
                 "corpus. Analysed by three-way ANOVA, with Scheirer–Ray–Hare where residual "
                 "assumptions fail (the full test output is in `../SUMMARY.md`).",
        backs="Tables `tab:anova`, `tab:foccablation`; Figures `fig:factorial`, `fig:foccablation`.",
        schema="Single-label: one row per cell × fold with `ordering`, `base_learner`, "
               "`rebalance`, `fold`, `macro_f1`, `gmean`, `secs`. Multi-label: `ordering`, "
               "`rebalance`, `base_learner`, `fold`, `label_macro_f1`.",
        caveats=["Six folds rather than ten, to bound the 8-cell × 3-corpus compute; the ANOVA "
                 "degrees of freedom reflect that."],
        table=sec_ablation,
    ),
    dict(
        folder="04_representation_rq5",
        title="Representation study (RQ5): frozen general versus biomedical encoders",
        files={"rq5_clinc150.jsonl": "rq5_clinc150.jsonl", "rq5_ohsumed_23.jsonl": "rq5_ohsumed_23.jsonl",
               "rq5_hoc.jsonl": "rq5_hoc.jsonl", "rq5_litcovid.jsonl": "rq5_litcovid.jsonl"},
        question="RQ5. Does the picture change when the general-purpose frozen encoder is replaced by "
                 "a larger general one or by an in-domain biomedical one?",
        protocol="The neural decomposition (RFOED-NN or FOCC-NN) against its TF-IDF baseline under "
                 "four frozen encoders plus TF-IDF: all-MiniLM-L6-v2, all-mpnet-base-v2 (capacity "
                 "control), pritamdeka/S-PubMedBert-MS-MARCO (retrieval-tuned biomedical), "
                 "NeuML/pubmedbert-base-embeddings (similarity-tuned biomedical). Identical mean "
                 "pooling. The TF-IDF Random-Forest baseline is run once (it has no encoder); the neural "
                 "decomposition is run under each frozen encoder. All 10 committed CV folds.",
        backs="Table `tab:rq5`; Figure `fig:representation`.",
        schema="One row per encoder × method × fold: `encoder`, `method`, `fold`, and the single- or "
               "multi-label metric set of section 01.",
        caveats=["All four encoders are frozen. Fine-tuning is tested separately in "
                 "`07_reviewer_response/finetuning`, and it changes the conclusion about the "
                 "decomposition's advantage."],
        table=sec_rq5,
    ),
    dict(
        folder="05_failure_diagnosis",
        title="Failure modes and boundary conditions",
        files={"wos46985_ncap_diagnostic_split.jsonl": "wos46985_ncap.jsonl",
               "wos46985_remediation_sweep.jsonl": "wos46985_sweep.jsonl",
               "wos46985_remediated_cv.jsonl": "wos46985_remediated.jsonl",
               "perclass_clinc150.jsonl": "perclass_clinc150.jsonl",
               "focc_exposure.jsonl": "focc_exposure.jsonl"},
        question="RQ4. Why does local decomposition underperform on WOS46985, can the failure be "
                 "remediated, how does the rare tail fare, and how much does FOCC lose to exposure "
                 "bias at deployment?",
        protocol="WOS46985 diagnosis and remediation sweep on one fixed stratified 80/20 split so that "
                 "the cross-validation test folds are untouched; the best remediation per base "
                 "learner is then re-run under full 5×2 CV. Per-class CLINC150 recall from the main "
                 "fixed split. Exposure bias: FOCC-NN evaluated with each link conditioned on earlier "
                 "links' predictions (deployed) versus their true labels (oracle), all 10 folds.",
        backs="Tables `tab:diagnosis`, `tab:remediation`; Figures `fig:failuremech`, "
              "`fig:remediation`, `fig:perclass`, `fig:exposure`.",
        schema="`wos46985_ncap_diagnostic_split`: `base`, `n_cap`, `macro_f1`, `own_node_miss`, "
               "`early_capture`, `fp_first_quartile`, `note`. `wos46985_remediation_sweep`: `base`, "
               "`config` (cap = cascade depth cap, thr = decision-threshold correction, hier = "
               "two-level hierarchy, morereg = stronger regularisation), `macro_f1`, `macro_recall`, "
               "`secs`; `capN` = only the N most frequent classes get a peel node and a flat tail classifier "
               "handles the rest, `thr` = decision-threshold correction, `hier` = two-level coarse-to-fine "
               "cascade, `morereg` = heavier node regularisation (RF min_samples_leaf 5 / tail 8; NN hidden "
               "64, weight decay 1e-3). `wos46985_remediated_cv`: section 01 schema. `perclass_clinc150`: `method`, "
               "`class`, `recall`, `train_count`. `focc_exposure`: `dataset`, `fold`, "
               "`label_macro_f1_predicted`, `label_macro_f1_oracle`, `exposure_gap`.",
        caveats=["`wos46985_ncap_diagnostic_split.jsonl` is the original single-split cap sweep "
                 "(values transcribed from the 2026-09-08 run log, see `note`). It is superseded as "
                 "a sensitivity analysis by the full cross-validated sweep in "
                 "`07_reviewer_response/ncap_sensitivity`, but remains the source of the error-mode "
                 "decomposition (early capture 89–98% at every cap).",
                 "The node-level error-type decomposition in `tab:diagnosis` (own-node miss 98.7% for "
                 "RFOED-RF, early capture 95.4% for RFOED-NN, 79.7% of false captures in the first depth "
                 "quartile) was produced by `diagnose_wos.py` and recorded in a report rather than a jsonl; "
                 "that report is copied here as `node_level_diagnosis_source.md`."],
        extra_copy={"node_level_diagnosis_source.md": "Imb_Learn/revision/RESULTS-REPORT.md"},
        table=sec_diagnosis,
    ),
    dict(
        folder="06_base_learner_pilot",
        title="Base-learner unit pilot and ensemble-size sweep",
        files={"unit_pilot.jsonl": "unit_pilot.jsonl"},
        question="Design stage. Frozen embedding + MLP ensemble versus a trainable TextCNN as the "
                 "neural base learner, and how macro-F1 and cost respond to ensemble size M.",
        protocol="One deterministic validation split per corpus (CLINC150's native val partition; "
                 "stratified 80/20 at seed 20260914 elsewhere); CV test folds untouched. Unit arm at "
                 "M=10 on three corpora; sweep arm at M ∈ {5, 10, 25, 50, 100} on CLINC150 and "
                 "WOS46985. The unit arm's frozen M=10 cells and the sweep's M=10 cells are the same "
                 "computation, reported once, so the two halves cannot disagree.",
        backs="Table `tab:pilot`.",
        schema="One row per arm × dataset × unit × M: `macro_f1`, `macro_recall`, `gmean`, `fit_s`, "
               "`infer_s`, `ms_per_member`, `n_nodes`, `n_classes`, `n_train`, `n_test`, `split`, "
               "`nn_epochs`, `seed`.",
        caveats=["Re-run on 2026-09-14 because the original design-stage table had no backing data "
                 "and contradicted itself; see `origin_README.md`."],
        extra_copy={"origin_README.md": "Imb_Learn/Result/unit-pilot-2026-09-14/README.md"},
        table=sec_pilot,
    ),
    dict(
        folder="07_reviewer_response/ncap_sensitivity",
        title="n_cap sensitivity on all five single-label corpora",
        files={"ncap_sweep.jsonl": "ncap_sweep.jsonl"},
        question="Is the single cross-dataset default n_cap=2000 justified, or does it distort the "
                 "results?",
        protocol="RFOED-NN at n_cap ∈ {250, 500, 1000, 2000, 4000, uncapped} on the main results' own "
                 "folds (CLINC150's fixed split × 10 seeds; committed 5×2 folds elsewhere). "
                 "RFOED-RF run at two caps on WOS46985 as a control on the claim that the cap does not "
                 "reach the Random-Forest base learner.",
        backs="Table `tab:ncap` and the rebalancing-cap passage of the method section.",
        schema="One row per dataset × base × n_cap × fold: `n_cap` (null = uncapped), `base`, "
               "`macro_f1`, `macro_recall`, `gmean`, `mcc`, `bottom_quartile_recall`, "
               "`fit_predict_s`, `n_members`, `epochs`.",
        caveats=["The sweep reproduces the main table at the default (WOS46985 RFOED-NN 0.331 vs "
                 "0.3308; RFOED-RF 0.3277 vs 0.3277), which validates the harness against the "
                 "original pipeline."],
        table=sec_ncap,
    ),
    dict(
        folder="07_reviewer_response/finetuning",
        title="End-to-end fine-tuning versus frozen representations",
        files={"finetune_study.jsonl": "finetune_study.jsonl"},
        question="Are the findings an artifact of frozen feature vectors? Does the collapse survive a "
                 "dynamic representation, and does the decomposition's advantage?",
        protocol="all-MiniLM-L6-v2 fine-tuned end-to-end on the K-class task (12 epochs, lr 5e-5, "
                 "OneCycle, batch 32, fp16; max length 64 for CLINC150, 256 otherwise), with plain and "
                 "class-weighted cross-entropy. The plain fine-tuned encoder is then frozen and its "
                 "mean-pooled hidden states fed to the same method set as the frozen-encoder arm. One "
                 "held-out split per corpus (CLINC150 native val; stratified 80/20 at seed 20260914 "
                 "elsewhere).",
        backs="Table `tab:finetune`; Section `sec:finetune`; the qualified contribution 2, "
              "Assumption A2 and the first gate of the decision procedure.",
        schema="One row per dataset × arm × method: `arm` ∈ {`ft-endtoend`, "
               "`ft-endtoend-classweighted`, `frozen-features`, `finetuned-features`}, `macro_f1` and "
               "the section 01 metrics, `fit_predict_s`, `K`, `split`, and for fine-tuning rows "
               "`max_len`, `epochs`.",
        caveats=["A first pass at 4 epochs / lr 3e-5 was discarded as unconverged (training loss "
                 "3.06, CLINC150 macro-F1 0.667, below the frozen encoder). Its log is kept in "
                 "`99_superseded_do_not_cite/logs/` and its rows are not in this file.",
                 "Single split per corpus, so the fine-tuned-feature margins of 0.02–0.07 are "
                 "consistent in sign across all three corpora but not tested at fold resolution."],
        table=sec_finetune,
    ),
    dict(
        folder="07_reviewer_response/cascade_bound",
        title="Empirical check of the cascade-admissibility proposition",
        files={"cascade_bound.jsonl": "cascade_bound.jsonl"},
        question="Does the proved bound 1 − α_j − Σ_{k<j} β_kj ≤ R_j ≤ min(1 − α_j, min_k(1 − β_kj)) "
                 "describe the cascade's real per-class recall, and does the admissibility quantity "
                 "½(K−1)β̄ separate the corpora where decomposition works from where it fails?",
        protocol="RFOED fitted on one held-out split per corpus; every node then re-evaluated on the "
                 "whole test set (not only the documents still active when the cascade reaches it) "
                 "to recover the full node-fire matrix, from which α_j (own-node miss) and β_kj "
                 "(false capture of class j by earlier node k) are measured and compared with "
                 "observed per-class recall.",
        backs="Table `tab:cascadecheck`; Proposition `prop:cascade`; Section `sec:cascadecheck`.",
        schema="Per-class rows: `dataset`, `base`, `position`, `cls`, `n_test`, `alpha`, `beta_mean`, "
               "`beta_max`, `beta_sum`, `pred_lower`, `pred_upper`, `pred_indep`, `observed`. Summary "
               "rows (`kind`=`summary`): `K`, `alpha_bar`, `beta_bar`, `beta_mean_overall`, "
               "`macro_recall_observed`, `macro_recall_bound`, `macro_recall_bound_mean`, "
               "`within_bounds_frac`, `indep_mae`, `indep_corr`, `fit_s`.",
        caveats=["The file contains an earlier probe summary for 20 Newsgroups/RF written before "
                 "`macro_recall_bound_mean` was added; the table uses the latest summary per "
                 "configuration.",
                 "α and β are estimated on the same partition recall is compared on, so this shows "
                 "the proposition describes the cascade correctly, not that β̄ transfers across "
                 "corpora.",
                 "The worst-case form (β̄ as a maximum) is vacuous on every configuration; the "
                 "averaged form is the usable one."],
        table=sec_cascade,
    ),
    dict(
        folder="07_reviewer_response/protocol_check",
        title="CLINC150 under pooled 5×2 cross-validation",
        files={"clinc150_cv.jsonl": "clinc150_cv.jsonl"},
        question="Does any CLINC150 conclusion depend on evaluating it on a fixed split while every "
                 "other corpus is cross-validated?",
        protocol="The full single-label method set on CLINC150 with the provided split discarded: "
                 "pooled repeated stratified 5×2 CV at the paper's seed, with TF-IDF refitted inside "
                 "each fold on that fold's training rows (stricter than the main pipeline).",
        backs="Table `tab:protocolcheck` and the aggregation paragraph of the statistical protocol.",
        schema="Section 01 single-label schema plus `dataset`=`clinc150_cv` and `protocol`.",
        caveats=["ADASYN raises its ValueError on all ten pooled folds (it runs on the fixed split): "
                 "configuration sensitivity, recorded as `error` rows."],
        table=sec_protocol,
    ),
    dict(
        folder="07_reviewer_response/tfidf_leak_check",
        title="TF-IDF transductive-fit check",
        files={"leak_check.jsonl": "leak_check.jsonl"},
        question="How much did fitting the TF-IDF vectoriser over all rows (including test folds) "
                 "inflate the cross-validated results?",
        protocol="Every TF-IDF method re-run on the identical committed folds with the vectoriser "
                 "refitted inside each fold on that fold's training rows only (the `featurize` hook "
                 "in `_shared.score`, exposed as `run_single_label_benchmark.py "
                 "--per-fold-features`), then differenced against the main results. Embedding "
                 "methods are excluded because no corpus statistic is fitted for them.",
        backs="Not yet in the paper; to be written once all corpora complete.",
        schema="Section 01 single-label schema plus `dataset` and `features`=`per-fold TF-IDF`.",
        caveats=["**Run in progress at export time.** Re-run `scripts/export_results.py` to refresh "
                 "this folder when it finishes."],
        table=sec_leak,
    ),
    dict(
        folder="08_implementation_checks",
        title="Implementation spot checks",
        files={"spot_trees.jsonl": "spot_trees.jsonl"},
        question="Does reducing the Random Forest from 200 to 100 trees (to halve campaign wall-clock) "
                 "shift macro-F1 materially or non-uniformly?",
        protocol="Flat-RF+SMOTE and RFOED-RF on CLINC150 at 100 and 200 trees, five seeds each.",
        backs="The 100-tree configuration choice in the experimental setup.",
        schema="One row per method × trees × seed: `n_estimators`, `method`, `seed`, `macro_f1`, `secs`.",
        caveats=[],
        table=sec_spot,
    ),
]


# --------------------------------------------------------------------------- #
# build
# --------------------------------------------------------------------------- #

def git(*args):
    try:
        return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True,
                              encoding="utf-8").stdout
    except Exception as e:  # pragma: no cover
        return "git unavailable: %s" % e


def write(p: Path, text: str):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8", newline="\n")


def build(dst: Path):
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)
    imb = REPO.parent / "Imb_Learn"
    now = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    index_rows = []

    for sec in SECTIONS:
        d = dst / sec["folder"]
        d.mkdir(parents=True, exist_ok=True)
        counts = []
        for rel, src in sec["files"].items():
            s = RES / src
            if not s.exists():
                counts.append("`%s`: **missing**" % rel)
                continue
            (d / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(s, d / rel)
            n = len(rows_of(s))
            counts.append("`%s` (%d rows)" % (rel, n))
        for rel, src in sec.get("extra_copy", {}).items():
            s = REPO.parent / src
            if s.exists():
                shutil.copy2(s, d / rel)
        text = ["# %s" % sec["title"], "",
                "**Question.** " + sec["question"], "",
                "**Protocol.** " + sec["protocol"], "",
                "**Backs in the paper.** " + sec["backs"], "",
                "**Files.** " + "; ".join(counts), "",
                "**Schema.** " + sec["schema"], "",
                "## Summary (computed from the files in this folder)", "",
                sec["table"](d), ""]
        if sec["caveats"]:
            text += ["## Caveats", ""] + ["- " + c for c in sec["caveats"]] + [""]
        write(d / "README.md", "\n".join(text))
        index_rows.append(["[`%s/`](%s/README.md)" % (sec["folder"], sec["folder"]),
                           sec["title"], sec["backs"].split(";")[0]])
        print("  %-42s %s" % (sec["folder"], ", ".join(c.split(" (")[0] for c in counts)))

    # full statistical summary, regenerated from the current results
    summ = subprocess.run([sys.executable, str(REPO / "scripts" / "summarize.py")],
                          capture_output=True, text=True, encoding="utf-8", cwd=str(REPO))
    write(dst / "SUMMARY.md", "<!-- regenerated %s by scripts/summarize.py: mean ± 95%% CI, "
          "Holm-adjusted paired Wilcoxon vs the reference method, factorial ANOVA/SRH, RQ5 -->\n\n%s"
          % (now, summ.stdout))

    # committed folds
    shutil.copytree(REPO / "data" / "folds", dst / "folds")
    write(dst / "folds" / "README.md",
          "# Committed cross-validation folds\n\nOne `.npz` per corpus, written by "
          "`freqcascade.cv.save_folds` from seed 20260905 (5 repeats × 2 splits; iterative "
          "stratification for the multi-label corpora). Load with `freqcascade.cv.load_folds(path)`, "
          "which returns a list of `FoldSplit(train_idx, test_idx)`. Every method in the main "
          "results, the ablation, RQ5, the n_cap sweep and the leak check consumed exactly these "
          "indices. CLINC150's file is kept for completeness; its main results use the provided "
          "fixed split instead.\n")

    # logs
    logs = {
        "ncap_sweep.log": "ncap_main2.log",
        "finetune_clinc150.log": "ft_clinc2.log",
        "finetune_20newsgroups_wos46985.log": "ft_rest.log",
        "cascade_bound_nn.log": "cascade_nn.log",
        "cascade_bound_rf.log": "cascade_rf.log",
        "clinc150_cv.log": "clinc_cv.log",
        "tfidf_leak_check.log": "leak.log",
    }
    for dstname, src in logs.items():
        if (REPO / src).exists():
            (dst / "logs").mkdir(exist_ok=True)
            shutil.copy2(REPO / src, dst / "logs" / dstname)

    # provenance
    prov = dst / "provenance"
    prov.mkdir()
    write(prov / "git.txt", "repository: %s\nbranch: %s\ncommit: %s\n\nuncommitted changes "
          "(git status --short):\n%s" % (REPO, git("branch", "--show-current").strip(),
                                          git("rev-parse", "HEAD").strip(), git("status", "--short")))
    write(prov / "uncommitted.patch", git("diff"))
    import importlib
    vers = ["python %s (%s)" % (platform.python_version(), platform.platform())]
    for mod in ("numpy", "scipy", "sklearn", "imblearn", "torch", "transformers",
                "sentence_transformers", "freqcascade"):
        try:
            vers.append("%s %s" % (mod, getattr(importlib.import_module(mod), "__version__", "?")))
        except Exception:
            vers.append("%s not importable" % mod)
    try:
        import torch
        if torch.cuda.is_available():
            vers.append("GPU %s, %.1f GB" % (torch.cuda.get_device_name(0),
                                             torch.cuda.get_device_properties(0).total_memory / 1e9))
    except Exception:
        pass
    write(prov / "environment.txt", "\n".join(vers) + "\n")
    scripts_dst = prov / "scripts"
    shutil.copytree(REPO / "scripts", scripts_dst, ignore=shutil.ignore_patterns("__pycache__"))
    rr = imb / "Result" / "reviewer-response-2026-09-16" / "README.md"
    if rr.exists():
        shutil.copy2(rr, dst / "07_reviewer_response" / "origin_README.md")

    # superseded material, quarantined
    sup = dst / "99_superseded_do_not_cite"
    sup.mkdir()
    shutil.copytree(RES / "_pre_config_fix", sup / "pre_config_fix")
    for dstname, src in {"finetune_clinc150_unconverged_4epochs.log": "ft_clinc.log",
                         "ncap_sweep_interrupted_run1.log": "ncap_cheap.log",
                         "ncap_sweep_interrupted_run2.log": "ncap_main.log"}.items():
        if (REPO / src).exists():
            (sup / "logs").mkdir(exist_ok=True)
            shutil.copy2(REPO / src, sup / "logs" / dstname)
    lines = ["# Superseded material — do not cite", "",
             "Kept for audit only. None of these numbers appear in the paper.", "",
             "## `pre_config_fix/`", "",
             "Main-benchmark results from an earlier run that was superseded when the campaign "
             "configuration was corrected and the corpora re-run (folder `_pre_config_fix`, created "
             "2026-09-06). Per-method macro-F1 (or label-macro-F1), superseded versus current:", ""]
    for f in sorted((RES / "_pre_config_fix").glob("*.jsonl")):
        old, cur = rows_of(f), rows_of(RES / f.name)
        key = "label_macro_f1" if any("label_macro_f1" in r for r in old) else "macro_f1"
        go, gc = collections.defaultdict(list), collections.defaultdict(list)
        for r in old:
            if "error" not in r and key in r:
                go[r["method"]].append(r[key])
        for r in cur:
            if "error" not in r and key in r and r.get("encoder") is None:
                gc[r["method"]].append(r[key])
        body = [[m, "%.3f (n=%d)" % (mean(go[m]), len(go[m])),
                 ("%.3f (n=%d)" % (mean(gc[m]), len(gc[m]))) if gc.get(m) else "--"]
                for m in sorted(go, key=lambda m: -mean(go[m]))]
        lines += ["### %s" % PRETTY.get(f.stem, f.stem), "",
                  md_table(["method", "superseded", "current"], body), ""]
    lines += ["## `logs/`", "",
              "- `finetune_clinc150_unconverged_4epochs.log` — the discarded first fine-tuning pass "
              "(4 epochs, lr 3e-5; loss 3.06, macro-F1 0.667). Not converged; replaced by the "
              "12-epoch run.",
              "- `ncap_sweep_interrupted_run*.log` — n_cap sweep launches stopped for reordering and "
              "for the `max_bootstrap_per_class=None` TypeError fix. The sweep is resumable, so their "
              "completed cells are part of the final `ncap_sweep.jsonl`; the logs are kept only to "
              "document the crash.", ""]
    write(sup / "README.md", "\n".join(lines))

    # master README
    write(dst / "README.md", master_readme(dst, index_rows, now))

    # manifest last, so it covers everything
    man = ["# SHA-256 manifest", "", "| file | rows | bytes | sha256 |", "|---|---|---|---|"]
    for p in sorted(dst.rglob("*")):
        if p.is_file() and p.name != "MANIFEST.md":
            rel = p.relative_to(dst).as_posix()
            n = len(rows_of(p)) if p.suffix == ".jsonl" else ""
            man.append("| `%s` | %s | %d | `%s` |" % (rel, n, p.stat().st_size, sha256(p)))
    write(dst / "MANIFEST.md", "\n".join(man) + "\n")
    print("wrote %s" % dst)


def headline(dst: Path) -> str:
    """Headline numbers, pulled from the exported files rather than typed."""
    def sl(ds, m, key="macro_f1"):
        return mean([r[key] for r in rows_of(dst / "01_main_benchmark" / "single_label" / (ds + ".jsonl"))
                     if "error" not in r and r.get("method") == m])

    ncap = collections.defaultdict(list)
    for r in rows_of(dst / "07_reviewer_response" / "ncap_sensitivity" / "ncap_sweep.jsonl"):
        ncap[(r["dataset"], r["base"], r["n_cap"])].append(r["macro_f1"])
    ft = {(r["dataset"], r["arm"], r["method"]): r.get("macro_f1")
          for r in rows_of(dst / "07_reviewer_response" / "finetuning" / "finetune_study.jsonl")}
    casc = {}
    for r in rows_of(dst / "07_reviewer_response" / "cascade_bound" / "cascade_bound.jsonl"):
        if r.get("kind") == "summary":
            casc[(r["dataset"], r["base"])] = r
    out = []
    out.append("1. **The resampling ensembles collapse at high class count.** EasyEnsemble macro-F1 "
               "%.3f on CLINC150 (K=150), %.3f on WOS46985 (K=134), %.3f on Drug Reviews (K≈356); "
               "RUSBoost %.3f and %.3f, and it fails to fit on 20 Newsgroups and OHSUMED-23. "
               "Section 01." % (sl("clinc150", "EasyEnsemble"), sl("wos46985", "EasyEnsemble"),
                                sl("drug_reviews", "EasyEnsemble"), sl("clinc150", "RUSBoost"),
                                sl("wos46985", "RUSBoost")))
    sc = collections.defaultdict(list)
    for r in rows_of(dst / "02_class_count_sweep" / "scaling_clinc150.jsonl"):
        if "error" not in r:
            sc[(int(r["K"]), r["method"])].append(r["macro_f1"])
    out.append("2. **The collapse is caused by class count itself.** On CLINC150 subsets at fixed "
               "imbalance ratio, EasyEnsemble falls from %.2f at K=5 to %.3f at K=150 and RUSBoost "
               "from %.2f to %.3f, while RFOED-NN goes from %.2f to %.2f. Section 02."
               % (mean(sc[(5, "EasyEnsemble")]), mean(sc[(150, "EasyEnsemble")]),
                  mean(sc[(5, "RUSBoost")]), mean(sc[(150, "RUSBoost")]),
                  mean(sc[(5, "RFOED-NN")]), mean(sc[(150, "RFOED-NN")])))
    best_flat = lambda ds: max(sl(ds, m) for m in ["Flat-RF", "Flat-RF+SMOTE", "Flat-RF+Oversample",  # noqa: E731
                                                   "Flat Balanced-RF", "OVR-RF"]
                               if sl(ds, m) == sl(ds, m))
    out.append("3. **Under a frozen representation, local decomposition wins where classes separate "
               "and loses where they overlap.** RFOED-NN minus best flat baseline: %+.3f on CLINC150, "
               "%+.3f on 20 Newsgroups, %+.3f on WOS46985. Section 01."
               % (sl("clinc150", "RFOED-NN") - best_flat("clinc150"),
                  sl("20newsgroups_ir50", "RFOED-NN") - best_flat("20newsgroups_ir50"),
                  sl("wos46985", "RFOED-NN") - best_flat("wos46985")))
    if ("clinc150", "nn") in casc and ("wos46985", "nn") in casc:
        a, b = casc[("clinc150", "nn")], casc[("wos46985", "nn")]
        out.append("4. **One computable quantity governs that boundary.** The admissibility term "
                   "½(K−1)β̄ is %.3f on CLINC150 and %.3f on WOS46985 at near-identical K; every class "
                   "in every tested configuration lies inside the proved bound. Section 07/cascade_bound."
                   % (0.5 * (a["K"] - 1) * a["beta_mean_overall"], 0.5 * (b["K"] - 1) * b["beta_mean_overall"]))
    def margin(ds, arm):
        vals = [ft.get((ds, arm, m)) for m in ["Flat-RF", "Flat-RF+SMOTE", "Flat Balanced-RF"]]
        vals = [v for v in vals if v is not None and v == v]
        r = ft.get((ds, arm, "RFOED-NN"))
        return (r - max(vals)) if (vals and r is not None) else float("nan")
    out.append("5. **Fine-tuning separates the two claims.** On fine-tuned features EasyEnsemble still "
               "scores %.3f on CLINC150 and %.3f on WOS46985, but RFOED-NN's margin over the best flat "
               "baseline becomes %+.3f, %+.3f and %+.3f on CLINC150, 20 Newsgroups and WOS46985. "
               "End-to-end fine-tuning (class-weighted) reaches %.3f, %.3f and %.3f. Section 07/finetuning."
               % (ft.get(("clinc150", "finetuned-features", "EasyEnsemble")) or float("nan"),
                  ft.get(("wos46985", "finetuned-features", "EasyEnsemble")) or float("nan"),
                  margin("clinc150", "finetuned-features"), margin("20newsgroups_ir50", "finetuned-features"),
                  margin("wos46985", "finetuned-features"),
                  ft.get(("clinc150", "ft-endtoend-classweighted", "FineTuned-MiniLM")) or float("nan"),
                  ft.get(("20newsgroups_ir50", "ft-endtoend-classweighted", "FineTuned-MiniLM")) or float("nan"),
                  ft.get(("wos46985", "ft-endtoend-classweighted", "FineTuned-MiniLM")) or float("nan")))
    out.append("6. **The fixed n_cap=2000 handicaps only the decomposition.** It is saturating on 20 "
               "Newsgroups and costs %.3f on CLINC150 and %.3f on WOS46985 relative to uncapped. "
               "Section 07/ncap_sensitivity."
               % (mean(ncap[("clinc150", "nn", 2000)]) - mean(ncap[("clinc150", "nn", None)]),
                  mean(ncap[("wos46985", "nn", 2000)]) - mean(ncap[("wos46985", "nn", None)])))
    return "\n".join(out)


def master_readme(dst: Path, index_rows, now) -> str:
    return "\n".join([
        "# Results bundle — imbalance remedies for high-cardinality text classification",
        "",
        "Every experimental result behind the manuscript *An Empirical Study of Resampling Collapse "
        "and Local Decomposition for Class Imbalance in High-Cardinality Text Classification*, "
        "prepared for submission to ACM Transactions on Knowledge Discovery from Data (TKDD).",
        "",
        "Exported %s by `scripts/export_results.py` from the `freqcascade` repository. Every summary "
        "table in this bundle is computed from the raw per-fold files it sits next to; nothing is "
        "transcribed by hand. Re-running the export rebuilds the whole folder." % now,
        "",
        "## Headline findings",
        "",
        headline(dst),
        "",
        "## Contents",
        "",
        md_table(["folder", "what it holds", "backs (paper labels)"], index_rows),
        "",
        "Also at the top level:",
        "",
        "- [`SUMMARY.md`](SUMMARY.md) — full statistical summary regenerated from the current "
        "results: mean ± 95% CI for every metric, Holm-adjusted paired Wilcoxon against the "
        "reference method, the factorial ANOVA / Scheirer–Ray–Hare tables, and RQ5.",
        "- [`folds/`](folds/README.md) — the committed cross-validation fold indices every method consumed.",
        "- `logs/` — run logs for the experiments added on 2026-09-16.",
        "- [`provenance/`](provenance/) — repository commit and branch, the uncommitted diff at export "
        "time (`uncommitted.patch`), library and GPU versions, and a snapshot of every script.",
        "- [`99_superseded_do_not_cite/`](99_superseded_do_not_cite/README.md) — results replaced by "
        "later runs, kept for audit with a superseded-versus-current comparison.",
        "- [`MANIFEST.md`](MANIFEST.md) — SHA-256, byte size and row count for every file.",
        "",
        "## Corpora",
        "",
        md_table(["corpus", "type", "classes / labels", "examples", "domain", "imbalance ratio", "protocol"], [
            ["CLINC150", "single-label", "150", "18,025", "general (intents)", "4.0×", "fixed split × 10 seeds"],
            ["20 Newsgroups", "single-label", "20", "5,272", "general (topics)", "≈50× (injected)", "5×2 CV"],
            ["WOS46985", "single-label", "134", "46,985", "scientific abstracts", "17.5×", "5×2 CV"],
            ["OHSUMED-23", "single-label", "23", "23,166", "biomedical (MeSH C14)", "29.3×", "5×2 CV"],
            ["Drug Reviews", "single-label", "≈356", "49,995", "biomedical (conditions)", "≈1830×", "5×2 CV"],
            ["Reuters-21578", "multi-label", "90", "10,788", "general (news)", "1982× per label", "iterative 5×2 CV"],
            ["Hallmarks of Cancer", "multi-label", "10", "1,580", "biomedical abstracts", "4.4× per label", "iterative 5×2 CV"],
            ["LitCovid", "multi-label", "7", "33,699", "biomedical abstracts", "16.8× per label", "iterative 5×2 CV"],
        ]),
        "",
        "20 Newsgroups imbalance is injected by deterministic geometric subsampling at a fixed seed. "
        "CLINC150's imbalance ratio is computed on its training partition.",
        "",
        "## Methods",
        "",
        "- **RFOED** (single-label): recursive peel in descending class frequency; node k solves "
        "\"class c_k vs every class still active\", resolved documents leave contention. Base learner "
        "RF (`RFOED-RF`, TF-IDF) or a GPU-batched bagged MLP ensemble (`RFOED-NN`, frozen embeddings).",
        "- **FOCC** (multi-label): classifier chain in descending label frequency with feature "
        "augmentation by earlier links' labels; nothing removed. `FOCC-RF`, `FOCC-NN`.",
        "- **Local rebalancing** at every node/link: balanced bootstrap drawing both strata to "
        "min(max(n+, n−), n_cap); optional Elkan prior correction.",
        "- **Flat baselines** (Random Forest on TF-IDF): none, random under/over-sampling, SMOTE, "
        "ADASYN (floored-minority variant), SMOTE+ENN, class-weighted Balanced RF, One-vs-Rest RF.",
        "- **Resampling ensembles**: EasyEnsemble; RUSBoost with its canonical decision-stump learner.",
        "- **Multi-label baselines**: Binary Relevance, Balanced Binary Relevance, Classifier Chains, "
        "Ensemble of Classifier Chains (all RF).",
        "",
        "## Metrics",
        "",
        "- **macro-F1 / macro-recall**: unweighted mean over classes. Headline metric.",
        "- **G-mean**: geometric mean of per-class recall; zero if any class is never recalled.",
        "- **MCC**: multiclass Matthews correlation.",
        "- **bottom-quartile recall**: mean recall over the rarest 25% of classes by training frequency.",
        "- Multi-label analogues: **label-macro-F1**, **label-macro-recall**, **bottom-quartile label "
        "recall**, **example-F1** (per-document set F1), **micro-F1** (continuity only), Hamming loss, "
        "subset accuracy.",
        "- Accuracy is deliberately not reported as a headline metric.",
        "",
        "## Statistical protocol",
        "",
        "Per corpus: paired Wilcoxon signed-rank over matched folds/seeds, Holm–Bonferroni within each "
        "(metric, corpus) family. Across corpora: Friedman on within-corpus ranks with Nemenyi "
        "post-hoc (datasets as blocks, so mixing the fixed-split and CV protocols does not enter the "
        "ranks). Factorial ablation: three-way ANOVA, Scheirer–Ray–Hare where assumptions fail. The "
        "paired Wilcoxon on 10 folds has a two-sided resolution floor of p ≈ 0.002.",
        "",
        "## Reproducing",
        "",
        "From the `freqcascade` repository root (scripts are also snapshotted in `provenance/scripts/`):",
        "",
        "```",
        "python scripts/run_single_label_benchmark.py            # 01, single-label",
        "python scripts/run_multilabel_benchmark.py              # 01, multi-label",
        "python scripts/scaling_curve.py                         # 02",
        "python scripts/run_factorial_ablation.py                # 03",
        "python scripts/run_representation_study.py             # 04",
        "python scripts/diagnose_ncap_wos.py; python scripts/diagnose_wos.py     # 05 diagnosis",
        "python scripts/diagnose_wos_sweep.py; python scripts/run_wos_remediation.py  # 05 remediation",
        "python scripts/dump_per_class_recall.py; python scripts/focc_exposure_bias.py  # 05 tail, exposure",
        "python scripts/run_unit_pilot.py                        # 06",
        "python scripts/run_ncap_sweep.py                        # 07 ncap_sensitivity",
        "python scripts/run_finetune_study.py                    # 07 finetuning",
        "python scripts/verify_cascade_bound.py                  # 07 cascade_bound",
        "python scripts/run_clinc150_cv.py                       # 07 protocol_check",
        "python scripts/run_leak_check.py && python scripts/run_leak_check.py --compare  # 07 leak",
        "python scripts/spot_check_trees.py                      # 08",
        "python scripts/summarize.py > SUMMARY.md",
        "python scripts/export_results.py <this folder>",
        "```",
        "",
        "All long runs are resumable: finished cells are skipped on re-run.",
        "",
        "## Known limitations",
        "",
        "- The cross-validated main results fit the TF-IDF vectoriser over all rows (transductive, no "
        "labels). Its effect is measured in `07_reviewer_response/tfidf_leak_check` "
        "(**still running at export time**). Embedding methods are unaffected.",
        "- Every RFOED-NN number uses n_cap=2000, which understates it on four of five corpora.",
        "- The fine-tuning study and the cascade-bound check use one held-out split per corpus, not "
        "the fold protocol.",
        "- Drug Reviews runs a reduced method set (depth-capped RFOED-NN, no RFOED-RF).",
        "- The factorial ablation uses 6 of the 10 folds.",
        "",
    ])


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: python scripts/export_results.py <destination folder>")
    build(Path(sys.argv[1]).resolve())
