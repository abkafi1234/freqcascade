"""LaTeX table bodies for the round-3 experiments (R5, R6, R8, R9).

Same discipline as latex_reviewer_tables.py: every number comes from the jsonl,
every difference is computed from unrounded means, and nothing is transcribed by
hand. Each table tolerates partial data, so it can be previewed while the runs
are still going -- missing cells print as "--" and the header line reports how
much of the target is in.

    python scripts/latex_round3_tables.py r5        # weak-learner capacity ladder
    python scripts/latex_round3_tables.py r6        # cascade node-decision dependence
    python scripts/latex_round3_tables.py r8        # fixed vs adaptive n_cap
    python scripts/latex_round3_tables.py r9        # per-link exposure bias
    python scripts/latex_round3_tables.py all
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
    "hoc": "HoC",
    "litcovid": "LitCovid",
    "reuters21578": "Reuters-21578",
}
LEARNER = {"stump": "Stump (depth 1)", "tree3": "Tree (depth 3)", "tree10": "Tree (depth 10)",
           "treefull": "Tree (unbounded)", "mlp": "MLP", "xgb": "XGBoost"}
LEARNER_ORDER = ["stump", "tree3", "tree10", "treefull", "mlp", "xgb"]
K_VALUES = [5, 10, 25, 50, 100, 150]


def load(name):
    p = RESULTS_DIR / name
    if not p.exists():
        return []
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]


def fmt(v, nd=3):
    return "--" if v is None else ("%." + str(nd) + "f") % v


def mean(vals):
    return st.fmean(vals) if vals else None


def pretty(name):
    return PRETTY.get(name, name)


# --------------------------------------------------------------------------- #
# R5: does weak-learner capacity remove the high-K collapse?
# --------------------------------------------------------------------------- #

def r5_table(features="tfidf"):
    rows = [r for r in load("weaklearner_sweep.jsonl")
            if r.get("protocol") == "class-count sweep" and r.get("features") == features
            and "macro_f1" in r]
    g = collections.defaultdict(list)
    for r in rows:
        g[(r["ensemble"], r["learner"], r["K"])].append(r["macro_f1"])
    seeds = {r["seed"] for r in rows}
    out = ["%% R5 capacity ladder, features=%s, %d cells, seeds %s"
           % (features, len(rows), sorted(seeds))]
    out.append("%s & %s %s" % ("Ensemble & Weak learner",
                               " & ".join("$K{=}%d$" % k for k in K_VALUES), B + B))
    for ens in ("RUSBoost", "EasyEnsemble", "flat"):
        label = "Flat (no resampling)" if ens == "flat" else ens
        first = True
        for ln in LEARNER_ORDER:
            cells = [mean(g.get((ens, ln, k), [])) for k in K_VALUES]
            if all(c is None for c in cells):
                continue
            head = ("%s{%s}" % (B + "multirow{4}*", label)) if first else ""
            out.append("%s & %s & %s %s" % (head, LEARNER[ln],
                                            " & ".join(fmt(c) for c in cells), B + B))
            first = False
        if not first:
            out.append(B + "midrule")
    return "\n".join(out)


def r5_delta_note():
    """The sentence the paper needs: what capacity buys at the top of the sweep,
    computed from unrounded means rather than read off the table."""
    rows = [r for r in load("weaklearner_sweep.jsonl")
            if r.get("protocol") == "class-count sweep" and "macro_f1" in r]
    g = collections.defaultdict(list)
    for r in rows:
        g[(r["features"], r["ensemble"], r["learner"], r["K"])].append(r["macro_f1"])
    lines = []
    for feats in ("tfidf", "emb"):
        for ens in ("RUSBoost", "EasyEnsemble"):
            for k in (100, 150):
                lo, hi = mean(g.get((feats, ens, "stump", k), [])), mean(g.get((feats, ens, "treefull", k), []))
                flat = mean(g.get((feats, ens.replace(ens, "flat"), "treefull", k), []))
                if lo is None or hi is None:
                    continue
                lines.append("%-6s %-13s K=%-4d stump %.4f -> unbounded tree %.4f  (gain %+.4f; "
                             "flat same learner %s)" % (feats, ens, k, lo, hi, hi - lo, fmt(flat, 4)))
    return "\n".join(lines) or "(no complete stump/treefull pair yet)"


def r5_compact_table():
    """Main-text R5 table: stump -> tree -> unbounded tree -> MLP ladder at
    K = 5, 50, 150 on both feature sets.  Mean over the three class subsets (seeds).
    A cell with no completed seed prints "--"; for the unbounded tree under
    EasyEnsemble that means the cell exceeded the 6 GB cgroup cap the rung ran
    under, not that the method cannot run (see make_r5_summary.py).  XGBoost is
    absent from the resampling rows because AdaBoost rejects it as a weak learner."""
    rows = [r for r in load("weaklearner_sweep.jsonl")
            if r.get("protocol") == "class-count sweep" and "macro_f1" in r]
    g = collections.defaultdict(list)
    for r in rows:
        g[(r["features"], r["ensemble"], r["learner"], r["K"])].append(r["macro_f1"])
    ks = (5, 50, 150)
    spec = [("RUSBoost", ("stump", "tree3", "tree10", "treefull", "mlp")),
            ("EasyEnsemble", ("stump", "tree3", "tree10", "treefull", "mlp")),
            ("flat", ("tree10", "treefull", "mlp", "xgb"))]
    out = ["%% R5 compact, generated by latex_round3_tables.py r5compact"]
    for ens, learners in spec:
        label = "Flat (no resampling)" if ens == "flat" else ens
        for i, ln in enumerate(learners):
            cells = [fmt(mean(g.get((f, ens, ln, k), []))) for f in ("tfidf", "emb") for k in ks]
            head = ("%s{%d}*{%s}" % (B + "multirow", len(learners), label)) if i == 0 else ""
            out.append("%s & %s & %s %s" % (head, LEARNER[ln], " & ".join(cells), B + B))
        out.append(B + "midrule")
    out[-1] = B + "bottomrule"
    return "\n".join(out)


def diagnosis_table():
    """tab:diagnosis body from results/wos46985_diagnosis.jsonl (diagnose_wos.py):
    RFOED-RF and RFOED-NN at the fixed cap, and RFOED-NN uncapped (which the
    gamma = 128 cap of the main results matches on WOS46985 to 2e-5)."""
    rows = {(r["base"], r["cap"]): r for r in load("wos46985_diagnosis.jsonl")}
    cols = [rows[("rf", 2000)], rows[("nn", 2000)], rows[("nn", None)]]
    pct = lambda v: "%.1f\\%%" % (100 * v)
    spec = [("Errors on the diagnostic split", lambda r: "{:,}".format(r["n_errors"]).replace(",", "{,}")),
            ("Own-node-miss share of errors", lambda r: pct(r["own_node_miss_frac"])),
            ("Early-capture share of errors", lambda r: pct(r["early_capture_frac"])),
            ("False captures in first 25\\% of depth", lambda r: pct(r["fp_captures_first_quartile"])),
            ("Own-class test recall, mean over nodes", lambda r: "%.2f" % r["own_class_test_recall_mean"]),
            ("Own-class training recall, mean over nodes", lambda r: "%.2f" % r["own_class_train_recall_mean"]),
            ("False-capture rate on later classes", lambda r: "%.3f" % r["fpr_later_mean"]),
            ("Macro-F1 on the diagnostic split", lambda r: "%.3f" % r["macro_f1"])]
    out = ["%% generated by latex_round3_tables.py diagnosis"]
    for lab, f in spec:
        out.append("%s & %s %s" % (lab, " & ".join(f(r) for r in cols), B + B))
    return "\n".join(out)


# --------------------------------------------------------------------------- #
# R6: how far from independent are the node decisions?
# --------------------------------------------------------------------------- #

def r6_table():
    """One row per (dataset, base learner). Identity error and agreement check
    Eq. chainrule; `within` is the share of classes inside [union bound,
    Bonferroni bound]; width is that interval's mean width; excess is the mean
    over classes of sum_{a<b} [P(E_a & E_b) - P(E_a)P(E_b)], i.e. pairwise
    co-failure beyond independence; phi is the median adjacent-node phi with the
    number of pairs on which it is defined."""
    rows = [r for r in load("cascade_dependence.jsonl") if r.get("kind") == "summary"]
    order = {"clinc150": 0, "20newsgroups_ir50": 1, "wos46985": 2}
    out = ["%% R6 dependence, %d configurations" % len(rows)]
    out.append("Dataset & Base & $K$ & identity err. & agreement & within & width & excess "
               "& product-form MAE & adjacent $%sphi$ [pairs] %s" % (B, B + B))
    for r in sorted(rows, key=lambda r: (r["base"], order.get(r["dataset"], 9))):
        phi = ("%.3f [%d]" % (r["adj_phi_median"], r["adj_pairs_phi_defined"])
               if r.get("adj_phi_median") is not None else "-- [0]")
        out.append("%s & %s & %d & %.0f & %.3f & %.3f & %.3f & %+.3f & %.3f & %s %s" % (
            pretty(r["dataset"]), "NN" if r["base"] == "nn" else "RF", r["K"],
            r["identity_max_abs_err_vs_firstfire"], r["firstfire_vs_predict_agreement"],
            r["within_union_and_bonf2"], r["bonf2_interval_width_mean"], r["pair_cofire_excess_mean"],
            r["indep_form_mae"], phi, B + B))
    return "\n".join(out)


# --------------------------------------------------------------------------- #
# R8: fixed cap vs data-scaled cap vs uncapped
# --------------------------------------------------------------------------- #

def r8_table():
    rows = [r for r in load("ncap_adaptive.jsonl") if r.get("score") is not None]
    errs = [r for r in load("ncap_adaptive.jsonl") if r.get("score") is None]
    g, mem = collections.defaultdict(list), collections.defaultdict(list)
    for r in rows:
        arm = r["rule"] if r["rule"] != "adaptive" else ("adaptive $%sgamma{=}%s$" % (B, r["gamma"]))
        g[(r["dataset"], arm)].append(r["score"])
        if r.get("peak_gpu_mem_gb") is not None:
            mem[(r["dataset"], arm)].append(r["peak_gpu_mem_gb"])
    arms = ["fixed", "uncapped"] + ["adaptive $%sgamma{=}%d$" % (B, x) for x in (2, 8, 32, 128)]
    datasets = sorted({d for d, _ in g})
    out = ["%% R8 adaptive n_cap: %d scored cells, %d errors" % (len(rows), len(errs))]
    out.append("Dataset & %s %s" % (" & ".join(a.replace("fixed", "fixed 2000") for a in arms), B + B))
    for d in datasets:
        cells = []
        for a in arms:
            v = mean(g.get((d, a), []))
            m = mean(mem.get((d, a), []))
            cells.append("--" if v is None else ("%.3f" % v + ("" if m is None else " {%ssmall (%.1f\\,GB)}" % (B, m))))
        out.append("%s & %s %s" % (pretty(d), " & ".join(cells), B + B))
    # the decision rule, evaluated from unrounded means
    verdict = []
    for d in datasets:
        unc = mean(g.get((d, "uncapped"), []))
        if unc is None:
            continue
        worse = [a for a in arms if a.startswith("adaptive") and mean(g.get((d, a), []) or []) is not None
                 and mean(g[(d, a)]) > unc]
        verdict.append("%s: uncapped %.4f%s" % (pretty(d), unc,
                       "" if not worse else "  BEATEN BY " + ", ".join(
                           "%s (%.4f)" % (a, mean(g[(d, a)])) for a in worse)))
    out.append("%% D2 rule check (uncapped >= every adaptive gamma on every corpus):")
    out += ["%% " + v for v in verdict]
    return "\n".join(out)


# --------------------------------------------------------------------------- #
# R9: does error accumulate down a long chain?
# --------------------------------------------------------------------------- #

def r9_table(bins=5):
    rows = load("focc_perlink.jsonl")
    g = collections.defaultdict(list)
    for r in rows:
        g[(r["dataset"], r["order"], r["fold"])].append(r)
    complete = {k: v for k, v in g.items() if len(v) == v[0]["n_links"]}
    out = ["%% R9 per-link exposure bias, %d complete folds of 60" % len(complete)]
    out.append("Dataset & Order & %s & overall %s"
               % (" & ".join("links %d--%d\\%%" % (100 * i // bins, 100 * (i + 1) // bins)
                             for i in range(bins)), B + B))
    by = collections.defaultdict(lambda: collections.defaultdict(list))
    overall = collections.defaultdict(list)
    for (ds, order, _), links in complete.items():
        links = sorted(links, key=lambda r: r.get("rank", 0))
        n = len(links)
        for i, r in enumerate(links):
            dep, orc = r.get("deployed") or {}, r.get("oracle") or {}
            if dep.get("f1") is None or orc.get("f1") is None:
                continue
            gap = orc["f1"] - dep["f1"]          # oracle minus deployed: the exposure-bias cost
            by[(ds, order)][min(bins - 1, i * bins // n)].append(gap)
            overall[(ds, order)].append(gap)
    for key in sorted(by):
        cells = [fmt(mean(by[key][i]), 3) for i in range(bins)]
        out.append("%s & %s & %s & %s %s" % (pretty(key[0]), key[1], " & ".join(cells),
                                             fmt(mean(overall[key]), 3), B + B))
    return "\n".join(out)


def r9_order_by_mode_table():
    """Frequency-minus-random ordering effect on the chain under deployed and
    under oracle inference, on the same matched folds. Differences, fold counts
    and p-values are computed from unrounded per-fold means; Holm correction is
    applied across the three corpora within each inference mode."""
    import numpy as np
    from scipy.stats import wilcoxon

    rows = load("focc_perlink.jsonl")
    by = collections.defaultdict(list)
    for r in rows:
        by[(r["dataset"], r["order"], r["fold"])].append(r)
    corpora = [d for d in ("reuters21578", "hoc", "litcovid")
               if all((d, o, k) in by for o in ("frequency", "random") for k in range(10))]
    out = ["%% R9 ordering x inference mode, %d corpora x 10 matched folds" % len(corpora)]
    out.append("Inference & Dataset & frequency & random & difference & folds freq.\\ $>$ random "
               "& $p_{%s}$ %s" % (B + "mathrm{Holm}", B + B))
    for mode, label in (("deployed", "Deployed"), ("oracle", "Oracle")):
        stats = []
        for d in corpora:
            f = np.array([np.mean([l[mode]["f1"] for l in by[(d, "frequency", k)]]) for k in range(10)])
            r = np.array([np.mean([l[mode]["f1"] for l in by[(d, "random", k)]]) for k in range(10)])
            stats.append((d, f.mean(), r.mean(), (f - r).mean(), int(((f - r) > 0).sum()), wilcoxon(f, r).pvalue))
        order = sorted(range(len(stats)), key=lambda i: stats[i][5])
        adj, prev = [0.0] * len(stats), 0.0
        for j, i in enumerate(order):
            prev = max(prev, min(1.0, (len(stats) - j) * stats[i][5]))
            adj[i] = prev
        for n, ((d, fm, rm, diff, npos, _), pa) in enumerate(zip(stats, adj)):
            head = ("%s{%d}*{%s}" % (B + "multirow", len(stats), label)) if n == 0 else ""
            out.append("%s & %s & %.3f & %.3f & %+.3f & %d/10 & %.3f %s"
                       % (head, pretty(d), fm, rm, diff, npos, pa, B + B))
        out.append(B + "midrule")
    out.pop()
    return "\n".join(out)


# --------------------------------------------------------------------------- #
# R7: does the fine-tuned encoder change either claim?
# --------------------------------------------------------------------------- #

PROPOSED = {False: "RFOED-NN", True: "FOCC-NN"}
HEADLINE = {False: "macro_f1", True: "label_macro_f1"}
TAIL = {False: "bottom_quartile_recall", True: "bottom_quartile_label_recall"}
R7_ORDER = ["clinc150", "20newsgroups_ir50", "wos46985", "ohsumed_23", "drug_reviews",
            "reuters21578", "hoc", "litcovid"]


def _r7_cells(rows, arm):
    """(dataset, method) -> {(loss, fold): headline score} for one arm."""
    g = collections.defaultdict(dict)
    for r in rows:
        if r.get("arm") != arm:
            continue
        met = HEADLINE[bool(r.get("multilabel"))]
        if r.get(met) is None:
            continue
        g[(r["dataset"], r["method"])][(r["loss"], r["fold"])] = r[met]
    return g


def r7_table():
    """Headline metric on fine-tuned DeBERTa features, proposed vs the best rival."""
    rows = [r for r in load("finetune_grid.jsonl") if not r.get("error")]
    errs = [r for r in load("finetune_grid.jsonl") if r.get("error")]
    feat = _r7_cells(rows, "finetuned-features")
    e2e = _r7_cells(rows, "ft-endtoend")
    datasets = [d for d in R7_ORDER if any(k[0] == d for k in feat)]
    out = ["%% R7 DeBERTa-v3-base fine-tuning grid: %d scored rows, %d error rows" % (len(rows), len(errs)),
           "% metric: macro-F1 (single-label), label-macro-F1 (multi-label); mean over 3 losses x 3 folds",
           "Dataset & $K$ & End-to-end & Proposed & Best rival & Margin %s" % (B + B)]
    wins = 0
    for d in datasets:
        multi = any(r.get("multilabel") for r in rows if r["dataset"] == d)
        prop_name = PROPOSED[bool(multi)]
        prop = mean(list(feat.get((d, prop_name), {}).values()))
        rivals = {m: mean(list(v.values())) for (dd, m), v in feat.items()
                  if dd == d and m != prop_name and v}
        ee = mean(list(e2e.get((d, "FineTuned-DeBERTa"), {}).values()))
        K = next((r.get("K") for r in rows if r["dataset"] == d and r.get("K")), None)
        if not rivals or prop is None:
            out.append("%s & %s & %s & %s & -- & -- %s"
                       % (pretty(d), K or "--", fmt(ee), fmt(prop), B + B))
            continue
        bm = max(rivals, key=rivals.get)
        margin = prop - rivals[bm]
        if margin >= 0:
            wins += 1
        out.append("%s & %s & %s & %s & %s (%s) & %s%.3f %s"
                   % (pretty(d), K or "--", fmt(ee), fmt(prop), fmt(rivals[bm]), bm,
                      "$+$" if margin >= 0 else "$-$", abs(margin), B + B))
    out.append("%% proposed >= best rival on %d of %d corpora (headline metric)" % (wins, len(datasets)))
    return "\n".join(out)


def r7_tail_table():
    """The paper's actual claim: rare-label recall, frozen encoder vs fine-tuned."""
    from freqcascade.stats import paired_comparison

    rows = [r for r in load("finetune_grid.jsonl") if not r.get("error")]
    out = ["% R7 rare-label recall, FOCC-NN vs Flat-MLP on the same features",
           "% frozen: <corpus>.jsonl (10 folds); fine-tuned: finetune_grid.jsonl (3 losses x 3 folds)",
           ("Dataset & Frozen FOCC & Frozen flat & Margin & FT FOCC & FT flat & Margin & $p$ %s"
            % (B + B))]
    for d in ["reuters21578", "hoc", "litcovid"]:
        met = TAIL[True]
        # frozen: paired on fold
        fz = collections.defaultdict(dict)
        for r in load("%s.jsonl" % d):
            if r.get(met) is not None:
                fz[r["method"]][r["fold"]] = r[met]
        # fine-tuned: paired on (loss, fold)
        ft = collections.defaultdict(dict)
        for r in rows:
            if r["dataset"] == d and r.get("arm") == "finetuned-features" and r.get(met) is not None:
                ft[r["method"]][(r["loss"], r["fold"])] = r[met]
        cells, pval = [], None
        for src in (fz, ft):
            a, b = src.get("FOCC-NN", {}), src.get("Flat-MLP", {})
            ks = sorted(set(a) & set(b))
            if not ks:
                cells += ["--", "--", "--"]
                continue
            av = [a[k] for k in ks]
            bv = [b[k] for k in ks]
            m = mean(av) - mean(bv)
            cells += [fmt(mean(av)), fmt(mean(bv)),
                      "%s%.3f" % ("$+$" if m >= 0 else "$-$", abs(m))]
            if src is ft:
                pval = paired_comparison(av, bv).wilcoxon_p
        out.append("%s & %s & %s %s" % (pretty(d), " & ".join(cells), fmt(pval, 3), B + B))
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("which", choices=["r5", "r5compact", "diagnosis", "r6", "r7", "r7tail", "r8", "r9", "r9order", "all"])
    ap.add_argument("--features", default="tfidf", help="r5 only: tfidf or emb")
    a = ap.parse_args()
    if a.which in ("r5", "all"):
        print("%% ---- R5 (features=%s) ----" % a.features)
        print(r5_table(a.features))
        print("\n%% R5 capacity gain, unrounded:")
        print("\n".join("%% " + x for x in r5_delta_note().splitlines()), "\n")
    if a.which in ("r5compact", "all"):
        print("%% ---- R5 compact ----"); print(r5_compact_table(), "\n")
    if a.which in ("diagnosis", "all"):
        print("%% ---- diagnosis ----"); print(diagnosis_table(), "\n")
    if a.which in ("r6", "all"):
        print("%% ---- R6 ----"); print(r6_table(), "\n")
    if a.which in ("r7", "all"):
        print("%% ---- R7 headline ----"); print(r7_table(), "\n")
    if a.which in ("r7tail", "all"):
        print("%% ---- R7 rare-label recall ----"); print(r7_tail_table(), "\n")
    if a.which in ("r8", "all"):
        print("%% ---- R8 ----"); print(r8_table(), "\n")
    if a.which in ("r9", "all"):
        print("%% ---- R9 ----"); print(r9_table(), "\n")
    if a.which in ("r9order", "all"):
        print("%% ---- R9 ordering x inference mode ----"); print(r9_order_by_mode_table(), "\n")


if __name__ == "__main__":
    main()
