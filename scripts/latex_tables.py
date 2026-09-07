"""Emit the paper's results tables as LaTeX tabular blocks, from
results/*.jsonl. Writes revision/latex-tables.tex-ready text to stdout.

    python scripts/latex_tables.py > ../Imb_Learn/revision/latex-tables.tex
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from freqcascade import stats

R = Path(__file__).resolve().parent.parent / "results"


def rows(name):
    p = R / f"{name}.jsonl"
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []


def agg(name, metric):
    d = defaultdict(list)
    for r in rows(name):
        if "error" not in r and metric in r and "method" in r:
            d[r["method"]].append(r[metric])
    return {m: np.array(v) for m, v in d.items()}


def ci(v):
    c = stats.t_confidence_interval(v)
    return f"{c.mean:.3f}$\\pm${c.half_width:.3f}" if not np.isnan(c.half_width) else f"{c.mean:.3f}"


def ptex(p):
    s = stats.format_p(p)
    return "$p<0.001$" if s.startswith("<") else f"$p={s}$"


def sl_panel(name, letter, caption, ref="RFOED-NN"):
    metrics = ["macro_f1", "gmean", "mcc", "bottom_quartile_recall"]
    a0 = agg(name, "macro_f1")
    if not a0:
        return f"% {name}: no data\n"
    order = sorted(a0, key=lambda m: -a0[m].mean())
    have_ref = ref in a0
    raw = {}
    if have_ref:
        for m in order:
            if m == ref:
                continue
            b = a0[m]
            if len(b) == len(a0[ref]) and len(b) > 1:
                raw[m] = stats.paired_comparison(a0[ref], b).wilcoxon_p
    adj = dict(zip(raw, stats.holm_bonferroni(list(raw.values())))) if raw else {}
    lines = [r"\begin{tabular}{lcccc}", r"\toprule",
             r"Method & macro-F1 & G-mean & MCC & bottom-quartile recall \\", r"\midrule"]
    per = {mt: agg(name, mt) for mt in metrics}
    for m in order:
        cells = [ci(per[mt][m]) if m in per[mt] else "--" for mt in metrics]
        star = ""
        if have_ref and m != ref and m in adj:
            star = "*" if adj[m] < 0.05 else ""
        elif m == ref:
            star = ""
        name_cell = f"\\textbf{{{m}}}" if m == ref else m
        lines.append(f"{name_cell} & " + " & ".join(f"\\textbf{{{c}}}" if m == ref else c for c in cells) + f"{star} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", f"\\caption*{{({letter}) {caption}}}", ""]
    return "\n".join(lines)


def ml_panel(name, letter, caption, ref="FOCC-NN"):
    metrics = ["label_macro_f1", "bottom_quartile_label_recall", "example_f1", "micro_f1"]
    a0 = agg(name, "label_macro_f1")
    if not a0:
        return f"% {name}: no data\n"
    order = sorted(a0, key=lambda m: -a0[m].mean())
    raw = {}
    for m in order:
        if m == ref:
            continue
        b = a0[m]
        if ref in a0 and len(b) == len(a0[ref]) and len(b) > 1:
            raw[m] = stats.paired_comparison(a0[ref], b).wilcoxon_p
    adj = dict(zip(raw, stats.holm_bonferroni(list(raw.values())))) if raw else {}
    per = {mt: agg(name, mt) for mt in metrics}
    lines = [r"\begin{tabular}{lcccc}", r"\toprule",
             r"Method & label-macro-F1 & bq-label recall & example-F1 & micro-F1 \\", r"\midrule"]
    for m in order:
        cells = [ci(per[mt][m]) if m in per[mt] else "--" for mt in metrics]
        star = "*" if (m != ref and adj.get(m, 1) < 0.05) else ""
        name_cell = f"\\textbf{{{m}}}" if m == ref else m
        lines.append(f"{name_cell} & " + " & ".join(cells) + f"{star} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", f"\\caption*{{({letter}) {caption}}}", ""]
    return "\n".join(lines)


def factorial_anova():
    import pandas as pd
    out = ["% ---- Table: 3-way ANOVA on macro-F1, single-label factorial ----",
           r"\begin{tabular}{lccc}", r"\toprule",
           r"Factor & 20 Newsgroups & CLINC150 & WOS46985 \\", r"\midrule"]
    tabs = {}
    for ds in ["20newsgroups_ir50", "clinc150", "wos46985"]:
        rr = rows(f"ablation_{ds}")
        if not rr:
            continue
        df = pd.DataFrame(rr)
        res = stats.anova_or_srh(df, "macro_f1", ("ordering", "base_learner", "rebalance"))
        tabs[ds] = res.anova.table
    labels = [("C(ordering)", "Ordering"), ("C(base_learner)", "Base learner"),
              ("C(rebalance)", "Rebalance"), ("C(ordering):C(base_learner)", "Ord$\\times$Base"),
              ("C(ordering):C(rebalance)", "Ord$\\times$Reb"), ("C(base_learner):C(rebalance)", "Base$\\times$Reb"),
              ("C(ordering):C(base_learner):C(rebalance)", "Ord$\\times$Base$\\times$Reb")]
    for key, lab in labels:
        cells = []
        for ds in ["20newsgroups_ir50", "clinc150", "wos46985"]:
            t = tabs.get(ds)
            if t is None or key not in t.index:
                cells.append("--"); continue
            F = t.loc[key, "F"]; p = t.loc[key, "PR(>F)"]
            cells.append(f"$F$={F:.1f}, {ptex(p)}")
        out.append(f"{lab} & " + " & ".join(cells) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}", ""]
    return "\n".join(out)


def factorial_ml():
    import pandas as pd
    import statsmodels.formula.api as smf
    from statsmodels.stats.anova import anova_lm
    out = ["% ---- Table: FOCC 2x2 ordering x rebalance, label-macro-F1 ----",
           r"\begin{tabular}{llccc}", r"\toprule",
           r"Dataset & & label-macro-F1 & $p$(ordering) & $p$(rebalance) \\", r"\midrule"]
    for ds, pretty in [("reuters21578", "Reuters-21578"), ("hoc", "HoC"), ("litcovid", "LitCovid")]:
        rr = rows(f"ablation_{ds}")
        if not rr:
            continue
        df = pd.DataFrame(rr)
        m = smf.ols("label_macro_f1 ~ C(ordering) * C(rebalance)", data=df).fit()
        t = anova_lm(m, typ=2)
        po = ptex(t.loc["C(ordering)", "PR(>F)"])
        pr = ptex(t.loc["C(rebalance)", "PR(>F)"])
        cells = df.groupby(["ordering", "rebalance"])["label_macro_f1"].mean()
        fon = cells.get(("frequency", True), float("nan"))
        rand = df[df.ordering == "random"]["label_macro_f1"].mean()
        freq = df[df.ordering == "frequency"]["label_macro_f1"].mean()
        out.append(f"{pretty} & freq {freq:.3f} / rand {rand:.3f} & {fon:.3f} & {po} & {pr} \\\\")
    out += [r"\bottomrule", r"\end{tabular}", ""]
    return "\n".join(out)


def rq5_table():
    out = ["% ---- Table: RQ5 representation study ----",
           r"\begin{tabular}{lcccc}", r"\toprule",
           r"Dataset & TF-IDF baseline & MiniLM & S-PubMedBERT & MPNet \\", r"\midrule"]
    for ds, pretty, metric in [("clinc150", "CLINC150", "macro_f1"), ("ohsumed_23", "OHSUMED-23", "macro_f1"),
                               ("hoc", "HoC", "label_macro_f1"), ("litcovid", "LitCovid", "label_macro_f1")]:
        rr = rows(f"rq5_{ds}")
        if not rr:
            continue
        d = defaultdict(list)
        for r in rr:
            if "error" not in r and metric in r:
                d[r.get("encoder")].append(r[metric])
        g = {k: np.mean(v) for k, v in d.items()}
        row = [f"{g.get('tfidf', float('nan')):.3f}", f"{g.get('minilm', float('nan')):.3f}",
               f"{g.get('pubmedbert', float('nan')):.3f}", f"{g.get('mpnet', float('nan')):.3f}"]
        out.append(f"{pretty} & " + " & ".join(row) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}", ""]
    return "\n".join(out)


if __name__ == "__main__":
    print("%% === Table 3: single-label main results (paste into tab:mainresults) ===\n")
    print(sl_panel("clinc150", "a", "CLINC150 --- fixed split, 10 seeds."))
    print(sl_panel("20newsgroups_ir50", "b", "20 Newsgroups --- $5\\times2$ CV, 10 folds."))
    print(sl_panel("wos46985", "c", "WOS46985 --- $5\\times2$ CV, 10 folds (plain configuration)."))
    print(sl_panel("ohsumed_23", "d", "OHSUMED-23 (biomedical) --- $5\\times2$ CV, 10 folds."))
    print(sl_panel("drug_reviews", "e", "Drug Reviews (biomedical, $K\\approx356$) --- $5\\times2$ CV. RFOED-NN capped; RFOED-RF omitted (compute)."))
    print("\n%% === Table 4: multi-label results (paste into tab:focc) ===\n")
    print(ml_panel("reuters21578", "a", "Reuters-21578 --- $5\\times2$ iterative-stratified CV."))
    print(ml_panel("hoc", "b", "Hallmarks of Cancer (biomedical)."))
    print(ml_panel("litcovid", "c", "LitCovid (biomedical)."))
    print("\n%% === Table 7: factorial ANOVA ===\n")
    print(factorial_anova())
    print("\n%% === Table 8: FOCC 2x2 ablation, n=3 ===\n")
    print(factorial_ml())
    print("\n%% === Table: RQ5 ===\n")
    print(rq5_table())
