"""One-shot: read every results/*.jsonl and print a Markdown summary --
per-dataset method tables (mean +/- 95% CI, Holm-adjusted Wilcoxon vs the
proposed method), the factorial ANOVA, and the RQ5 representation study.

    python scripts/summarize.py                 # everything found
    python scripts/summarize.py > results/SUMMARY.md
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from freqcascade import stats

RESULTS = Path(__file__).resolve().parent.parent / "results"

SL_METRICS = ["macro_f1", "gmean", "mcc", "bottom_quartile_recall"]
ML_METRICS = ["label_macro_f1", "bottom_quartile_label_recall", "example_f1", "micro_f1"]
SL_DATASETS = ["clinc150", "20newsgroups_ir50", "wos46985", "ohsumed_23", "drug_reviews"]
ML_DATASETS = ["reuters21578", "hoc", "litcovid"]


def _rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def _by_method(rows: list[dict], metric: str) -> dict[str, np.ndarray]:
    d: dict[str, list] = defaultdict(list)
    for r in rows:
        if "error" not in r and metric in r:
            d[r["method"]].append(r[metric])
    return {m: np.array(v) for m, v in d.items()}


def method_table(name: str, rows: list[dict], metrics: list[str], ref_candidates: list[str]) -> None:
    have_any = _by_method(rows, metrics[0])
    if not have_any:
        return
    ref = next((r for r in ref_candidates if r in have_any), None)
    print(f"\n### {name}  (n folds: {max(len(v) for v in have_any.values())})\n")
    print("| method | " + " | ".join(metrics) + " | vs " + (ref or "-") + " |")
    print("|" + "---|" * (len(metrics) + 2))

    order = sorted(have_any, key=lambda m: -float(np.mean(have_any[m])))
    # Holm across the non-ref family on the first metric
    raw_p = {}
    if ref:
        for m in order:
            if m == ref:
                continue
            a, b = _by_method(rows, metrics[0]).get(ref), _by_method(rows, metrics[0]).get(m)
            if a is not None and b is not None and len(a) == len(b) and len(a) > 1:
                raw_p[m] = stats.paired_comparison(a, b).wilcoxon_p
    adj = dict(zip(raw_p, stats.holm_bonferroni(list(raw_p.values())))) if raw_p else {}

    for m in order:
        cells = []
        for met in metrics:
            v = _by_method(rows, met).get(m)
            if v is None or not len(v):
                cells.append("--")
            else:
                ci = stats.t_confidence_interval(v)
                hw = "" if np.isnan(ci.half_width) else f"$\\pm${ci.half_width:.3f}"
                cells.append(f"{ci.mean:.3f}{hw}")
        mark = "**ref**" if m == ref else (
            ("sig " if adj.get(m, 1) < 0.05 else "ns ") + f"p={stats.format_p(adj.get(m, float('nan')))}"
            if m in adj else "")
        print(f"| {m} | " + " | ".join(cells) + f" | {mark} |")


def factorial_section() -> None:
    files = sorted(RESULTS.glob("ablation_*.jsonl"))
    if not files:
        return
    print("\n\n## Factorial ablation\n")
    for f in files:
        rows = _rows(f)
        if not rows:
            continue
        name = f.stem.replace("ablation_", "")
        ml = "label_macro_f1" in rows[0]
        resp = "label_macro_f1" if ml else "macro_f1"
        import pandas as pd
        df = pd.DataFrame(rows)
        factors = ["ordering", "rebalance"] + ([] if ml else ["base_learner"])
        try:
            if ml:
                from freqcascade.stats import three_way_anova
                # 2-way
                import statsmodels.formula.api as smf
                from statsmodels.stats.anova import anova_lm
                model = smf.ols(f"{resp} ~ C(ordering) * C(rebalance)", data=df).fit()
                tbl = anova_lm(model, typ=2)
                print(f"\n### {name} (2x2, response {resp})\n```\n{tbl}\n```")
            else:
                res = stats.anova_or_srh(df, resp, tuple(factors))
                print(f"\n### {name} (2x2x2, response {resp}; recommended: {res.recommended})\n```\n{res.anova.table}\n```")
        except Exception as e:  # noqa: BLE001
            print(f"\n### {name}: ANOVA failed ({e})")
        # cell means
        print("\ncell means:\n")
        gb = df.groupby(factors)[resp].agg(["mean", "std", "count"])
        print("```\n" + gb.to_string() + "\n```")


def rq5_section() -> None:
    files = sorted(RESULTS.glob("rq5_*.jsonl"))
    if not files:
        return
    print("\n\n## RQ5 -- representation study\n")
    print("| dataset | method | encoder | metric | mean |")
    print("|---|---|---|---|---|")
    for f in files:
        rows = _rows(f)
        name = f.stem.replace("rq5_", "")
        metric = "label_macro_f1" if rows and "label_macro_f1" in rows[0] else "macro_f1"
        agg: dict = defaultdict(list)
        for r in rows:
            if "error" not in r and metric in r:
                agg[(r["method"], r.get("encoder", "?"))].append(r[metric])
        for (m, enc), v in sorted(agg.items()):
            print(f"| {name} | {m} | {enc} | {metric} | {np.mean(v):.3f} |")


def main() -> None:
    print("# freqcascade -- Phase 2 results summary\n")
    print(f"generated from {len(list(RESULTS.glob('*.jsonl')))} result files\n")

    print("## Single-label benchmarks\n")
    for name in SL_DATASETS:
        method_table(name, _rows(RESULTS / f"{name}.jsonl"), SL_METRICS, ["RFOED-NN", "RFOED-RF"])

    print("\n\n## Multi-label benchmarks\n")
    for name in ML_DATASETS:
        method_table(name, _rows(RESULTS / f"{name}.jsonl"), ML_METRICS, ["FOCC-NN", "FOCC-RF"])

    factorial_section()
    rq5_section()


if __name__ == "__main__":
    main()
