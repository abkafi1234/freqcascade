"""Turn results/*.jsonl into the paper's significance tables.

- Per-dataset: mean +/- 95% CI per method per metric, plus paired Wilcoxon vs
  the reference method with Holm-Bonferroni correction across the baseline
  family (freqcascade.stats).
- Cross-dataset: Friedman + Nemenyi critical difference on the shared metric.

    python scripts/analyze.py --metric macro_f1 --ref RFOED-NN results/clinc150.jsonl
    python scripts/analyze.py --metric label_macro_f1 --ref FOCC-NN results/reuters21578.jsonl results/hoc.jsonl
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from freqcascade import stats


def load_rows(paths: list[str]):
    by_dataset: dict[str, dict[str, list[float]]] = {}
    for p in paths:
        name = Path(p).stem
        per_method: dict[str, list] = defaultdict(list)
        for line in Path(p).read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            if "error" in r:
                continue
            per_method[r["method"]].append(r)
        by_dataset[name] = per_method
    return by_dataset


def per_dataset_table(name, per_method, metric, ref):
    print(f"\n### {name} -- {metric}")
    have = {m: np.array([r[metric] for r in rows]) for m, rows in per_method.items() if rows and metric in rows[0]}
    if ref not in have:
        print(f"  (reference {ref!r} absent; skipping paired tests)")
        ref = None
    others = [m for m in have if m != ref]
    raw_p = []
    for m in others:
        raw_p.append(stats.paired_comparison(have[ref], have[m]).wilcoxon_p if ref else float("nan"))
    adj = stats.holm_bonferroni([p for p in raw_p if p == p]) if ref else []
    adj_iter = iter(adj)
    for m in sorted(have, key=lambda k: -have[k].mean()):
        ci = stats.t_confidence_interval(have[m])
        mark = ""
        if ref and m != ref:
            a = next(adj_iter, float("nan"))
            mark = " *" if a < 0.05 else ""
        star = " (ref)" if m == ref else mark
        print(f"  {m:<28} {ci.mean:.3f} +/- {ci.half_width:.3f}{star}")


def cross_dataset(by_dataset, metric):
    import pandas as pd

    methods = sorted(set().union(*[set(pm) for pm in by_dataset.values()]))
    mat = {}
    for name, pm in by_dataset.items():
        mat[name] = {m: np.mean([r[metric] for r in pm[m]]) if pm.get(m) and metric in (pm[m][0] if pm[m] else {}) else np.nan
                     for m in methods}
    df = pd.DataFrame(mat).T
    if len(df) < 2:
        return
    fr = stats.friedman_test(df)
    cd = stats.nemenyi_critical_difference(k=fr.k_methods, n_blocks=fr.n_blocks)
    print(f"\n### Cross-dataset ({fr.n_blocks} datasets) -- {metric}")
    print(f"  Friedman chi2={fr.statistic:.2f}  p={stats.format_p(fr.p_value)}   Nemenyi CD={cd:.2f}")
    for m, r in fr.avg_ranks.items():
        print(f"  {m:<28} avg rank {r:.2f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("results", nargs="+")
    ap.add_argument("--metric", default="macro_f1")
    ap.add_argument("--ref", default="RFOED-NN")
    args = ap.parse_args()
    by_dataset = load_rows(args.results)
    for name, pm in by_dataset.items():
        per_dataset_table(name, pm, args.metric, args.ref)
    cross_dataset(by_dataset, args.metric)


if __name__ == "__main__":
    main()
