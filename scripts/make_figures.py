"""Regenerate the paper's figures from results/*.jsonl.

    python scripts/make_figures.py            # all
    python scripts/make_figures.py cd margin  # a subset

Figures:
  schematic         hand-drawn (kept in figures/, not generated here)
  rank_frequency    class/label frequency-rank curves, all datasets
  cd_diagram        Friedman + Nemenyi CD diagram, single-label macro-F1
                    (freqcascade.stats.plot_cd_diagram)
  margin_vs_K       RFOED macro-F1 margin over the best baseline vs class count
  focc_ablation     FOCC 2x2 ordering x rebalance bars, per multi-label dataset
  per_class_recall  rarest-quartile per-class recall, RFOED-NN vs best baseline
  representation    RQ5: margin over best baseline per frozen encoder

Each function takes the parsed results and writes a PDF into figures/. This
file currently wires up the two that need only the dataset objects
(rank_frequency) or freqcascade.stats (cd_diagram); the results-driven ones
are filled in once the benchmark run is complete.
"""

from __future__ import annotations

import sys
from pathlib import Path

FIG_DIR = Path(__file__).resolve().parent.parent / "figures"


def fig_rank_frequency() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    from freqcascade.datasets import REGISTRY, load

    fig, ax = plt.subplots(figsize=(7, 4.2))
    for name in REGISTRY:
        try:
            ds = load(name)
        except Exception as e:  # noqa: BLE001
            print(f"  skip {name}: {e}")
            continue
        freqs = np.sort(ds.class_frequencies)[::-1]
        ax.plot(np.arange(1, len(freqs) + 1), freqs, marker=".", ms=4, label=f"{name} (K={len(freqs)})")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("class / label rank"); ax.set_ylabel("training frequency")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    out = FIG_DIR / "figure2_rank_frequency.pdf"
    fig.savefig(out); print(f"  -> {out}")


def fig_cd_diagram() -> None:
    import json

    import pandas as pd

    from freqcascade import stats

    results = Path(__file__).resolve().parent.parent / "results"
    files = sorted(results.glob("*.jsonl"))
    if not files:
        print("  no results/*.jsonl yet -- run the benchmarks first")
        return
    rows = {}
    for f in files:
        per = {}
        for line in f.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            if "error" in r or "macro_f1" not in r:
                continue
            per.setdefault(r["method"], []).append(r["macro_f1"])
        if per:
            rows[f.stem] = {m: sum(v) / len(v) for m, v in per.items()}
    df = pd.DataFrame(rows).T.dropna(axis=1)
    if len(df) < 2:
        print("  need >= 2 datasets with shared methods for a CD diagram")
        return
    fr = stats.friedman_test(df)
    cd = stats.nemenyi_critical_difference(k=fr.k_methods, n_blocks=fr.n_blocks)
    stats.plot_cd_diagram(fr.avg_ranks, cd, FIG_DIR / "figure3_cd_diagram.pdf")
    print(f"  -> {FIG_DIR / 'figure3_cd_diagram.pdf'}")


FIGURES = {
    "rank_frequency": fig_rank_frequency,
    "cd_diagram": fig_cd_diagram,
    # "margin_vs_K", "focc_ablation", "per_class_recall", "representation":
    # results-driven, added once the full run lands.
}


def main(which: list[str]) -> None:
    FIG_DIR.mkdir(exist_ok=True)
    for key in (which or FIGURES):
        fn = FIGURES.get(key)
        if fn is None:
            print(f"skip unknown/not-yet-wired figure: {key}")
            continue
        print(f"[{key}]")
        fn()


if __name__ == "__main__":
    main(sys.argv[1:])
