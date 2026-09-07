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
    sl = ["clinc150", "20newsgroups_ir50", "wos46985", "ohsumed_23", "drug_reviews"]
    files = [results / f"{n}.jsonl" for n in sl]
    rows = {}
    for f in files:
        if not f.exists():
            continue
        per = {}
        for line in f.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if "error" in r or "macro_f1" not in r or "method" not in r:
                continue
            per.setdefault(r["method"], []).append(r["macro_f1"])
        if per:
            rows[f.stem] = {m: sum(v) / len(v) for m, v in per.items()}
    if not rows:
        print("  no single-label benchmark results found")
        return
    df = pd.DataFrame(rows).T.dropna(axis=1)
    if len(df) < 2:
        print("  need >= 2 datasets with shared methods for a CD diagram")
        return
    fr = stats.friedman_test(df)
    cd = stats.nemenyi_critical_difference(k=fr.k_methods, n_blocks=fr.n_blocks)
    stats.plot_cd_diagram(fr.avg_ranks, cd, FIG_DIR / "figure3_cd_diagram.pdf")
    print(f"  -> {FIG_DIR / 'figure3_cd_diagram.pdf'}")


RESULTS = Path(__file__).resolve().parent.parent / "results"
_CLASS_COUNT = {
    "clinc150": 150, "20newsgroups_ir50": 20, "wos46985": 134,
    "ohsumed_23": 23, "drug_reviews": 356,
}


def _load_jsonl(path: Path) -> list[dict]:
    import json
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def _mean_by_method(rows: list[dict], metric: str) -> dict[str, float]:
    import numpy as np
    out: dict[str, list] = {}
    for r in rows:
        if metric in r and "error" not in r:
            out.setdefault(r["method"], []).append(r[metric])
    return {m: float(np.mean(v)) for m, v in out.items()}


def fig_margin_vs_k() -> None:
    """RFOED's macro-F1 margin over each dataset's best baseline, vs class count."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pts = []
    for name, k in _CLASS_COUNT.items():
        means = _mean_by_method(_load_jsonl(RESULTS / f"{name}.jsonl"), "macro_f1")
        if not means:
            continue
        for proposed in ("RFOED-NN", "RFOED-RF"):
            if proposed not in means:
                continue
            baselines = {m: v for m, v in means.items() if not m.startswith("RFOED")}
            if not baselines:
                continue
            pts.append((k, means[proposed] - max(baselines.values()), proposed, name))
    if not pts:
        print("  no single-label results yet")
        return
    fig, ax = plt.subplots(figsize=(6.4, 4))
    for tag, marker in (("RFOED-NN", "o"), ("RFOED-RF", "s")):
        sub = sorted((k, d, n) for k, d, t, n in pts if t == tag)
        if sub:
            ax.plot([k for k, _, _ in sub], [d for _, d, _ in sub], marker=marker, label=tag)
            for k, d, n in sub:
                ax.annotate(n, (k, d), fontsize=7, xytext=(4, 4), textcoords="offset points")
    ax.axhline(0, color="#888", lw=0.8, ls="--")
    ax.set_xlabel("number of classes K"); ax.set_ylabel("macro-F1 margin over best baseline")
    ax.legend()
    fig.tight_layout()
    out = FIG_DIR / "figure4_margin_vs_classcount.pdf"
    fig.savefig(out); print(f"  -> {out}")


def fig_focc_ablation() -> None:
    """FOCC 2x2 ordering x rebalance, label-macro-F1, one group of bars per dataset."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    files = sorted(RESULTS.glob("ablation_*.jsonl"))
    ml = [(f.stem.replace("ablation_", ""), _load_jsonl(f)) for f in files]
    ml = [(n, rows) for n, rows in ml if rows and "label_macro_f1" in rows[0]]
    if not ml:
        print("  no FOCC ablation results yet")
        return
    cells = [("frequency", True), ("frequency", False), ("random", True), ("random", False)]
    labels = ["freq / reb", "freq / -", "rand / reb", "rand / -"]
    fig, ax = plt.subplots(figsize=(2 + 1.8 * len(ml), 4))
    width = 0.19
    for ci, (order, reb) in enumerate(cells):
        vals, errs = [], []
        for _, rows in ml:
            v = [r["label_macro_f1"] for r in rows if r["ordering"] == order and r["rebalance"] == reb]
            vals.append(np.mean(v) if v else 0.0)
            errs.append(np.std(v) if v else 0.0)
        x = np.arange(len(ml)) + (ci - 1.5) * width
        ax.bar(x, vals, width, yerr=errs, capsize=2, label=labels[ci])
    ax.set_xticks(np.arange(len(ml))); ax.set_xticklabels([n for n, _ in ml])
    ax.set_ylabel("label-macro-F1"); ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    out = FIG_DIR / "figure5_focc_ablation.pdf"
    fig.savefig(out); print(f"  -> {out}")


def fig_representation() -> None:
    """RQ5: proposed-method margin over the strongest baseline, per frozen encoder."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    files = sorted(RESULTS.glob("rq5_*.jsonl"))
    data = {f.stem.replace("rq5_", ""): _load_jsonl(f) for f in files}
    data = {k: v for k, v in data.items() if v}
    if not data:
        print("  no RQ5 results yet")
        return
    encoders = sorted({r.get("encoder") for rows in data.values() for r in rows if r.get("encoder") and r["encoder"] != "tfidf"})
    fig, ax = plt.subplots(figsize=(2 + 1.6 * len(data), 4))
    width = 0.8 / max(len(encoders), 1)
    for ei, enc in enumerate(encoders):
        margins = []
        for name, rows in data.items():
            metric = "label_macro_f1" if "label_macro_f1" in rows[0] else "macro_f1"
            prop = [r[metric] for r in rows if r.get("encoder") == enc and r["method"].startswith(("RFOED", "FOCC"))]
            base = [r[metric] for r in rows if r["method"].endswith("-RF") and not r["method"].startswith(("RFOED", "FOCC"))]
            margins.append((np.mean(prop) - np.mean(base)) if prop and base else 0.0)
        x = np.arange(len(data)) + (ei - (len(encoders) - 1) / 2) * width
        ax.bar(x, margins, width, label=enc)
    ax.axhline(0, color="#888", lw=0.8, ls="--")
    ax.set_xticks(np.arange(len(data))); ax.set_xticklabels(list(data))
    ax.set_ylabel("margin over best baseline"); ax.legend(fontsize=8)
    fig.tight_layout()
    out = FIG_DIR / "figure7_representation.pdf"
    fig.savefig(out); print(f"  -> {out}")


def fig_per_class_recall() -> None:
    """Rarest-quartile per-class recall, proposed vs best baseline. Needs
    results/perclass_<dataset>.jsonl from scripts/dump_per_class_recall.py."""
    files = sorted(RESULTS.glob("perclass_*.jsonl"))
    if not files:
        print("  no per-class recall dumps yet (run scripts/dump_per_class_recall.py)")
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    for f in files:
        rows = _load_jsonl(f)
        by_method: dict[str, list] = {}
        for r in rows:
            by_method.setdefault(r["method"], []).append(r)
        fig, ax = plt.subplots(figsize=(7, 4))
        for m, rr in by_method.items():
            rr = sorted(rr, key=lambda x: x["train_count"])[: max(1, len(rr) // 4)]
            ax.plot(range(len(rr)), [x["recall"] for x in rr], marker=".", label=f"{m} (mean {np.mean([x['recall'] for x in rr]):.2f})")
        ax.set_xlabel("rarest classes (by train frequency)"); ax.set_ylabel("test recall"); ax.legend(fontsize=8)
        fig.tight_layout()
        out = FIG_DIR / f"figure6_per_class_recall_{f.stem.replace('perclass_', '')}.pdf"
        fig.savefig(out); print(f"  -> {out}")


FIGURES = {
    "rank_frequency": fig_rank_frequency,
    "cd_diagram": fig_cd_diagram,
    "margin_vs_k": fig_margin_vs_k,
    "focc_ablation": fig_focc_ablation,
    "representation": fig_representation,
    "per_class_recall": fig_per_class_recall,
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
