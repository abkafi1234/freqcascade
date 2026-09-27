"""Figures for the Knowledge-Based Systems version of the paper.

Everything is matplotlib: data-driven result plots (from results/*.jsonl), the
two synthetic concept plots, and the flow / schematic diagrams (RFOED and FOCC
flowcharts, local-vs-global rebalancing, batched-NN ensemble, the two failure
mechanisms, and the practitioner decision tree) -- so the whole figure set is
reproduced by one script with one palette and no TikZ/LaTeX dependency.

    python scripts/kbs_figures.py            # all
    python scripts/kbs_figures.py landscape collapse   # a subset

Every figure is written to figures/ as a PDF at a fixed name.
A consistent, colour-vision-safe palette (Okabe-Ito) is shared across figures;
each method keeps the same colour everywhere it appears.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

R = Path(__file__).resolve().parent.parent / "results"
FIG = Path(__file__).resolve().parent.parent / "figures"

# --- Okabe-Ito, colour-vision-deficiency safe -------------------------------
OI = {
    "black": "#000000", "orange": "#E69F00", "sky": "#56B4E9", "green": "#009E73",
    "yellow": "#F0E442", "blue": "#0072B2", "vermillion": "#D55E00", "purple": "#CC79A7",
    "grey": "#8c8c8c",
}
# method -> (colour, marker) held fixed across every figure
METHOD_STYLE = {
    "RFOED-NN": (OI["green"], "o"),
    "RFOED-NN (capped)": (OI["green"], "o"),
    "FOCC-NN": (OI["green"], "o"),
    "RFOED-RF": (OI["sky"], "^"),
    "FOCC-RF": (OI["sky"], "^"),
    "Flat-RF+SMOTE": (OI["blue"], "s"),
    "Flat Balanced-RF": (OI["purple"], "D"),
    "OVR-RF": (OI["grey"], "v"),
    "Binary Relevance-RF": (OI["blue"], "s"),
    "Classifier Chains-RF": (OI["grey"], "v"),
    "EasyEnsemble": (OI["orange"], "X"),
    "RUSBoost": (OI["vermillion"], "P"),
}


def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 9,
        "axes.titlesize": 9,
        "axes.labelsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.5,
        "legend.frameon": False,
        "legend.fontsize": 8,
        "figure.dpi": 200,
    })
    return plt


def _rows(name):
    p = R / f"{name}.jsonl"
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def _agg(name, metric):
    d = defaultdict(list)
    err = set()
    for r in _rows(name):
        if "error" in r:
            err.add(r.get("method"))
        elif metric in r and "method" in r:
            d[r["method"]].append(r[metric])
    return {m: np.array(v) for m, v in d.items()}, err


def _ci(v):
    v = np.asarray(v, float)
    if len(v) < 2:
        return float(v.mean()), 0.0
    from freqcascade import stats
    c = stats.t_confidence_interval(v)
    return c.mean, (0.0 if np.isnan(c.half_width) else c.half_width)


def _style(m):
    return METHOD_STYLE.get(m, (OI["grey"], "o"))


# ======================================================================= #
# 1. Dataset landscape: class count vs imbalance ratio
# ======================================================================= #
DATASETS = [
    # name, K, IR, track, domain
    ("CLINC150", 150, 4.0, "single", "general"),
    ("20 Newsgroups", 20, 50.0, "single", "general"),
    ("WOS46985", 134, 17.5, "single", "scientific"),
    ("OHSUMED-23", 23, 29.3, "single", "biomedical"),
    ("Drug Reviews", 356, 1830.0, "single", "biomedical"),
    ("Reuters-21578", 90, 1982.0, "multi", "general"),
    ("Hallmarks of Cancer", 10, 4.4, "multi", "biomedical"),
    ("LitCovid", 7, 16.8, "multi", "biomedical"),
]


def fig_landscape():
    plt = _mpl()
    from matplotlib.lines import Line2D
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    dom_col = {"general": OI["blue"], "scientific": OI["grey"], "biomedical": OI["vermillion"]}
    off = {  # per-point label offset in points (dx, dy) + alignment
        "CLINC150": (9, 0, "left"), "20 Newsgroups": (0, 10, "center"),
        "WOS46985": (9, 0, "left"), "OHSUMED-23": (0, -12, "center"),
        "Drug Reviews": (-9, 6, "right"), "Reuters-21578": (0, 11, "center"),
        "Hallmarks of Cancer": (10, 0, "left"), "LitCovid": (10, 0, "left"),
    }
    for name, K, IR, track, dom in DATASETS:
        mk = "o" if track == "single" else "s"
        ax.scatter(K, IR, s=110, marker=mk, facecolor=dom_col[dom], edgecolor="black",
                   linewidth=0.7, zorder=3)
        dx, dy, ha = off[name]
        ax.annotate(name, (K, IR), (dx, dy), textcoords="offset points",
                    fontsize=7.5, ha=ha, va="center")
    ax.axvspan(100, 500, color="black", alpha=0.05, zorder=0)
    ax.text(220, 3.1, "$K>100$", fontsize=7.5, color="#555", ha="center")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("number of classes / labels $K$")
    ax.set_ylabel("imbalance ratio (max / min class frequency)")
    ax.set_xlim(4.5, 620); ax.set_ylim(2.5, 4000)
    leg = [Line2D([], [], marker="o", ls="", mfc="0.6", mec="k", label="single-label"),
           Line2D([], [], marker="s", ls="", mfc="0.6", mec="k", label="multi-label"),
           Line2D([], [], marker="o", ls="", mfc=dom_col["general"], mec="k", label="general"),
           Line2D([], [], marker="o", ls="", mfc=dom_col["scientific"], mec="k", label="scientific"),
           Line2D([], [], marker="o", ls="", mfc=dom_col["biomedical"], mec="k", label="biomedical")]
    ax.legend(handles=leg, loc="upper left", fontsize=7.5, handletextpad=0.4,
              borderpad=0.6, labelspacing=0.35)
    ax.set_title("The eight benchmarks span an order of magnitude of class count "
                 "and three of imbalance ratio", fontsize=8)
    fig.tight_layout()
    out = FIG / "figure_landscape.pdf"
    fig.savefig(out); print(f"  -> {out}")


# ======================================================================= #
# 2. Collapse bars: EE / RUSBoost / best flat baseline / RFOED-NN at native K
# ======================================================================= #
def fig_collapse_bars():
    plt = _mpl()
    order = [("clinc150", "CLINC150", 150), ("20newsgroups_ir50", "20 Newsgroups", 20),
             ("ohsumed_23", "OHSUMED-23", 23), ("wos46985", "WOS46985", 134),
             ("drug_reviews", "Drug Reviews", 356)]
    order.sort(key=lambda t: t[2])
    fig, ax = plt.subplots(figsize=(6.6, 3.7))
    groups = ["best flat baseline", "RFOED-NN", "EasyEnsemble", "RUSBoost"]
    gcol = {"best flat baseline": OI["blue"], "RFOED-NN": OI["green"],
            "EasyEnsemble": OI["orange"], "RUSBoost": OI["vermillion"]}
    W = 0.2
    x = np.arange(len(order))
    for gi, g in enumerate(groups):
        xx = x + (gi - 1.5) * W
        for xi, (name, _, K) in zip(xx, order):
            a, err = _agg(name, "macro_f1")
            if g == "best flat baseline":
                flats = [np.mean(v) for m, v in a.items() if m.startswith(("Flat", "OVR"))]
                val, dnf = (max(flats) if flats else 0.0), False
            elif g == "RFOED-NN":
                v = a.get("RFOED-NN", a.get("RFOED-NN (capped)"))
                val, dnf = (float(np.mean(v)) if v is not None else 0.0), False
            else:
                val = float(np.mean(a[g])) if g in a else 0.0
                dnf = g in err
            ax.bar(xi, val, W, color=gcol[g], edgecolor="black", linewidth=0.4,
                   label=g if xi == xx[0] else None)
            if dnf:
                ax.plot(xi, 0.02, marker="x", color=OI["vermillion"], ms=6, mew=1.6)
                ax.annotate("did not fit", (xi, 0.03), rotation=90, fontsize=6,
                            ha="center", va="bottom", color=OI["vermillion"])
            elif g in ("EasyEnsemble", "RUSBoost") and val < 0.02:
                ax.annotate(f"{val:.3f}", (xi, val), (xi, 0.04), rotation=90, fontsize=6,
                            ha="center", va="bottom", color=gcol[g])
    ax.set_xticks(x)
    ax.set_xticklabels([f"{lbl}\n$K$={K}" for _, lbl, K in order], fontsize=7.5)
    ax.set_ylabel("macro-F1")
    ax.set_ylim(0, 1.0)
    ax.legend(ncol=2, loc="upper center", fontsize=7.5)
    ax.set_title("At native class count: resampling ensembles collapse or fail to fit "
                 "once $K>100$", fontsize=8.5)
    ax.grid(axis="x", alpha=0)
    fig.tight_layout()
    out = FIG / "figure_collapse_bars.pdf"
    fig.savefig(out); print(f"  -> {out}")


# ======================================================================= #
# 3 & 4. Main results as dot + 95% CI (single- and multi-label)
# ======================================================================= #
def _dotci_panel(ax, name, metric, ref, title, dnf_note=True):
    a, err = _agg(name, metric)
    if not a:
        ax.set_visible(False); return
    order = sorted(a, key=lambda m: np.mean(a[m]))
    ys = np.arange(len(order))
    for y, m in zip(ys, order):
        mean, hw = _ci(a[m])
        col, mk = _style(m)
        is_ref = (m == ref)
        ax.errorbar(mean, y, xerr=hw, fmt=mk, ms=6.5 if is_ref else 4.5,
                    color=col, ecolor=col, elinewidth=1.4, capsize=2,
                    mec="black" if is_ref else col, mew=0.9 if is_ref else 0.5, zorder=3)
    ax.set_yticks(ys)
    ax.set_yticklabels(order, fontsize=7)
    for tick, m in zip(ax.get_yticklabels(), order):
        if m == ref:
            tick.set_fontweight("bold")
    if err and dnf_note:
        title = title + "\n(" + ", ".join(sorted(err)) + " did not fit)"
    ax.set_title(title, fontsize=8)
    ax.set_xlim(-0.02, min(1.0, max(np.mean(a[m]) for m in a) + 0.12))
    ax.grid(axis="x", alpha=0.25)
    ax.grid(axis="y", alpha=0.0)


def fig_mainresults_sl():
    plt = _mpl()
    panels = [("clinc150", "CLINC150 ($K$=150, well separated)"),
              ("20newsgroups_ir50", "20 Newsgroups ($K$=20)"),
              ("wos46985", "WOS46985 ($K$=134, fine-grained)"),
              ("ohsumed_23", "OHSUMED-23 (biomedical)"),
              ("drug_reviews", "Drug Reviews ($K\\approx$356, biomedical)")]
    fig, axes = plt.subplots(3, 2, figsize=(7.2, 7.6))
    axes = axes.ravel()
    refs = {"drug_reviews": "RFOED-NN (capped)"}
    for ax, (name, title) in zip(axes, panels):
        _dotci_panel(ax, name, "macro_f1", refs.get(name, "RFOED-NN"), title)
    axes[-1].set_visible(False)
    fig.supxlabel("macro-F1  (mean $\\pm$ 95% CI over $5{\\times}2$ folds / 10 seeds; "
                  "RFOED highlighted)", fontsize=8)
    fig.tight_layout()
    out = FIG / "figure_mainresults_sl.pdf"
    fig.savefig(out); print(f"  -> {out}")


def fig_mainresults_ml():
    plt = _mpl()
    panels = [("reuters21578", "Reuters-21578 ($L$=90, general)"),
              ("hoc", "Hallmarks of Cancer (biomedical)"),
              ("litcovid", "LitCovid (biomedical)")]
    fig, axes = plt.subplots(1, 3, figsize=(8.4, 3.1))
    for ax, (name, title) in zip(axes, panels):
        _dotci_panel(ax, name, "label_macro_f1", "FOCC-NN", title, dnf_note=False)
    fig.supxlabel("label-macro-F1  (mean $\\pm$ 95% CI; FOCC-NN highlighted)", fontsize=8)
    fig.tight_layout()
    out = FIG / "figure_mainresults_ml.pdf"
    fig.savefig(out); print(f"  -> {out}")


# ======================================================================= #
# 5. Factorial effect-size heatmap
# ======================================================================= #
def fig_factorial_heatmap():
    import pandas as pd
    from freqcascade import stats
    plt = _mpl()
    dss = [("ablation_20newsgroups_ir50", "20 Newsgroups"),
           ("ablation_clinc150", "CLINC150"), ("ablation_wos46985", "WOS46985")]
    labels = [("C(ordering)", "Ordering"), ("C(base_learner)", "Base learner"),
              ("C(rebalance)", "Rebalance"),
              ("C(ordering):C(base_learner)", "Ord×Base"),
              ("C(ordering):C(rebalance)", "Ord×Reb"),
              ("C(base_learner):C(rebalance)", "Base×Reb"),
              ("C(ordering):C(base_learner):C(rebalance)", "Ord×Base×Reb")]
    M = np.full((len(labels), len(dss)), np.nan)
    P = np.empty_like(M, dtype=object)
    for j, (f, _) in enumerate(dss):
        df = pd.DataFrame(_rows(f))
        res = stats.anova_or_srh(df, "macro_f1", ("ordering", "base_learner", "rebalance"))
        t = res.anova.table
        for i, (key, _) in enumerate(labels):
            if key in t.index:
                M[i, j] = np.log10(max(t.loc[key, "F"], 1.0))
                p = t.loc[key, "PR(>F)"]
                P[i, j] = "***" if p < 1e-3 else ("**" if p < 1e-2 else ("*" if p < 0.05 else "n.s."))
    fig, ax = plt.subplots(figsize=(4.6, 4.0))
    im = ax.imshow(M, cmap="Blues", aspect="auto")
    ax.set_xticks(range(len(dss))); ax.set_xticklabels([d for _, d in dss], fontsize=8)
    ax.set_yticks(range(len(labels))); ax.set_yticklabels([l for _, l in labels], fontsize=8)
    for i in range(len(labels)):
        for j in range(len(dss)):
            if not np.isnan(M[i, j]):
                ax.text(j, i, P[i, j], ha="center", va="center", fontsize=7,
                        color="white" if M[i, j] > 1.8 else "black")
    ax.set_title("Effect size ($\\log_{10}F$) and significance\nfrom the $2^3$ factorial ablation",
                 fontsize=8.5)
    cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
    cb.set_label("$\\log_{10} F$", fontsize=8)
    ax.grid(False)
    fig.tight_layout()
    out = FIG / "figure_factorial_heatmap.pdf"
    fig.savefig(out); print(f"  -> {out}")


# ======================================================================= #
# 6. WOS46985 remediation sweep
# ======================================================================= #
def fig_remediation_sweep():
    plt = _mpl()
    rows = _rows("wos46985_sweep")
    by = {(r["base"], r["config"]): r["macro_f1"] for r in rows}
    steps = [("plain", "plain"), ("threshold_only", "+ threshold"),
             ("cap10+thr", "+ cap 10"), ("cap15+thr", "+ cap 15"),
             ("cap20+thr", "+ cap 20"), ("cap30+thr", "+ cap 30"),
             ("cap20+thr+morereg", "+ more reg."),
             ("hier_plain", "hierarchical"), ("hier+thr", "hier. + thr.")]
    x = np.arange(len(steps))
    fig, ax = plt.subplots(figsize=(6.0, 3.4))
    for base, lbl, col, mk in [("rf", "RFOED-RF", OI["sky"], "^"),
                               ("nn", "RFOED-NN", OI["green"], "o")]:
        y = [by.get((base, k), np.nan) for k, _ in steps]
        ax.plot(x, y, mk + "-", color=col, label=lbl, lw=1.6, ms=5)
    ax.axhline(0.781, color=OI["blue"], ls="--", lw=1.2, label="best baseline (SMOTE)")
    ax.axvspan(1.5, 5.5, color="black", alpha=0.05)
    ax.text(3.5, 0.72, "plateau", ha="center", fontsize=7.5, style="italic")
    ax.set_xticks(x); ax.set_xticklabels([l for _, l in steps], rotation=35, ha="right", fontsize=7.5)
    ax.set_ylabel("macro-F1 (fixed 80/20 diagnostic split)")
    ax.set_ylim(0.2, 0.85)
    ax.legend(loc="lower right")
    ax.set_title("Threshold correction is the lever; structure changes do not help", fontsize=8.5)
    fig.tight_layout()
    out = FIG / "figure_remediation_sweep.pdf"
    fig.savefig(out); print(f"  -> {out}")


# ======================================================================= #
# 7. FOCC exposure bias vs chain length
# ======================================================================= #
def fig_exposure_bias():
    plt = _mpl()
    rows = _rows("focc_exposure")
    agg = defaultdict(lambda: defaultdict(list))
    for r in rows:
        agg[r["dataset"]]["pred"].append(r["label_macro_f1_predicted"])
        agg[r["dataset"]]["orac"].append(r["label_macro_f1_oracle"])
        agg[r["dataset"]]["gap"].append(r["exposure_gap"])
    meta = [("hoc", "HoC", 10), ("litcovid", "LitCovid", 7), ("reuters21578", "Reuters-21578", 90)]
    meta.sort(key=lambda t: t[2])
    fig, ax = plt.subplots(figsize=(5.4, 3.3))
    x = np.arange(len(meta)); W = 0.36
    p = [np.mean(agg[d]["pred"]) for d, _, _ in meta]
    o = [np.mean(agg[d]["orac"]) for d, _, _ in meta]
    ax.bar(x - W / 2, p, W, color=OI["green"], label="deployed (predicted earlier labels)",
           edgecolor="black", linewidth=0.4)
    ax.bar(x + W / 2, o, W, color=OI["sky"], label="oracle (true earlier labels)",
           edgecolor="black", linewidth=0.4)
    for i, (d, _, _) in enumerate(meta):
        g = np.mean(agg[d]["gap"])
        ax.annotate(f"gap {g:+.3f}", (i, max(p[i], o[i]) + 0.03), ha="center",
                    fontsize=7.5, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{nm}\n$L$={L} labels" for _, nm, L in meta], fontsize=8)
    ax.set_ylabel("FOCC-NN label-macro-F1")
    ax.set_ylim(0, 1.12)
    ax.legend(fontsize=7.5, loc="upper right", borderpad=0.6)
    ax.set_title("Exposure bias: the price of predicting earlier links\n"
                 "is large on the 90-label chain, modest on the short ones", fontsize=8)
    ax.grid(axis="x", alpha=0)
    fig.tight_layout()
    out = FIG / "figure_exposure_bias.pdf"
    fig.savefig(out, bbox_inches="tight"); print(f"  -> {out}")


def fig_perlink_exposure(dataset="reuters21578", order="frequency"):
    """Per-link precision, recall and false-positive rate down the chain,
    deployed (each link conditions on predicted earlier labels) against oracle
    (true earlier labels). This is the propagation plot the review asked for:
    the aggregate gap in fig_exposure_bias says the 90-label chain pays a price,
    but not where along the chain it is paid."""
    plt = _mpl()
    rows = [r for r in _rows("focc_perlink")
            if r["dataset"] == dataset and r["order"] == order]
    if not rows:
        print("  (no focc_perlink rows for %s/%s yet)" % (dataset, order))
        return
    L = rows[0]["n_links"]
    folds = sorted({r["fold"] for r in rows})
    by = defaultdict(list)
    for r in rows:
        by[r["rank"]].append(r)
    ranks = [k for k in sorted(by) if len(by[k])]

    def series(kind, metric):
        return np.array([np.mean([x[kind][metric] for x in by[k]]) for k in ranks])

    def smooth(v, w=9):
        if len(v) < w:
            return v
        pad = np.r_[np.repeat(v[0], w // 2), v, np.repeat(v[-1], w // 2)]
        return np.convolve(pad, np.ones(w) / w, mode="valid")

    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.5), sharex=True)
    for ax, metric, label in zip(axes, ("precision", "recall", "fpr"),
                                 ("precision", "recall", "false-positive rate")):
        for kind, colour, name in (("oracle", OI["sky"], "oracle (true earlier labels)"),
                                   ("deployed", OI["vermillion"], "deployed (predicted)")):
            v = series(kind, metric)
            ax.plot(ranks, v, color=colour, lw=0.6, alpha=0.35)
            ax.plot(ranks, smooth(v), color=colour, lw=1.8, label=name)
        ax.set_xlabel("link position in chain")
        ax.set_title(label, fontsize=8.5)
        ax.set_ylim(-0.02, 1.02)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("per-link score")
    up = np.array([np.mean([x["upstream_err_mean"] for x in by[k]]) for k in ranks])
    ax2 = axes[2].twinx()
    ax2.plot(ranks, smooth(up), color=OI["grey"], lw=1.2, ls="--")
    ax2.set_ylabel("wrong upstream labels\nper document (mean)", fontsize=7, color=OI["grey"])
    ax2.tick_params(axis="y", labelsize=7, colors=OI["grey"])
    ax2.set_ylim(bottom=0)
    axes[1].legend(fontsize=6.8, loc="lower left", framealpha=0.9)
    gap = float(np.mean(series("oracle", "f1") - series("deployed", "f1")))
    pretty = {"reuters21578": "Reuters-21578", "hoc": "HoC", "litcovid": "LitCovid"}
    fig.suptitle("%s, $L$=%d links, %s order: the deployed chain loses ground to the oracle "
                 "as upstream error accumulates\n(mean per-link F1 gap %+.3f over %d folds; "
                 "dashed grey: mean number of wrong upstream labels fed into the link)"
                 % (pretty.get(dataset, dataset), L, order, gap, len(folds)), fontsize=7.6)
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    out = FIG / ("figure_perlink_exposure_%s.pdf" % dataset)
    fig.savefig(out, bbox_inches="tight"); print(f"  -> {out}")


# ======================================================================= #
# 8. Separability, not K
# ======================================================================= #
_LOADER = {"20newsgroups_ir50": "20newsgroups"}


def _silhouette(name):
    """Mean silhouette of the MiniLM embeddings under the true labels
    (subsampled for tractability). `name` is the results-file stem."""
    from sklearn.metrics import silhouette_score

    from freqcascade.datasets import load
    cache = Path(__file__).resolve().parent.parent / "data" / "cache" / "embeddings" / f"{name}.minilm.npy"
    X = np.load(cache)
    ds = load(_LOADER.get(name, name))
    y = np.asarray(ds.target)
    if len(y) != len(X):
        return np.nan
    if y.ndim > 1:  # multi-label: skip
        return np.nan
    rng = np.random.default_rng(0)
    idx = rng.choice(len(y), size=min(4000, len(y)), replace=False)
    Xs, ys = X[idx], y[idx]
    keep = np.array([np.sum(ys == c) >= 2 for c in ys])
    Xs, ys = Xs[keep], ys[keep]
    if len(np.unique(ys)) < 2:
        return np.nan
    return float(silhouette_score(Xs, ys, metric="cosine"))


def _separability_terms():
    """Per single-label corpus: the RFOED-NN margin over the best RF-family flat
    baseline, split into a learner term (best Flat-MLP minus best RF-family
    baseline) and a decomposition term (RFOED-NN minus best Flat-MLP, paired by
    fold).  Returns rows (label, K, silhouette, learner, decomp, decomp_ci, n_neg, n)."""
    spec = [("clinc150", "CLINC150", 150), ("20newsgroups_ir50", "20 Newsgroups", 20),
            ("ohsumed_23", "OHSUMED-23", 23), ("drug_reviews", "Drug Reviews", 356),
            ("wos46985", "WOS46985", 134)]
    out = []
    for name, lbl, K in spec:
        folds = defaultdict(dict)
        for r in _rows(name):
            if "error" not in r and "macro_f1" in r and "method" in r:
                folds[r["method"]][r["fold"]] = r["macro_f1"]
        prop = folds.get("RFOED-NN") or folds.get("RFOED-NN (capped)")
        mean = {m: np.mean(list(v.values())) for m, v in folds.items()}
        mlp = max((m for m in mean if m.startswith("Flat-MLP")), key=mean.get)
        rf = max((m for m in mean if m.startswith(("Flat", "OVR")) and "MLP" not in m), key=mean.get)
        common = sorted(set(prop) & set(folds[mlp]))
        d = np.array([prop[f] - folds[mlp][f] for f in common])
        m, h = _ci(d)
        out.append((lbl, K, _silhouette(name), mean[mlp] - mean[rf], m, h,
                    int((d < 0).sum()), len(d), mlp, rf, np.mean([prop[f] for f in common]) - mean[rf]))
    return out


def fig_separability():
    """Left: the RFOED-NN margin over the best RF-family flat baseline split into a
    learner term and a decomposition term.  Right: the decomposition term against
    class separability (mean silhouette of the frozen MiniLM embedding)."""
    plt = _mpl()
    rows = _separability_terms()
    for r in rows:
        print("  %-14s K=%3d sil=%+.3f learner=%+.4f decomp=%+.4f +-%.4f (%d/%d folds <0) net=%+.4f  [%s vs %s]"
              % (r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7], r[10], r[8], r[9]))
    fig, (axl, axr) = plt.subplots(1, 2, figsize=(7.1, 3.3), gridspec_kw=dict(width_ratios=[1.25, 1]))
    y = np.arange(len(rows))[::-1]
    hgt = 0.36
    axl.barh(y + hgt / 2, [r[3] for r in rows], hgt, color=OI["sky"], edgecolor="white", linewidth=1,
             label="representation term (Flat-MLP on embeddings $-$ best TF-IDF RF)")
    axl.barh(y - hgt / 2, [r[4] for r in rows], hgt, color=OI["vermillion"], edgecolor="white", linewidth=1,
             xerr=[r[5] for r in rows], error_kw=dict(lw=0.8, capsize=2),
             label="decomposition term (RFOED-NN $-$ Flat-MLP)")
    axl.scatter([r[10] for r in rows], y, marker="D", s=30, color="black", zorder=3,
                label="net margin (RFOED-NN $-$ best TF-IDF RF)")
    axl.axvline(0, color="black", lw=0.8)
    axl.set_yticks(y)
    axl.set_yticklabels([f"{r[0]} ($K{{=}}{r[1]}$)" for r in rows], fontsize=7.5)
    axl.set_xlabel("macro-F1 difference")
    fig.legend(*axl.get_legend_handles_labels(), fontsize=6.8, loc="upper center", ncol=3,
               frameon=False, bbox_to_anchor=(0.5, 1.07))
    axl.set_title("(a) margin over best TF-IDF RF baseline, split", fontsize=8.5)

    off = {"CLINC150": (0, 9), "20 Newsgroups": (0, 9), "OHSUMED-23": (34, 2),
           "Drug Reviews": (14, 9), "WOS46985": (24, -12)}
    for r in rows:
        axr.errorbar(r[2], r[4], yerr=r[5], fmt="o", ms=7, color=OI["vermillion"],
                     mec="black", mew=0.6, ecolor="black", elinewidth=0.8, capsize=2, zorder=3)
        axr.annotate(r[0], (r[2], r[4]), off[r[0]], textcoords="offset points", fontsize=7, ha="center")
    axr.axhline(0, color="black", lw=0.8, ls="--")
    axr.set_xlabel("class separability\n(mean silhouette, frozen MiniLM embedding)")
    axr.set_ylabel("decomposition term\n(paired over folds, 95% CI)")
    axr.set_title("(b) decomposition term vs separability", fontsize=8.5)
    axr.set_xlim(-0.12, 0.25)
    fig.tight_layout()
    out = FIG / "figure_separability.pdf"
    fig.savefig(out, bbox_inches="tight"); print(f"  -> {out}")


# ======================================================================= #
# 9. Neighbourhood contamination as K grows (concept, synthetic)
# ======================================================================= #
def fig_neighbourhood():
    plt = _mpl()
    N = 900          # total sample size, held FIXED across panels
    KAPPA = 12        # fixed neighbourhood size (SMOTE default k)
    fig, axes = plt.subplots(1, 3, figsize=(7.8, 3.0))
    for ax, K in zip(axes, (6, 24, 90)):
        rng = np.random.default_rng(7)
        cen = rng.uniform(0.6, 9.4, size=(K, 2))
        per = N // K
        P = np.vstack([c + rng.normal(0, 0.42, size=(per, 2)) for c in cen])
        lab = np.repeat(np.arange(K), per)
        ax.scatter(P[:, 0], P[:, 1], s=4, c="#c8c8c8", linewidths=0, zorder=1)
        # seed = a point of class 0
        seed = P[lab == 0][0]
        d = np.hypot(*(P - seed).T)
        nn = np.argsort(d)[1:KAPPA + 1]
        rad = d[nn].max()
        ax.add_patch(plt.Circle(seed, rad, fill=False, color=OI["vermillion"], lw=1.5, zorder=3))
        same = nn[lab[nn] == 0]; foreign = nn[lab[nn] != 0]
        ax.scatter(P[same, 0], P[same, 1], s=22, c=OI["green"], linewidths=0, zorder=4,
                   label="same-class neighbour")
        ax.scatter(P[foreign, 0], P[foreign, 1], s=26, c=OI["vermillion"], marker="X",
                   linewidths=0, zorder=4, label="foreign-class neighbour")
        ax.scatter(*seed, s=70, marker="*", c="black", zorder=5)
        ax.set_title(f"$K={K}$  ($n/K\\approx{per}$ per class)\n"
                     f"{len(foreign)} of {KAPPA} nearest neighbours are foreign", fontsize=7.8)
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.set_aspect("equal")
    axes[0].legend(loc="lower left", fontsize=6.8)
    fig.suptitle("Fixed total sample $n$; a fixed $\\kappa$-neighbourhood captures more "
                 "foreign-class points as class count $K$ grows", fontsize=8.5, y=1.02)
    fig.tight_layout()
    out = FIG / "figure_neighbourhood.pdf"
    fig.savefig(out, bbox_inches="tight"); print(f"  -> {out}")


# ======================================================================= #
# 10. Calibration schematic: RF under-confident, NN over-confident
# ======================================================================= #
def fig_calibration():
    plt = _mpl()
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.0), sharey=True)
    xs = np.linspace(0, 1, 400)

    def gauss(m, s):
        return np.exp(-0.5 * ((xs - m) / s) ** 2)

    for ax in axes:
        ax.set_ylim(0, 1.42)
        ax.set_xlim(0, 1)
        ax.set_yticks([])
        ax.grid(False)
        ax.set_xlabel("node score $p$")

    # -- LEFT: RF under-confident -- positives drift low, both peaks below 0.5
    ax = axes[0]
    ax.fill_between(xs, 0.92 * gauss(0.15, 0.09), color=OI["grey"], alpha=0.5, lw=0)
    ax.fill_between(xs, 0.58 * gauss(0.34, 0.13), color=OI["green"], alpha=0.6, lw=0)
    ax.axvline(0.5, color="black", lw=1.4)
    ax.axvline(0.2, color=OI["vermillion"], lw=1.7, ls="--")
    ax.text(0.045, 0.52, "negatives", color="#555", fontsize=7.5, ha="left")
    ax.text(0.44, 0.34, "own-class\npositives", color=OI["green"], fontsize=7.5,
            ha="left", va="center", linespacing=1.2)
    ax.text(0.70, 1.30, "default $\\tau{=}0.5$:\nmisses most positives", fontsize=7,
            ha="center", va="top")
    ax.annotate("corrected $\\tau$", (0.2, 0.45), (0.30, 0.95), fontsize=7,
                color=OI["vermillion"], ha="left",
                arrowprops=dict(arrowstyle="->", color=OI["vermillion"], lw=1))
    ax.set_title("RFOED-RF: under-confident\n(test-time generalization gap)", fontsize=8)

    # -- RIGHT: NN over-confident -- negatives pushed above 0.5
    ax = axes[1]
    ax.fill_between(xs, 0.86 * gauss(0.33, 0.19), color=OI["grey"], alpha=0.5, lw=0)
    ax.fill_between(xs, 0.46 * gauss(0.82, 0.10), color=OI["green"], alpha=0.6, lw=0)
    ax.axvline(0.5, color="black", lw=1.4)
    ax.axvline(0.95, color=OI["vermillion"], lw=1.7, ls="--")
    ax.text(0.06, 1.12, "negatives", color="#555", fontsize=7.5, ha="left")
    ax.text(0.82, 0.52, "own-class\npositives", color=OI["green"], fontsize=7.5,
            ha="center", va="center", linespacing=1.2)
    ax.text(0.72, 1.30, "default $\\tau{=}0.5$\nfires on negatives",
            fontsize=7, ha="center", va="top")
    ax.annotate("corrected $\\tau$", (0.95, 0.45), (0.70, 0.95), fontsize=7,
                color=OI["vermillion"], ha="left",
                arrowprops=dict(arrowstyle="->", color=OI["vermillion"], lw=1))
    ax.set_title("RFOED-NN: over-confident\n(training-prior / threshold mismatch)", fontsize=8)

    fig.suptitle("Every node is trained at a balanced prior but deployed against a sub-1% node-local "
                 "prior;\nthe two base learners miscalibrate in opposite directions",
                 fontsize=8.3, y=1.13)
    fig.tight_layout()
    out = FIG / "figure_calibration.pdf"
    fig.savefig(out, bbox_inches="tight"); print(f"  -> {out}")


# ======================================================================= #
# Schematic / flow diagrams  (matplotlib replacements for the old TikZ art)
# ======================================================================= #
_INK = "#333333"
_FILL = "#f4f4f4"
_GREEN_F, _GREY_F, _SKY_F, _ORANGE_F, _VERM_F = (
    "#d9efe6", "#e9e9e9", "#e2f1fb", "#fbeccf", "#f7ddcc")


class _Node:
    __slots__ = ("cx", "cy", "w", "h")

    def __init__(self, cx, cy, w, h):
        self.cx, self.cy, self.w, self.h = cx, cy, w, h

    @property
    def N(self):
        return (self.cx, self.cy + self.h / 2)

    @property
    def S(self):
        return (self.cx, self.cy - self.h / 2)

    @property
    def E(self):
        return (self.cx + self.w / 2, self.cy)

    @property
    def W(self):
        return (self.cx - self.w / 2, self.cy)


def _canvas(w, h):
    """A blank drawing surface; coords are 0..100 in x and 0..(100 h/w) in y,
    so one unit is the same physical length on both axes.  The final crop and
    figure size are recomputed from the drawn content by ``_save``."""
    plt = _mpl()
    fig = plt.figure(figsize=(w, h))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100 * h / w)
    ax.axis("off")
    return plt, fig, ax


def _box(ax, cx, cy, w, h, text, *, fc=_FILL, ec=_INK, fs=8.5, tc="black",
         weight="normal", rounded=True, lw=1.0):
    from matplotlib.patches import FancyBboxPatch
    bs = "round,pad=0,rounding_size=1.0" if rounded else "square,pad=0"
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h, boxstyle=bs,
                                fc=fc, ec=ec, lw=lw, zorder=3))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fs, color=tc,
            zorder=4, fontweight=weight, linespacing=1.3)
    return _Node(cx, cy, w, h)


def _diamond(ax, cx, cy, w, h, text, *, fc=_SKY_F, ec=_INK, fs=8):
    from matplotlib.patches import Polygon
    ax.add_patch(Polygon([(cx, cy + h / 2), (cx + w / 2, cy),
                          (cx, cy - h / 2), (cx - w / 2, cy)],
                         closed=True, fc=fc, ec=ec, lw=1.0, zorder=3))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fs, zorder=4,
            linespacing=1.25)
    return _Node(cx, cy, w, h)


def _arrow(ax, p0, p1, *, color=_INK, lw=1.5, cs="arc3,rad=0"):
    from matplotlib.patches import FancyArrowPatch
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=13,
                                 color=color, lw=lw, zorder=2,
                                 connectionstyle=cs, shrinkA=1, shrinkB=1))


def _elabel(ax, x, y, s, *, color=_INK, fs=7.5, ha="center"):
    ax.text(x, y, s, ha=ha, va="center", fontsize=fs, color=color, zorder=5,
            bbox=dict(fc="white", ec="none", pad=0.4))


def _title(ax, x, y, s, *, fs=9, color="black"):
    ax.text(x, y, s, ha="center", va="center", fontsize=fs, fontweight="bold",
            color=color, linespacing=1.3, zorder=4)


def _save(fig, ax, name, maxdim=7.1):
    """Crop the axes to the drawn content and size the figure so one data unit
    is square, with the larger side == ``maxdim`` inches."""
    import matplotlib.pyplot as plt
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    inv = ax.transData.inverted()
    xs, ys = [], []
    for art in list(ax.texts) + list(ax.patches) + list(ax.lines):
        try:
            bb = art.get_window_extent(renderer=r)
        except TypeError:
            bb = art.get_window_extent()
        (ax0, ay0) = inv.transform((bb.x0, bb.y0))
        (ax1, ay1) = inv.transform((bb.x1, bb.y1))
        xs += [ax0, ax1]
        ys += [ay0, ay1]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    mx = (x1 - x0) * 0.015 + 0.6
    my = (y1 - y0) * 0.015 + 0.6
    x0, x1, y0, y1 = x0 - mx, x1 + mx, y0 - my, y1 + my
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    span_x, span_y = x1 - x0, y1 - y0
    if span_x >= span_y:
        fig.set_size_inches(maxdim, maxdim * span_y / span_x)
    else:
        fig.set_size_inches(maxdim * span_x / span_y, maxdim)
    out = FIG / f"{name}.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"  -> {out}")


# ----------------------------------------------------------------------- #
def fig_graphical_abstract():
    """Three data panels: (a) the collapse is class count x weak-learner capacity
    (R5 sweep, embedding features); (b) the single-label peel's decomposition
    term against a flat MLP on the same embedding; (c) FOCC-NN minus flat MLP on
    rarest-quartile label recall, at the flat MLP's default threshold and at a
    threshold tuned on validation data (results/threshold_flat_mlp.jsonl)."""
    plt = _mpl()
    fig, axs = plt.subplots(1, 3, figsize=(7.4, 2.6), gridspec_kw=dict(width_ratios=[1.15, 1, 1]))

    # (a) R5 capacity ladder
    ax = axs[0]
    g = defaultdict(list)
    for x in (R / "weaklearner_sweep.jsonl").read_text().splitlines():
        r = json.loads(x)
        if r.get("protocol") == "class-count sweep" and r.get("features") == "emb" and "macro_f1" in r:
            g[(r["ensemble"], r["learner"], r["K"])].append(r["macro_f1"])
    ks = [5, 10, 25, 50, 100, 150]
    # RUSBoost is included at two rungs because the settled headline distinguishes the two
    # ensembles by how much capacity each needs: EasyEnsemble recovers at depth 10, RUSBoost
    # only once the tree is grown without a depth limit.
    for (ens, ln), lab, col, ls in [(("EasyEnsemble", "stump"), "EasyEnsemble, stump", OI["vermillion"], "-"),
                                    (("EasyEnsemble", "tree10"), "EasyEnsemble, depth-10 tree", OI["orange"], "--"),
                                    (("RUSBoost", "tree10"), "RUSBoost, depth-10 tree", OI["purple"], ":"),
                                    (("RUSBoost", "treefull"), "RUSBoost, unbounded tree", OI["sky"], "-."),
                                    (("flat", "mlp"), "flat MLP, no resampling", OI["blue"], "-")]:
        pts = [(k, np.mean(g[(ens, ln, k)])) for k in ks if g.get((ens, ln, k))]
        ax.plot([k for k, _ in pts], [v for _, v in pts], ls, color=col, lw=1.6, marker="o", ms=3.0, label=lab)
    ax.set_xscale("log"); ax.set_xticks([5, 10, 25, 50, 150]); ax.set_xticklabels(["5", "10", "25", "50", "150"], fontsize=6.5)
    ax.xaxis.set_minor_locator(plt.NullLocator())
    ax.set_ylim(0, 1.02); ax.set_xlabel("class count $K$", fontsize=7.5); ax.set_ylabel("macro-F1", fontsize=7.5)
    ax.legend(fontsize=5.0, frameon=False, loc="lower left")
    ax.set_title("(a) the collapse is class count\n$\\times$ weak-learner capacity", fontsize=8)

    # (b) decomposition term, single-label
    ax = axs[1]
    rows = _separability_terms()
    y = np.arange(len(rows))[::-1]
    ax.barh(y, [r[4] for r in rows], 0.6, color=OI["vermillion"], xerr=[r[5] for r in rows],
            error_kw=dict(lw=0.7, capsize=1.5))
    ax.axvline(0, color="black", lw=0.8)
    ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows], fontsize=6.5)
    ax.set_xlabel("RFOED-NN $-$ flat MLP, macro-F1", fontsize=7)
    ax.set_title("(b) single-label: the peel\ncosts accuracy everywhere", fontsize=8)

    # (c) FOCC-NN minus flat MLP on rare-label recall
    ax = axs[2]
    names = [("reuters21578", "Reuters-21578"), ("litcovid", "LitCovid"), ("hoc", "HoC")]
    thr = defaultdict(dict)
    tp = R / "threshold_flat_mlp.jsonl"
    if tp.exists():
        for x in tp.read_text().splitlines():
            r = json.loads(x)
            thr[(r["dataset"], r["arm"])][r["fold"]] = r["bottom_quartile_label_recall"]
    y = np.arange(len(names))[::-1]
    for off, arm, lab, col in [(0.17, "0.5 (stored)", "flat MLP at $t{=}0.5$", OI["green"]),
                               (-0.17, "B per-label-tuned", "flat MLP, tuned $t$", OI["grey"])]:
        vals, errs = [], []
        for ds, _ in names:
            focc = {r["fold"]: r["bottom_quartile_label_recall"] for r in _rows(ds)
                    if r.get("method") == "FOCC-NN" and "error" not in r}
            other = thr[(ds, arm)] or ({r["fold"]: r["bottom_quartile_label_recall"] for r in _rows(ds)
                                        if r.get("method") == "Flat-MLP" and "error" not in r}
                                       if arm == "0.5 (stored)" else {})
            f = sorted(set(focc) & set(other))
            m, h = _ci([focc[i] - other[i] for i in f]) if f else (np.nan, 0)
            vals.append(m); errs.append(h)
        ax.barh(y + off, vals, 0.32, color=col, xerr=errs, error_kw=dict(lw=0.7, capsize=1.5), label=lab)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_yticks(y); ax.set_yticklabels([n for _, n in names], fontsize=6.5)
    ax.set_xlabel("FOCC-NN $-$ flat MLP,\nrarest-quartile label recall", fontsize=7)
    ax.set_xlim(-0.14, 0.75)
    ax.legend(fontsize=5.8, frameon=False, loc="lower right", handlelength=1.2)
    ax.set_title("(c) multi-label: the chain's\nrare-label gain vs threshold", fontsize=8)
    for a_ in axs:
        a_.tick_params(labelsize=6.5)
    fig.tight_layout(w_pad=1.2)
    out = FIG / "graphical_abstract.pdf"
    fig.savefig(out, bbox_inches="tight"); print(f"  -> {out}")


# ----------------------------------------------------------------------- #
def fig_rfoed_flow():
    plt, fig, ax = _canvas(5.2, 6.4)
    cx, bw, bh = 45, 54, 9
    ys = [116, 103, 90, 77, 64, 51]
    texts = [
        "Order classes by descending\ntraining frequency:  $c_1,\\dots,c_K$",
        "Active set $A \\leftarrow \\{1,\\dots,n\\}$;   $k \\leftarrow 1$",
        "$\\bf{Node}$ $\\bf{k}$:  binary target\n$y^{(k)}_i=\\mathbf{1}[y_i{=}c_k]$  for  $i\\in A$",
        "Balanced bootstrap of $A$\n(two strata, $\\leq 2n_{\\mathrm{cap}}$ draws)",
        "Fit base learner $f_k$",
        "Remove resolved:\n$A \\leftarrow A \\setminus \\{i : y_i = c_k\\}$",
    ]
    nodes = [_box(ax, cx, y, bw, bh, t, fs=8) for y, t in zip(ys, texts)]
    dia = _diamond(ax, cx, 39, 27, 12, "$k < K-1$ ?", fs=8)
    ret = _box(ax, cx, 24, 62, 9,
               "Return nodes  $(c_1,f_1),\\dots,(c_{K-1},f_{K-1})$", fs=8)
    for a, b in zip(nodes, nodes[1:]):
        _arrow(ax, a.S, b.N)
    _arrow(ax, nodes[-1].S, dia.N)
    _arrow(ax, dia.S, ret.N)
    _elabel(ax, cx + 3.5, 31.5, "no")
    # loop-back on the left: diamond -> up -> Node k
    lx = 8
    _arrow(ax, dia.W, (lx, 39), cs="arc3,rad=0")
    ax.plot([lx, lx], [39, 90], color=_INK, lw=1.5, zorder=2)
    _arrow(ax, (lx, 90), nodes[2].W)
    ax.text(lx - 1.5, 64, "yes,  $k \\leftarrow k+1$", rotation=90, va="center",
            ha="center", fontsize=7.5, color=_INK)
    # green "one binary sub-problem" bracket around nodes 2..5
    from matplotlib.patches import FancyBboxPatch
    gx0, gx1 = cx - bw / 2 - 3, cx + bw / 2 + 3
    gy0, gy1 = nodes[5].cy - bh / 2 - 2.5, nodes[2].cy + bh / 2 + 2.5
    ax.add_patch(FancyBboxPatch((gx0, gy0), gx1 - gx0, gy1 - gy0,
                                boxstyle="round,pad=0,rounding_size=1.5",
                                fc="none", ec=OI["green"], lw=1.3, ls=(0, (4, 3)),
                                zorder=1))
    ax.text(gx1 + 2.5, (gy0 + gy1) / 2,
            "one binary\nsub-problem;\nits size does\nnot grow with $K$",
            ha="left", va="center", fontsize=7.3, color=OI["green"], linespacing=1.3)
    _save(fig, ax, "figure_rfoed_flow")


# ----------------------------------------------------------------------- #
def fig_focc_chain():
    plt, fig, ax = _canvas(7.2, 3.3)
    Y = 100 * 3.3 / 7.2
    labs = ["link $\\ell_1$\n(most freq.)", "link $\\ell_2$", "link $\\ell_3$",
            "$\\cdots$", "link $\\ell_L$\n(rarest)"]
    xs = [11, 30, 49, 68, 87]
    ln = [_box(ax, x, Y - 8, 15, 9, t, fc=OI["green"] + "22", ec=OI["green"], fs=7.6)
          for x, t in zip(xs, labs)]
    for a, b in zip(ln, ln[1:]):
        _arrow(ax, a.E, b.W)
    ax.text(49, Y - 1.4, "frequency order  $\\longrightarrow$", ha="center", fontsize=8)

    tr = _box(ax, 50, 24, 86, 9,
              "$\\bf{Training}$ augments link $k$ with the $\\it{true}$ earlier labels:"
              "   $[\\,X \\mid Y_{:,\\ell_1},\\dots,Y_{:,\\ell_{k-1}}\\,]$", fs=8)
    inf = _box(ax, 50, 13, 86, 9,
               "$\\bf{Inference}$ must use the link's own $\\it{predicted}$ earlier labels:"
               "   $[\\,x \\mid \\hat{y}_{\\ell_1},\\dots,\\hat{y}_{\\ell_{k-1}}\\,]$",
               fc=_ORANGE_F, ec=OI["orange"], fs=8)
    _arrow(ax, tr.S, inf.N, color=OI["vermillion"], lw=1.8)
    _elabel(ax, 62, 18.5, "exposure bias", color=OI["vermillion"], fs=7.5, ha="left")
    ax.text(50, 4.5, "No example is ever removed; every link sees every document.",
            ha="center", fontsize=7.5, style="italic")
    _save(fig, ax, "figure_focc_chain")


# ----------------------------------------------------------------------- #
def fig_local_vs_global():
    from matplotlib.patches import Rectangle
    plt, fig, ax = _canvas(7.2, 3.0)
    Y = 100 * 3.0 / 7.2

    _title(ax, 24, Y - 4, "$\\it{Local}$  (RFOED / FOCC):\na binary sub-problem", fs=8.5)
    _box(ax, 15, 22, 9, 15, "class\n$c_k$", fc=_GREEN_F, ec=OI["green"], fs=8)
    _box(ax, 31, 22, 13, 15, "rest", fc=_GREY_F, ec=OI["grey"], fs=8)
    ax.text(24, 8, "balanced bootstrap over 2 strata;\n"
            "draw $\\leq 2n_{\\mathrm{cap}}$, independent of $K$",
            ha="center", fontsize=7.2, linespacing=1.3)

    ax.plot([50, 50], [4, Y - 2], color=OI["grey"], lw=1.2, ls=(0, (5, 3)))

    _title(ax, 76, Y - 4, "$\\it{Global}$  (SMOTE / EasyEnsemble):\none $K$-way problem", fs=8.5)
    for i in range(9):
        a = 0.22 + i * 0.085
        ax.add_patch(Rectangle((57.5 + i * 3.3, 14.5), 2.5, 15,
                               fc=(0.0, 0.45, 0.70, a), ec="#555", lw=0.8, zorder=3))
    ax.text(91.5, 22, "$K$ strata", ha="left", va="center", fontsize=7.5)
    ax.text(76, 8, "$\\kappa$-nearest-neighbour interpolation across $K$ strata;\n"
            "neighbourhoods overlap as $n/K$ shrinks",
            ha="center", fontsize=7.2, linespacing=1.3)
    _save(fig, ax, "figure_local_vs_global")


# ----------------------------------------------------------------------- #
def fig_batched_nn():
    plt, fig, ax = _canvas(7.6, 2.15)
    Y = 100 * 2.15 / 7.6
    my = [Y - 5, Y - 10, Y - 15]
    mb = [_box(ax, 16, y, 27, 4.6, f"member {i}:  bootstrap $X_{i}$", fs=7.6)
          for i, y in zip((1, 2, 3), my)]
    ax.text(16, Y - 19.5, "$\\vdots$    ($K$ members total)", ha="center", fontsize=7.6)
    mid = Y - 10
    tns = _box(ax, 45, mid, 22, 14,
               "stack into one tensor\n$X\\in\\mathbb{R}^{K\\times n_{\\mathrm{boot}}\\times d}$",
               fc=_SKY_F, ec=OI["sky"], fs=7.6)
    bmm = _box(ax, 71, mid, 22, 14,
               "one batched matrix\nproduct per layer\n(torch.bmm)",
               fc=_GREEN_F, ec=OI["green"], fs=7.6)
    vote = _box(ax, 92, mid, 14, 14, "soft-vote\nacross the\n$K$ members", fs=7.6)
    for m in mb:
        _arrow(ax, m.E, (tns.cx - tns.w / 2, m.cy), cs="arc3,rad=0.0")
    _arrow(ax, tns.E, bmm.W)
    _arrow(ax, bmm.E, vote.W)
    _save(fig, ax, "figure_batched_nn")


# ----------------------------------------------------------------------- #
def fig_failure_mechanisms():
    plt, fig, ax = _canvas(7.2, 3.9)
    Y = 100 * 3.9 / 7.2
    lx, rx, bw, bh = 27, 73, 42, 6.6
    ys = [Y - 9, Y - 19.5, Y - 30, Y - 40.5]

    _title(ax, lx, Y - 3, "RFOED-RF:  own-node-miss\n($98.7\\%$ of errors)", fs=8.5)
    lft = [
        _box(ax, lx, ys[0], bw, bh, "test document, true class $c_j$",
             fc=_SKY_F, ec=OI["sky"], fs=7.8),
        _box(ax, lx, ys[1], bw, bh, "correctly passes earlier\nnodes $c_1,\\dots,c_{j-1}$", fs=7.8),
        _box(ax, lx, ys[2], bw, bh, "reaches its own node $c_j$", fs=7.8),
        _box(ax, lx, ys[3], bw, bh + 2.5,
             "node $c_j$ $\\bf{does}$ $\\bf{not}$ $\\bf{fire}$\n"
             "(under-confident: test score $< \\tau$)", fc=_VERM_F, ec=OI["vermillion"], fs=7.8),
    ]
    for a, b in zip(lft, lft[1:]):
        _arrow(ax, a.S, b.N)

    _title(ax, rx, Y - 3, "RFOED-NN:  early-capture\n($95.4\\%$ of errors)", fs=8.5)
    rgt = [
        _box(ax, rx, ys[0], bw, bh, "test document, true class $c_j$",
             fc=_SKY_F, ec=OI["sky"], fs=7.8),
        _box(ax, rx, ys[1], bw, bh + 3.5,
             "an $\\it{earlier}$ node $c_i$ ($i{<}j$)\n$\\bf{fires}$ $\\bf{a}$ $\\bf{false}$ $\\bf{positive}$\n"
             "(over-confident: neg. score $> \\tau$)", fc=_VERM_F, ec=OI["vermillion"], fs=7.6),
        _box(ax, rx, ys[2], bw, bh, "the document is removed\nfrom contention", fs=7.8),
        _box(ax, rx, ys[3], bw, bh, "its own node $c_j$ is\nnever reached", fs=7.8),
    ]
    for a, b in zip(rgt, rgt[1:]):
        _arrow(ax, a.S, b.N)
    _save(fig, ax, "figure_failure_mechanisms")


# ----------------------------------------------------------------------- #
def fig_decision_tree():
    plt, fig, ax = _canvas(7.2, 7.6)
    Y = 100 * 7.6 / 7.2
    dw, dh, spine = 26, 15, 38
    yft, yrep, yml, yrec = Y - 9, Y - 29, Y - 49, Y - 71

    ft = _diamond(ax, spine, yft, dw, dh, "can the encoder be\nfine-tuned end-to-end?", fs=7.6)
    ftbox = _box(ax, 84, yft, 30, 15,
                 "fine-tune it; handle\nimbalance in the loss\n(class-weighted CE)",
                 fc=_GREEN_F, ec=OI["green"], fs=7.4, weight="bold")
    rep = _box(ax, spine, yrep, 40, 12,
               "choose the representation first:\nflat learner on TF-IDF vs on embeddings",
               fs=7.6)
    ml = _diamond(ax, spine, yml, dw, dh, "single- or\nmulti-label?", fs=7.8)
    flat = _box(ax, 84, yml, 32, 19,
                "flat learner on the better\nrepresentation; no peel\n(stump-based resampling\nensembles fail above $K{\\approx}100$)",
                fc=_GREEN_F, ec=OI["green"], fs=7.2, weight="bold")
    rec = _diamond(ax, spine, yrec, dw + 4, dh + 2, "rare-label recall\nworth some\nmicro-F1?", fs=7.6)
    focc = _box(ax, spine, yrec - 20, 40, 12,
                "FOCC-NN\n(per-link rebalancing on;\nordering optional)",
                fc=_GREEN_F, ec=OI["green"], fs=7.8, weight="bold")
    mlp = _box(ax, 84, yrec, 30, 11, "flat multi-output\nMLP", fc=_GREEN_F, ec=OI["green"],
               fs=7.8, weight="bold")

    _arrow(ax, ft.E, ftbox.W)
    _elabel(ax, (ft.E[0] + ftbox.W[0]) / 2, yft + 2.6, "yes")
    _arrow(ax, ft.S, rep.N)
    _elabel(ax, spine + 3.5, (ft.S[1] + rep.N[1]) / 2, "no")
    _arrow(ax, rep.S, ml.N)
    _arrow(ax, ml.E, flat.W)
    _elabel(ax, (ml.E[0] + flat.W[0]) / 2, yml + 2.6, "single")
    _arrow(ax, ml.S, rec.N)
    _elabel(ax, spine + 3.5, (ml.S[1] + rec.N[1]) / 2, "multi")
    _arrow(ax, rec.E, mlp.W)
    _elabel(ax, (rec.E[0] + mlp.W[0]) / 2, yrec + 2.6, "no")
    _arrow(ax, rec.S, focc.N)
    _elabel(ax, spine + 3.5, (rec.S[1] + focc.N[1]) / 2, "yes")
    _save(fig, ax, "figure_decision_tree")


FIGS = {
    "landscape": fig_landscape,
    "collapse": fig_collapse_bars,
    "mainresults_sl": fig_mainresults_sl,
    "mainresults_ml": fig_mainresults_ml,
    "factorial": fig_factorial_heatmap,
    "remediation": fig_remediation_sweep,
    "exposure": fig_exposure_bias,
    "perlink": fig_perlink_exposure,
    "separability": fig_separability,
    "neighbourhood": fig_neighbourhood,
    "calibration": fig_calibration,
    "graphical_abstract": fig_graphical_abstract,
    "rfoed_flow": fig_rfoed_flow,
    "focc_chain": fig_focc_chain,
    "local_vs_global": fig_local_vs_global,
    "batched_nn": fig_batched_nn,
    "failure_mechanisms": fig_failure_mechanisms,
    "decision_tree": fig_decision_tree,
}


def main(which):
    FIG.mkdir(exist_ok=True)
    for k in (which or FIGS):
        fn = FIGS.get(k)
        if not fn:
            print(f"skip {k}"); continue
        print(f"[{k}]")
        try:
            fn()
        except Exception as e:  # noqa: BLE001
            import traceback
            print(f"  FAILED: {e}\n{traceback.format_exc()}")


if __name__ == "__main__":
    main(sys.argv[1:])
