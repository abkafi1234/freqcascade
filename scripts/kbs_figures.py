"""Figures for the Knowledge-Based Systems version of the paper.

Data-driven result plots (from results/*.jsonl) and two programmatic concept
figures. True flow/schematic diagrams are drawn in TikZ inside the manuscript.

    python scripts/kbs_figures.py            # all
    python scripts/kbs_figures.py landscape collapse   # a subset

Every figure is written to figures/ as a PDF at a fixed name (figureNN_*.pdf).
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
        "Drug Reviews": (-9, 6, "right"), "Reuters-21578": (-9, 6, "right"),
        "Hallmarks of Cancer": (9, -2, "left"), "LitCovid": (9, 2, "left"),
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
    ax.legend(handles=leg, loc="lower left", fontsize=7.5, handletextpad=0.4)
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
    for xi, (d, _, _) in zip(x, meta):
        g = np.mean(agg[d]["gap"])
        ax.annotate(f"gap {g:+.3f}", (xi, max(p[list(x).index(xi)], o[list(x).index(xi)]) + 0.02),
                    ha="center", fontsize=7.5)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{nm}\n$L$={L} labels" for _, nm, L in meta], fontsize=8)
    ax.set_ylabel("FOCC-NN label-macro-F1")
    ax.set_ylim(0, 0.95)
    ax.legend(fontsize=7.5, loc="upper left")
    ax.set_title("Exposure bias: the price of predicting earlier links\n"
                 "is large on the 90-label chain, modest on the short ones", fontsize=8)
    ax.grid(axis="x", alpha=0)
    fig.tight_layout()
    out = FIG / "figure_exposure_bias.pdf"
    fig.savefig(out); print(f"  -> {out}")


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


def fig_separability():
    plt = _mpl()
    # RFOED-NN macro-F1 margin over the best *standard* baseline (flat family), single-label.
    # WOS46985 is the extreme boundary case analysed separately (Sec. 8.2) and shown apart.
    spec = [("clinc150", "CLINC150", 150, (6, 6)),
            ("20newsgroups_ir50", "20 Newsgroups", 20, (0, 9)),
            ("ohsumed_23", "OHSUMED-23", 23, (0, -13)),
            ("drug_reviews", "Drug Reviews", 356, (0, 9))]
    pts = []
    for name, lbl, K, off in spec:
        a, _ = _agg(name, "macro_f1")
        prop = a.get("RFOED-NN", a.get("RFOED-NN (capped)"))
        base = [np.mean(v) for m, v in a.items() if m.startswith(("Flat", "OVR"))]
        if prop is None or not base:
            continue
        pts.append((_silhouette(name), float(np.mean(prop)) - max(base), K, lbl, off))
    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    ks = np.array([p[2] for p in pts])
    sc = ax.scatter([p[0] for p in pts], [p[1] for p in pts], c=np.log10(ks),
                    s=120, cmap="cividis", edgecolor="black", linewidth=0.8, zorder=3,
                    vmin=1.0, vmax=2.6)
    for sil, mrg, K, lbl, off in pts:
        ax.annotate(lbl, (sil, mrg), off, textcoords="offset points", fontsize=8, ha="center")
    ax.axhline(0, color="black", lw=1.0, ls="--")
    ax.axhspan(0, 0.06, color=OI["green"], alpha=0.06)
    ax.axhspan(-0.05, 0, color=OI["vermillion"], alpha=0.06)
    ax.text(0.02, 0.052, "RFOED-NN wins", fontsize=7, color=OI["green"])
    ax.text(0.02, -0.043, "RFOED-NN loses", fontsize=7, color=OI["vermillion"])
    ax.set_xlabel("class separability  (mean silhouette of the frozen MiniLM embedding)")
    ax.set_ylabel("RFOED-NN macro-F1 margin\nover best flat baseline")
    ax.set_ylim(-0.05, 0.065)
    cb = fig.colorbar(sc, ax=ax, fraction=0.045, pad=0.03)
    cb.set_label("$\\log_{10} K$", fontsize=8)
    ax.set_title("Among directly-deployed datasets, the margin's sign follows separability, not $K$",
                 fontsize=8)
    ax.text(0.99, 0.02, "WOS46985 ($K$=134, sil.\\ $\\approx$0): margin $-$0.20 even remediated;\n"
            "the extreme boundary case, diagnosed in Section~8.2",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=6.5, style="italic")
    fig.tight_layout()
    out = FIG / "figure_separability.pdf"
    fig.savefig(out); print(f"  -> {out}")


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
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9), sharey=True)
    xs = np.linspace(0, 1, 400)

    def gauss(m, s):
        return np.exp(-0.5 * ((xs - m) / s) ** 2)

    # RF: true positives score low at test time (generalization gap) -> under-confident
    ax = axes[0]
    ax.fill_between(xs, gauss(0.16, 0.09), color=OI["grey"], alpha=0.55, label="negatives")
    ax.fill_between(xs, 0.55 * gauss(0.34, 0.13), color=OI["green"], alpha=0.65, label="own-class positives")
    ax.axvline(0.5, color="black", lw=1.4)
    ax.axvline(0.2, color=OI["vermillion"], lw=1.6, ls="--")
    ax.text(0.52, 0.82, r"default $\tau=0.5$" + "\nmisses most positives", fontsize=6.8, ha="left")
    ax.text(0.205, 0.30, r"lowered $\tau$", fontsize=6.8, color=OI["vermillion"], ha="left", rotation=90)
    ax.set_title("RFOED-RF: under-confident\n(test-time generalization gap)", fontsize=8)
    ax.set_xlabel("node score $p$"); ax.set_yticks([]); ax.grid(False)
    ax.legend(fontsize=6.8, loc="upper right")

    # NN: negatives pushed above 0.5 -> early captures -> over-confident
    ax = axes[1]
    ax.fill_between(xs, gauss(0.34, 0.20), color=OI["grey"], alpha=0.55)
    ax.fill_between(xs, 0.55 * gauss(0.82, 0.11), color=OI["green"], alpha=0.65)
    ax.axvline(0.5, color="black", lw=1.4)
    ax.axvline(0.95, color=OI["vermillion"], lw=1.6, ls="--")
    ax.text(0.06, 0.80, r"default $\tau=0.5$" + "\nfires on negatives\n($\\Rightarrow$ early capture)", fontsize=6.8, ha="left")
    ax.text(0.80, 0.30, r"raised $\tau$", fontsize=6.8, color=OI["vermillion"], ha="left", rotation=90)
    ax.set_title("RFOED-NN: over-confident\n(training-prior / threshold mismatch)", fontsize=8)
    ax.set_xlabel("node score $p$"); ax.grid(False)
    fig.suptitle("Every node is trained at a balanced prior but deployed against a sub-1% node-local prior;\n"
                 "the two base learners miscalibrate in opposite directions",
                 fontsize=8.3, y=1.06)
    fig.tight_layout()
    out = FIG / "figure_calibration.pdf"
    fig.savefig(out, bbox_inches="tight"); print(f"  -> {out}")


FIGS = {
    "landscape": fig_landscape,
    "collapse": fig_collapse_bars,
    "mainresults_sl": fig_mainresults_sl,
    "mainresults_ml": fig_mainresults_ml,
    "factorial": fig_factorial_heatmap,
    "remediation": fig_remediation_sweep,
    "exposure": fig_exposure_bias,
    "separability": fig_separability,
    "neighbourhood": fig_neighbourhood,
    "calibration": fig_calibration,
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
