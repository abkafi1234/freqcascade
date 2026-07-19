"""Statistical-analysis module (design §6/§6b; TIER3_PLAN.md T3.1).

Every significance artifact the paper's results section needs, built as
pure functions over numpy/pandas so each piece can be unit-tested against
hand-worked examples independent of the benchmark CSV (see
tests/test_stats.py):

- Paired Wilcoxon signed-rank + paired t-test + Shapiro-Wilk on the paired
  diffs, with matched-pairs rank-biserial correlation and Cohen's d
  reported alongside every p-value.
- Holm-Bonferroni step-down correction across a pairwise-comparison family.
- 95% CI via the t-distribution, generic in the number of matched
  seeds/folds available.
- Friedman omnibus test + Nemenyi post-hoc + Critical-Difference diagram
  across datasets (design §6, Figure 1) -- implemented from scratch since
  no posthoc package (scikit-posthocs/Orange) is installed.
- Three-way ANOVA (statsmodels OLS + anova_lm typ=2) for the 2x2x2
  factorial ablation, with a from-scratch Scheirer-Ray-Hare nonparametric
  fallback (rank-transform the response, reuse the same sum-of-squares
  decomposition, read off a chi-squared statistic instead of F) used when
  the ANOVA's residuals fail a Shapiro-Wilk normality check.

No plotting/heavy imports (matplotlib, statsmodels) happen at module import
time except where unavoidable (statsmodels for the ANOVA path) -- matplotlib
is imported lazily inside plot_cd_diagram so importing this module never
requires a display backend.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sstats

# ---------------------------------------------------------------------------
# Reporting helpers (design §6c)
# ---------------------------------------------------------------------------


def format_p(p: float, sig_figs: int = 3, floor: float = 1e-3) -> str:
    """3 significant figures; '<0.001' below floor -- design §6c."""
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return "n/a"
    if p < floor:
        return f"<{floor:g}"
    return f"{p:.{sig_figs}g}"


# ---------------------------------------------------------------------------
# Holm-Bonferroni step-down correction
# ---------------------------------------------------------------------------


def holm_bonferroni(pvalues) -> np.ndarray:
    """Holm step-down adjusted p-values, returned in the *original* input
    order.

    Procedure (Holm 1979): sort p-values ascending, p_(1) <= ... <= p_(m).
    adjusted_(i) = max_{j<=i} [ (m - j + 1) * p_(j) ], clipped to [0, 1].
    The cumulative max enforces the adjusted p-values are non-decreasing
    (required for a valid step-down procedure). See
    tests/test_stats.py::test_holm_bonferroni_textbook_example for a
    hand-worked check.
    """
    p = np.asarray(pvalues, dtype=float)
    m = len(p)
    if m == 0:
        return p
    order = np.argsort(p, kind="stable")
    sorted_p = p[order]
    multipliers = m - np.arange(m)  # m, m-1, ..., 1
    raw_adj = sorted_p * multipliers
    adj_sorted = np.maximum.accumulate(raw_adj)
    adj_sorted = np.clip(adj_sorted, 0.0, 1.0)
    adjusted = np.empty(m)
    adjusted[order] = adj_sorted
    return adjusted


# ---------------------------------------------------------------------------
# Confidence intervals
# ---------------------------------------------------------------------------


@dataclass
class CIResult:
    mean: float
    half_width: float
    lo: float
    hi: float
    n: int


def t_confidence_interval(values, confidence: float = 0.95) -> CIResult:
    """95% CI on the mean via the t-distribution, generic in n (currently
    3 seeds in the CSV; will become 10 once T2.7 lands -- no hardcoded n)."""
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    n = len(values)
    mean = float(np.mean(values)) if n else float("nan")
    if n < 2:
        return CIResult(mean=mean, half_width=float("nan"), lo=float("nan"), hi=float("nan"), n=n)
    sem = float(sstats.sem(values))
    tcrit = float(sstats.t.ppf((1 + confidence) / 2, df=n - 1))
    half = tcrit * sem
    return CIResult(mean=mean, half_width=half, lo=mean - half, hi=mean + half, n=n)


# ---------------------------------------------------------------------------
# Effect sizes
# ---------------------------------------------------------------------------


def cohens_d_paired(x, y) -> float:
    """Cohen's d_z for paired samples: mean(diff) / sd(diff, ddof=1)."""
    diff = np.asarray(x, dtype=float) - np.asarray(y, dtype=float)
    if len(diff) < 2:
        return float("nan")
    sd = np.std(diff, ddof=1)
    if sd == 0:
        return 0.0 if np.mean(diff) == 0 else float("inf") * np.sign(np.mean(diff))
    return float(np.mean(diff) / sd)


def rank_biserial_wilcoxon(x, y) -> float:
    """Matched-pairs rank-biserial correlation for the Wilcoxon signed-rank
    test (King & Minium; Kerby 2014): r = (W+ - W-) / (W+ + W-), where W+/W-
    are the sums of ranks of |d_i| assigned to positive/negative diffs.
    Zero diffs are dropped, matching scipy.stats.wilcoxon's default
    zero_method='wilcox' so the effect size and p-value are computed over
    the same pairs."""
    diff = np.asarray(x, dtype=float) - np.asarray(y, dtype=float)
    nz = diff[diff != 0]
    if len(nz) == 0:
        return 0.0
    ranks = sstats.rankdata(np.abs(nz))
    w_pos = ranks[nz > 0].sum()
    w_neg = ranks[nz < 0].sum()
    total = w_pos + w_neg
    if total == 0:
        return 0.0
    return float((w_pos - w_neg) / total)


# ---------------------------------------------------------------------------
# Paired comparison (Wilcoxon + t-test + Shapiro + effect sizes)
# ---------------------------------------------------------------------------


@dataclass
class PairedTestResult:
    n: int
    mean_diff: float
    wilcoxon_stat: float
    wilcoxon_p: float
    rank_biserial: float
    ttest_stat: float
    ttest_p: float
    cohens_d: float
    shapiro_p: float  # normality of the paired diffs
    note: str = ""


def paired_comparison(x, y) -> PairedTestResult:
    """x, y: matched-seed metric arrays for (method_a, method_b), same
    length and seed order. Returns Wilcoxon signed-rank (primary, per
    design §6) + paired t-test (supplementary) + Shapiro-Wilk on the
    diffs (appendix normality check) + both effect sizes."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) != len(y):
        raise ValueError(f"paired_comparison requires equal-length matched arrays, got {len(x)} vs {len(y)}")
    n = len(x)
    diff = x - y
    mean_diff = float(np.mean(diff)) if n else float("nan")
    notes = []

    if n == 0:
        w_stat, w_p = float("nan"), float("nan")
        notes.append("n=0")
    elif np.all(diff == 0):
        w_stat, w_p = float("nan"), 1.0
        notes.append("all diffs zero")
    else:
        try:
            w_stat, w_p = sstats.wilcoxon(x, y, alternative="two-sided", method="auto")
            w_stat, w_p = float(w_stat), float(w_p)
        except ValueError as e:
            w_stat, w_p = float("nan"), float("nan")
            notes.append(f"wilcoxon failed: {e}")

    rb = rank_biserial_wilcoxon(x, y)

    if n < 2:
        t_stat, t_p = float("nan"), float("nan")
        notes.append("n<2, t-test skipped")
    elif np.std(diff, ddof=1) == 0:
        t_stat, t_p = float("nan"), (1.0 if mean_diff == 0 else 0.0)
        notes.append("zero variance in diffs")
    else:
        t_res = sstats.ttest_rel(x, y)
        t_stat, t_p = float(t_res.statistic), float(t_res.pvalue)

    d = cohens_d_paired(x, y)

    if n < 3:
        shapiro_p = float("nan")
        notes.append("n<3, shapiro skipped")
    elif np.std(diff, ddof=1) == 0:
        shapiro_p = float("nan")
        notes.append("zero variance, shapiro skipped")
    else:
        try:
            shapiro_p = float(sstats.shapiro(diff).pvalue)
        except ValueError as e:
            shapiro_p = float("nan")
            notes.append(f"shapiro failed: {e}")

    return PairedTestResult(
        n=n, mean_diff=mean_diff, wilcoxon_stat=w_stat, wilcoxon_p=w_p,
        rank_biserial=rb, ttest_stat=t_stat, ttest_p=t_p, cohens_d=d,
        shapiro_p=shapiro_p, note="; ".join(notes),
    )


# ---------------------------------------------------------------------------
# Friedman + Nemenyi + Critical-Difference diagram
# ---------------------------------------------------------------------------


@dataclass
class FriedmanResult:
    statistic: float
    p_value: float
    avg_ranks: pd.Series  # method -> mean rank across blocks, ascending (1 = best)
    n_blocks: int
    k_methods: int


def friedman_test(score_matrix: pd.DataFrame, higher_is_better: bool = True) -> FriedmanResult:
    """score_matrix: rows = blocks (e.g. datasets, each already averaged
    over seeds), columns = methods, values = the metric. Ranks are
    computed per row (rank 1 = best). Rows with any NaN are dropped (a
    method missing on a dataset can't be ranked there)."""
    sm = score_matrix.dropna(axis=0, how="any")
    n, k = sm.shape
    if n < 2 or k < 2:
        raise ValueError(f"friedman_test needs >=2 blocks and >=2 methods, got n_blocks={n}, k_methods={k}")
    ranks = sm.rank(axis=1, ascending=not higher_is_better, method="average")
    avg_ranks = ranks.mean(axis=0).sort_values()
    groups = [sm[col].values for col in sm.columns]
    stat, p = sstats.friedmanchisquare(*groups)
    return FriedmanResult(statistic=float(stat), p_value=float(p), avg_ranks=avg_ranks, n_blocks=n, k_methods=k)


def nemenyi_q_alpha(k: int, alpha: float = 0.05) -> float:
    """Critical value of the studentized range statistic (infinite df,
    since ranks are asymptotically normal), scaled by sqrt(2) -- the q_alpha
    used in Demsar (2006) Table 5. Matches published values, e.g. k=9,
    alpha=0.05 -> q ~= 3.102 (verified in tests/test_stats.py)."""
    return float(sstats.studentized_range.ppf(1 - alpha, k, np.inf) / np.sqrt(2))


def nemenyi_critical_difference(k: int, n_blocks: int, alpha: float = 0.05) -> float:
    """CD = q_alpha * sqrt(k(k+1) / (6N)) -- Demsar (2006) eq. 6."""
    q = nemenyi_q_alpha(k, alpha)
    return float(q * np.sqrt(k * (k + 1) / (6.0 * n_blocks)))


def nemenyi_pairwise_table(avg_ranks: pd.Series, cd: float) -> pd.DataFrame:
    """All pairwise method comparisons: rank difference and whether it
    exceeds the Nemenyi CD (i.e. is a significant pairwise difference)."""
    methods = avg_ranks.index.tolist()
    rows = []
    for i in range(len(methods)):
        for j in range(i + 1, len(methods)):
            a, b = methods[i], methods[j]
            diff = abs(float(avg_ranks[a] - avg_ranks[b]))
            rows.append({
                "method_a": a, "method_b": b,
                "rank_a": float(avg_ranks[a]), "rank_b": float(avg_ranks[b]),
                "rank_diff": diff, "cd": cd, "significant": diff > cd,
            })
    return pd.DataFrame(rows)


def _find_cd_cliques(sorted_ranks: list[float], cd: float) -> list[list[int]]:
    """Maximal groups of consecutive (by rank) items whose rank spread is
    <= CD -- i.e. statistically-indistinguishable cliques, the standard
    CD-diagram grouping (Demsar 2006 / the Orange package's cd diagram).
    `sorted_ranks` must already be sorted ascending. Returns lists of
    (0-indexed) positions into `sorted_ranks`."""
    n = len(sorted_ranks)
    candidates = []
    for i in range(n):
        j = i
        while j + 1 < n and sorted_ranks[j + 1] - sorted_ranks[i] <= cd:
            j += 1
        if j > i:
            candidates.append((i, j))
    # keep only maximal intervals (drop any interval contained in another)
    maximal = [c for c in candidates if not any(c != o and o[0] <= c[0] and c[1] <= o[1] for o in candidates)]
    seen = set()
    result = []
    for c in maximal:
        if c not in seen:
            seen.add(c)
            result.append(list(range(c[0], c[1] + 1)))
    return result


def plot_cd_diagram(
    avg_ranks: pd.Series,
    cd: float,
    out_path: Path,
    title: str = "Critical Difference Diagram (Friedman + Nemenyi)",
) -> None:
    """Standard CD diagram: a rank axis, one marker+label per method, and
    thick horizontal bars joining methods whose mean-rank difference is
    within the Nemenyi CD (statistically indistinguishable cliques)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    INK = "#0b0b0b"
    MUTED = "#898781"
    ACCENT = "#2a78d6"

    methods = list(avg_ranks.index)
    ranks = np.asarray(avg_ranks.values, dtype=float)
    order = np.argsort(ranks)
    sorted_methods = [methods[i] for i in order]
    sorted_ranks = ranks[order]
    k = len(sorted_methods)

    lo, hi = float(sorted_ranks.min()) - 0.5, float(sorted_ranks.max()) + 0.5
    axis_y = 0.90

    fig_h = 2.2 + 0.34 * k
    fig, ax = plt.subplots(figsize=(9, fig_h))
    ax.set_xlim(lo, hi)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title(f"{title}\nCD = {cd:.3f} (alpha=0.05, k={k} methods)", fontsize=11, color=INK)

    # axis line + integer ticks
    ax.plot([lo, hi], [axis_y, axis_y], color=INK, lw=1.3, zorder=1)
    tick_lo, tick_hi = int(np.ceil(lo)), int(np.floor(hi))
    for r in range(tick_lo, tick_hi + 1):
        ax.plot([r, r], [axis_y - 0.012, axis_y + 0.012], color=INK, lw=1.2)
        ax.text(r, axis_y + 0.03, str(r), ha="center", va="bottom", fontsize=9, color=MUTED)

    # method markers, drop lines, and stacked labels (one row per method,
    # ordered by rank so nothing overlaps vertically)
    label_top = axis_y - 0.08
    row_h = (label_top - 0.05) / max(k, 1)
    mid = (lo + hi) / 2
    for i, (name, r) in enumerate(zip(sorted_methods, sorted_ranks)):
        y_row = label_top - i * row_h
        ax.plot([r, r], [axis_y, y_row], color=ACCENT, lw=0.9, zorder=2)
        ax.plot(r, axis_y, "o", color=ACCENT, ms=4, zorder=3)
        ha = "left" if r < mid else "right"
        dx = 0.03 if ha == "left" else -0.03
        ax.text(r + dx, y_row, f"{name}  ({r:.2f})", ha=ha, va="center", fontsize=9, color=INK)

    # clique bars: statistically-indistinguishable groups joined by a
    # thick bar along the axis
    cliques = _find_cd_cliques(list(sorted_ranks), cd)
    bar_y0 = 0.04
    for ci, clique in enumerate(cliques):
        r0, r1 = sorted_ranks[clique[0]], sorted_ranks[clique[-1]]
        y = bar_y0 + 0.035 * ci
        ax.plot([r0, r1], [y, y], color=INK, lw=3.0, solid_capstyle="butt")

    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Three-way ANOVA (statsmodels) + Scheirer-Ray-Hare nonparametric fallback
# ---------------------------------------------------------------------------


def _anova_formula(response: str, factors: tuple[str, ...]) -> str:
    return f"{response} ~ " + " * ".join(f"C({f})" for f in factors)


@dataclass
class AnovaResult:
    table: pd.DataFrame          # statsmodels anova_lm(typ=2) table
    residual_shapiro_p: float
    normal_residuals: bool


def three_way_anova(df: pd.DataFrame, response: str, factors: tuple[str, str, str]) -> AnovaResult:
    """Standard factorial ANOVA (Type II sum of squares -- appropriate for
    a design without a strict hierarchy assumption) via statsmodels OLS.
    Also runs a Shapiro-Wilk test on the model residuals, since the design
    (§6) requires checking residual normality and falling back to
    Scheirer-Ray-Hare when it fails."""
    from statsmodels.formula.api import ols
    from statsmodels.stats.anova import anova_lm

    formula = _anova_formula(response, factors)
    model = ols(formula, data=df).fit()
    table = anova_lm(model, typ=2)
    resid = model.resid
    if len(resid) >= 3:
        shapiro_p = float(sstats.shapiro(resid).pvalue)
    else:
        shapiro_p = float("nan")
    normal = bool(shapiro_p >= 0.05) if not np.isnan(shapiro_p) else False
    return AnovaResult(table=table, residual_shapiro_p=shapiro_p, normal_residuals=normal)


def scheirer_ray_hare(df: pd.DataFrame, response: str, factors: tuple[str, ...]) -> pd.DataFrame:
    """Scheirer-Ray-Hare nonparametric factorial test (Scheirer, Ray & Hare
    1976): rank-transform the response across the *whole* sample, run the
    same sum-of-squares decomposition used for the parametric ANOVA on the
    ranks (via statsmodels' Type II SS, reusing the same formula/engine as
    three_way_anova for consistency), then convert each term's SS to an
    H-statistic: H = SS_effect / (SS_total / (N - 1)), tested against a
    chi-squared distribution with the term's df. This is the standard SRH
    recipe -- "ANOVA on ranks, read off chi-squared instead of F".

    Verified in tests/test_stats.py against a manually hand-computed
    sum-of-squares decomposition on a small balanced synthetic factorial
    design (the SS machinery is shared with three_way_anova, so verifying
    the SS decomposition there covers this path too), plus a
    known-separation sanity check (an injected, non-overlapping factor
    effect must come out significant; a pure-noise factor must not).
    """
    from statsmodels.formula.api import ols
    from statsmodels.stats.anova import anova_lm

    work = df.copy()
    work["_rank"] = sstats.rankdata(work[response].values)
    formula = _anova_formula("_rank", factors)
    model = ols(formula, data=work).fit()
    table = anova_lm(model, typ=2).copy()

    n = len(work)
    ss_total = float(table["sum_sq"].sum())
    ms_total = ss_total / (n - 1) if n > 1 else float("nan")

    table["H"] = table["sum_sq"] / ms_total
    table["p_value"] = sstats.chi2.sf(table["H"], table["df"])
    table.loc["Residual", ["H", "p_value"]] = [np.nan, np.nan]
    return table


# ---------------------------------------------------------------------------
# Combined driver: ANOVA with automatic SRH fallback on non-normal residuals
# ---------------------------------------------------------------------------


@dataclass
class FactorialResult:
    anova: AnovaResult
    srh_table: pd.DataFrame
    recommended: str  # "anova" or "srh"


def anova_or_srh(df: pd.DataFrame, response: str, factors: tuple[str, str, str], alpha_normality: float = 0.05) -> FactorialResult:
    """Runs both the parametric three-way ANOVA and the Scheirer-Ray-Hare
    fallback unconditionally (design §6 wants both reported: 'Scheirer-Ray-
    Hare nonparametric analog if residual normality fails'), and flags
    which one is recommended based on the ANOVA residuals' Shapiro-Wilk
    p-value."""
    anova = three_way_anova(df, response, factors)
    srh = scheirer_ray_hare(df, response, factors)
    recommended = "anova" if anova.normal_residuals else "srh"
    return FactorialResult(anova=anova, srh_table=srh, recommended=recommended)
