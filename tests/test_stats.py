"""Unit tests for freqcascade.stats.

Focus: the Nemenyi CD critical value and the Holm-Bonferroni
ordering/adjustment logic against hand-worked examples, plus the
three-way ANOVA / Scheirer-Ray-Hare sum-of-squares machinery against a
manually hand-computed balanced 2x2x2 factorial design (verified by
hand in the design comment below, not against a memorized textbook SRH
table).

Run with: pip install -e ".[dev]" && pytest tests/test_stats.py -v
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd
import pytest
from scipy import stats as sstats

from freqcascade import stats as rstats


# ---------------------------------------------------------------------------
# Holm-Bonferroni
# ---------------------------------------------------------------------------


def test_holm_bonferroni_textbook_example():
    """Wikipedia / standard Holm (1979) worked example: p = [0.01, 0.04,
    0.03, 0.005], m=4, alpha=0.05.

    Hand computation (sorted ascending: 0.005, 0.01, 0.03, 0.04):
      i=1 (p=0.005): (4-1+1)*0.005 = 0.020
      i=2 (p=0.01):  (4-2+1)*0.01  = 0.030 -> cummax(0.020,0.030)=0.030
      i=3 (p=0.03):  (4-3+1)*0.03  = 0.060 -> cummax(...,0.060)=0.060
      i=4 (p=0.04):  (4-4+1)*0.04  = 0.040 -> cummax(...,0.040)=0.060
    Adjusted (sorted order): [0.020, 0.030, 0.060, 0.060]
    Mapped back to original order [0.01, 0.04, 0.03, 0.005]:
      -> [0.030, 0.060, 0.060, 0.020]
    At alpha=0.05: only the first two sorted hypotheses (p=0.005, p=0.01)
    are rejected (adjusted 0.020 and 0.030 < 0.05); p=0.03 and p=0.04 are
    not (adjusted 0.060 for both).
    """
    raw = [0.01, 0.04, 0.03, 0.005]
    adjusted = rstats.holm_bonferroni(raw)
    expected = [0.03, 0.06, 0.06, 0.02]
    assert adjusted == pytest.approx(expected, abs=1e-9)

    # reject/fail-to-reject pattern at alpha=0.05
    rejected = adjusted < 0.05
    assert list(rejected) == [True, False, False, True]


def test_holm_bonferroni_is_monotonic_and_original_order_preserved():
    """Adjusted p-values must be non-decreasing when read in *sorted*
    p-value order (a Holm-specific requirement), and the function must
    return results aligned to the caller's original (unsorted) order."""
    raw = [0.2, 0.001, 0.05, 0.5, 0.03]
    adjusted = rstats.holm_bonferroni(raw)
    assert len(adjusted) == len(raw)
    order = np.argsort(raw)
    sorted_adjusted = np.asarray(adjusted)[order]
    assert np.all(np.diff(sorted_adjusted) >= -1e-12)  # non-decreasing
    # every adjusted p must be >= its raw p (Holm never makes it easier)
    assert np.all(np.asarray(adjusted) >= np.asarray(raw) - 1e-12)
    # smallest raw p-value gets the largest per-hypothesis multiplier
    assert adjusted[1] == pytest.approx(raw[1] * len(raw))


def test_holm_bonferroni_single_pvalue_unchanged():
    adjusted = rstats.holm_bonferroni([0.02])
    assert adjusted == pytest.approx([0.02])


# ---------------------------------------------------------------------------
# Nemenyi q_alpha / Critical Difference
# ---------------------------------------------------------------------------


# Published q_alpha values at alpha=0.05, Demsar (2006) "Statistical
# Comparisons of Classifiers over Multiple Data Sets", Table 5 (right).
DEMSAR_Q_ALPHA_005 = {
    2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728,
    6: 2.850, 7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164,
}


@pytest.mark.parametrize("k,expected_q", DEMSAR_Q_ALPHA_005.items())
def test_nemenyi_q_alpha_matches_published_table(k, expected_q):
    q = rstats.nemenyi_q_alpha(k, alpha=0.05)
    assert q == pytest.approx(expected_q, abs=5e-3)


def test_nemenyi_critical_difference_formula():
    """CD = q_alpha * sqrt(k(k+1) / (6N)) -- Demsar (2006) eq. 6. Cross-
    check the helper against the formula computed independently (using
    scipy's studentized_range directly, not by calling nemenyi_q_alpha)
    for the project's actual scale: k=9 methods, N=3 datasets."""
    k, n_blocks = 9, 3
    q_independent = sstats.studentized_range.ppf(0.95, k, np.inf) / np.sqrt(2)
    expected_cd = q_independent * np.sqrt(k * (k + 1) / (6.0 * n_blocks))
    cd = rstats.nemenyi_critical_difference(k, n_blocks, alpha=0.05)
    assert cd == pytest.approx(expected_cd, rel=1e-9)
    # sanity: with only 3 datasets (low power), the CD should already be a
    # substantial fraction of the full rank range (k-1=8) -- i.e. most
    # pairwise differences among 9 methods won't be individually
    # significant at N=3, which is the expected real-world caveat here.
    assert cd > (k - 1) / 2


def test_nemenyi_critical_difference_shrinks_with_more_blocks():
    k = 5
    cd_small_n = rstats.nemenyi_critical_difference(k, n_blocks=3)
    cd_large_n = rstats.nemenyi_critical_difference(k, n_blocks=30)
    assert cd_large_n < cd_small_n


# ---------------------------------------------------------------------------
# CD-diagram clique finder
# ---------------------------------------------------------------------------


def test_find_cd_cliques_hand_worked():
    """sorted_ranks = [1.0, 1.5, 2.0, 4.0, 4.3, 4.6], cd = 1.0.
    Hand trace (spread from left endpoint of each growing window):
      i=0: 1.5-1.0=0.5<=1 -> extend; 2.0-1.0=1.0<=1 -> extend;
           4.0-1.0=3.0>1 -> stop. candidate (0,2)
      i=1: 2.0-1.5=0.5<=1 -> extend; 4.0-1.5=2.5>1 -> stop. candidate (1,2)
      i=2: 4.0-2.0=2.0>1 -> no extension (not a candidate)
      i=3: 4.3-4.0=0.3<=1 -> extend; 4.6-4.0=0.6<=1 -> extend; end.
           candidate (3,5)
      i=4: 4.6-4.3=0.3<=1 -> extend; end. candidate (4,5)
    Maximal (drop intervals contained in another): (1,2) subset of (0,2);
    (4,5) subset of (3,5) -> final cliques: [0,1,2] and [3,4,5].
    """
    sorted_ranks = [1.0, 1.5, 2.0, 4.0, 4.3, 4.6]
    cliques = rstats._find_cd_cliques(sorted_ranks, cd=1.0)
    assert cliques == [[0, 1, 2], [3, 4, 5]]


def test_find_cd_cliques_no_grouping_when_cd_tiny():
    sorted_ranks = [1.0, 2.0, 3.0]
    cliques = rstats._find_cd_cliques(sorted_ranks, cd=0.01)
    assert cliques == []


def test_find_cd_cliques_all_one_group_when_cd_huge():
    sorted_ranks = [1.0, 2.0, 3.0]
    cliques = rstats._find_cd_cliques(sorted_ranks, cd=100.0)
    assert cliques == [[0, 1, 2]]


# ---------------------------------------------------------------------------
# Friedman test
# ---------------------------------------------------------------------------


def test_friedman_perfect_agreement_gives_expected_ranks_and_low_p():
    """4 blocks (datasets), 3 methods, 'best' always highest, 'worst'
    always lowest -> avg ranks must be exactly 1.0/2.0/3.0 and the
    Friedman statistic must hit its theoretical max for perfect
    agreement: chi2 = n*(k-1) = 4*2 = 8 (Kendall's W = 1), giving
    p = chi2.sf(8, df=2) = exp(-4) (closed form for df=2)."""
    score_matrix = pd.DataFrame({
        "worst": [1, 2, 1, 3],
        "mid":   [5, 6, 4, 7],
        "best":  [9, 10, 8, 11],
    }, index=["ds1", "ds2", "ds3", "ds4"])
    result = rstats.friedman_test(score_matrix, higher_is_better=True)
    assert result.avg_ranks["best"] == pytest.approx(1.0)
    assert result.avg_ranks["mid"] == pytest.approx(2.0)
    assert result.avg_ranks["worst"] == pytest.approx(3.0)
    assert result.statistic == pytest.approx(8.0, abs=1e-9)
    assert result.p_value == pytest.approx(np.exp(-4), rel=1e-6)


# ---------------------------------------------------------------------------
# Paired comparison (Wilcoxon + t-test + effect sizes)
# ---------------------------------------------------------------------------


def test_paired_comparison_perfect_separation():
    """x uniformly 4 units above y -> rank-biserial must be exactly +1
    (every pair favors x), Cohen's d is +inf (zero variance in the
    diffs, nonzero mean), and the Wilcoxon two-sided p must be at its
    n=4 minimum (0.125 = 2 * (1/2)^4, the smallest attainable exact
    p-value with 4 matched pairs and no ties in sign)."""
    x = np.array([5.0, 6.0, 7.0, 8.0])
    y = np.array([1.0, 2.0, 3.0, 4.0])
    result = rstats.paired_comparison(x, y)
    assert result.rank_biserial == pytest.approx(1.0)
    assert result.cohens_d == float("inf")
    assert result.wilcoxon_p == pytest.approx(0.125, abs=1e-6)
    assert result.mean_diff == pytest.approx(4.0)


def test_paired_comparison_identical_arrays():
    x = np.array([0.9, 0.8, 0.95])
    result = rstats.paired_comparison(x, x)
    assert result.wilcoxon_p == pytest.approx(1.0)
    assert result.rank_biserial == pytest.approx(0.0)
    assert "zero" in result.note.lower()


def test_rank_biserial_matches_manual_formula():
    x = np.array([10, 12, 9, 15, 11])
    y = np.array([8, 13, 7, 14, 9])
    diff = x - y  # [2, -1, 2, 1, 2]
    ranks = sstats.rankdata(np.abs(diff))
    w_pos = ranks[diff > 0].sum()
    w_neg = ranks[diff < 0].sum()
    expected = (w_pos - w_neg) / (w_pos + w_neg)
    assert rstats.rank_biserial_wilcoxon(x, y) == pytest.approx(expected)


# ---------------------------------------------------------------------------
# t confidence interval
# ---------------------------------------------------------------------------


def test_t_confidence_interval_matches_scipy_interval():
    values = [0.80, 0.84, 0.79, 0.86, 0.81]
    ci = rstats.t_confidence_interval(values, confidence=0.95)
    mean = np.mean(values)
    sem = sstats.sem(values)
    lo_expected, hi_expected = sstats.t.interval(0.95, df=len(values) - 1, loc=mean, scale=sem)
    assert ci.mean == pytest.approx(mean)
    assert ci.lo == pytest.approx(lo_expected)
    assert ci.hi == pytest.approx(hi_expected)
    assert ci.n == 5


def test_t_confidence_interval_generic_across_seed_counts():
    """Must work for n=3 (current CSV) and n=10 (future T2.7 data)
    without any hardcoded seed count."""
    for n in (3, 10):
        values = np.random.RandomState(0).normal(0.8, 0.02, size=n)
        ci = rstats.t_confidence_interval(values)
        assert ci.n == n
        assert ci.lo < ci.mean < ci.hi


# ---------------------------------------------------------------------------
# Three-way ANOVA + Scheirer-Ray-Hare: hand-verified balanced 2x2x2 design
# ---------------------------------------------------------------------------


def _build_hand_verified_2x2x2():
    """A fully additive, balanced 2x2x2 factorial with 2 replicates/cell
    (N=16), constructed so every number below is hand-checkable and
    ties-free (so rank-transforming for SRH doesn't collide across
    cells).

    Construction: cell mean(a,b,c) = 9 + Ea(a) + Eb(b) + Ec(c), with
    Ea in {-4,+4}, Eb in {-2,+2}, Ec in {-1,+1} (purely additive -- no
    interaction terms at all), giving cell means exactly
    2,4,6,8,10,12,14,16 for the 8 (a,b,c) combinations in the order
    below. Each cell's 2 replicates are mean +/- 0.3 (small relative to
    the step-2 spacing between cells, so no replicate value ever ties
    with another cell's replicate -- ranks 1..16 land in the same
    order/spacing as the raw values).

    Hand-derived sums of squares (grand mean = 9 for raw values, 8.5 for
    the N=16 rank-transform of tie-free data 1..16):
      SS_A = 256 (df=1), SS_B = 64 (df=1), SS_C = 16 (df=1)
      SS_AB = SS_AC = SS_BC = SS_ABC = 0 (df=1 each) -- proved via the
        non-negative-SS-summing-to-zero argument: SS_total - SS_resid
        exactly equals SS_A+SS_B+SS_C on both the raw and the rank scale,
        so every interaction term (each >= 0) must individually be 0.
      Raw-value residual: SS_resid = 8 cells * 2*(0.3)^2 = 1.44 (df=8)
      Raw-value total: SS_total = 256+64+16+1.44 = 337.44 (df=15)
      Rank-transform (ties-free, ranks land as 1..16 in cell blocks of 2):
        SS_resid(ranks) = 8 cells * 2*(0.5)^2 = 4.0 (df=8)
        SS_total(ranks) = N(N^2-1)/12 = 16*255/12 = 340.0 (df=15)
    """
    cell_means = [2, 4, 6, 8, 10, 12, 14, 16]
    combos = [(a, b, c) for a in ("a1", "a2") for b in ("b1", "b2") for c in ("c1", "c2")]
    rows = []
    for (a, b, c), m in zip(combos, cell_means):
        for rep_offset in (-0.3, 0.3):
            rows.append({"ordering": a, "base_learner": b, "rebalance": c, "macro_f1": m + rep_offset})
    return pd.DataFrame(rows)


def test_three_way_anova_matches_hand_computed_sum_of_squares():
    df = _build_hand_verified_2x2x2()
    result = rstats.three_way_anova(df, "macro_f1", ("ordering", "base_learner", "rebalance"))
    table = result.table

    def ss(term):
        return table.loc[term, "sum_sq"]

    assert ss("C(ordering)") == pytest.approx(256.0, abs=1e-6)
    assert ss("C(base_learner)") == pytest.approx(64.0, abs=1e-6)
    assert ss("C(rebalance)") == pytest.approx(16.0, abs=1e-6)
    assert ss("C(ordering):C(base_learner)") == pytest.approx(0.0, abs=1e-6)
    assert ss("C(ordering):C(rebalance)") == pytest.approx(0.0, abs=1e-6)
    assert ss("C(base_learner):C(rebalance)") == pytest.approx(0.0, abs=1e-6)
    assert ss("C(ordering):C(base_learner):C(rebalance)") == pytest.approx(0.0, abs=1e-6)
    assert ss("Residual") == pytest.approx(1.44, abs=1e-6)
    assert table["sum_sq"].sum() == pytest.approx(337.44, abs=1e-6)

    # degrees of freedom: 1 for every main/interaction term, 8 residual
    for term in table.index:
        if term == "Residual":
            assert table.loc[term, "df"] == pytest.approx(8)
        else:
            assert table.loc[term, "df"] == pytest.approx(1)


def test_scheirer_ray_hare_matches_hand_computed_rank_sum_of_squares():
    df = _build_hand_verified_2x2x2()
    table = rstats.scheirer_ray_hare(df, "macro_f1", ("ordering", "base_learner", "rebalance"))

    def ss(term):
        return table.loc[term, "sum_sq"]

    assert ss("C(ordering)") == pytest.approx(256.0, abs=1e-6)
    assert ss("C(base_learner)") == pytest.approx(64.0, abs=1e-6)
    assert ss("C(rebalance)") == pytest.approx(16.0, abs=1e-6)
    assert ss("C(ordering):C(base_learner)") == pytest.approx(0.0, abs=1e-6)
    assert ss("Residual") == pytest.approx(4.0, abs=1e-6)
    assert table["sum_sq"].sum() == pytest.approx(340.0, abs=1e-6)

    ms_total = 340.0 / 15
    assert table.loc["C(ordering)", "H"] == pytest.approx(256.0 / ms_total, rel=1e-6)
    assert table.loc["C(base_learner)", "H"] == pytest.approx(64.0 / ms_total, rel=1e-6)
    assert table.loc["C(rebalance)", "H"] == pytest.approx(16.0 / ms_total, rel=1e-6)

    expected_p_a = sstats.chi2.sf(256.0 / ms_total, df=1)
    assert table.loc["C(ordering)", "p_value"] == pytest.approx(expected_p_a, rel=1e-6)
    # the strong main effect (A) must read as significant; the residual
    # row must not carry a chi-squared p-value at all
    assert table.loc["C(ordering)", "p_value"] < 0.001
    assert np.isnan(table.loc["Residual", "p_value"])


def test_anova_or_srh_runs_end_to_end_and_recommends_something():
    df = _build_hand_verified_2x2x2()
    result = rstats.anova_or_srh(df, "macro_f1", ("ordering", "base_learner", "rebalance"))
    assert result.recommended in ("anova", "srh")
    assert not result.srh_table.empty
    assert not result.anova.table.empty
    assert 0.0 <= result.anova.residual_shapiro_p <= 1.0


# ---------------------------------------------------------------------------
# format_p
# ---------------------------------------------------------------------------


def test_format_p_reporting_convention():
    assert rstats.format_p(0.123456) == "0.123"
    assert rstats.format_p(0.0456) == "0.0456"
    assert rstats.format_p(0.0001) == "<0.001"
    assert rstats.format_p(0.0009999) == "<0.001"
    assert rstats.format_p(float("nan")) == "n/a"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
