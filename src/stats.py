"""Statistics for judge validation and bias measurement. Pure functions, no I/O.

Cohen's kappa is easy to compute correctly and easy to compute *meaninglessly*,
so four guards are built in rather than left to the caller:

1. ``labels`` is always passed explicitly. sklearn weights on a label's POSITION
   in the label array rather than on its value, so when the class set is inferred
   from the observed union the gaps between scores collapse: with observed
   {1, 2, 5} the distance from 2 to 5 becomes 1 instead of 3, and a two-point
   disagreement is charged as a one-point one. Worth being precise about the
   scope, since this is usually stated too broadly: an evenly spaced observed set
   such as {3, 4, 5} rescales uniformly and cancels, leaving kappa unchanged. The
   distortion bites when the observed set is unevenly spaced -- which is what
   score clustering produces, and which varies run to run, so runs silently stop
   being comparable. Passing labels also fixes the confusion matrix at 5x5 so the
   before/after ablation pair can be read side by side. There is a test for both
   halves of this in tests/test_stats.py.

2. The signature takes ``(case_id, human, judge)`` triples, not two positional
   arrays. Two parallel lists is exactly the signature that admits a silent
   misalignment bug, and a toy-pair unit test would validate the formula while
   missing the alignment entirely. The triples are sorted by case id here, so
   input order cannot affect the result.

3. Degenerate marginals are detected and reported instead of being written into
   a report as ``NaN``. With both raters concentrated on {4, 5} at n=15, kappa is
   at its least interpretable (the Feinstein-Cicchetti paradox: high observed
   agreement, near-zero or negative kappa), and if either rater uses fewer than
   two distinct labels sklearn returns NaN outright.

4. Kappa is never returned alone. The result carries the confusion matrix, both
   marginal distributions, raw percent agreement, Spearman rho, and a percentile
   bootstrap CI -- because at n=10-20 the interval dominates the point estimate,
   and because the (rho, kappa) pair is what distinguishes a *mis-calibrated*
   judge (fixable by moving anchors or the threshold) from an *unreliable* one
   (not fixable by calibration).
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence

import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from scipy import stats as scipy_stats
from sklearn.metrics import cohen_kappa_score, confusion_matrix

from .schema import CorrelationResult, Interval, PairedTestResult, ScoreDistribution

DEFAULT_LABELS: tuple[int, ...] = (1, 2, 3, 4, 5)


# --------------------------------------------------------------------------
# Interval estimators
# --------------------------------------------------------------------------


def wilson_interval(successes: int, n: int, *, confidence: float = 0.95) -> Interval | None:
    """Wilson score interval -- the right binomial interval at small n, where
    the normal approximation is badly wrong and 0/5 or 5/5 are common."""
    if n <= 0:
        return None
    z = float(scipy_stats.norm.ppf(1 - (1 - confidence) / 2))
    p = successes / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))
    return Interval(
        low=max(0.0, center - half),
        high=min(1.0, center + half),
        method="wilson",
        confidence=confidence,
    )


def clopper_pearson_interval(
    successes: int, n: int, *, confidence: float = 0.95
) -> Interval | None:
    """Exact binomial interval. Used for the A/B win rate, where the honest
    result at n=20-25 is an interval wide enough to be visibly uninformative."""
    if n <= 0:
        return None
    alpha = 1 - confidence
    low = (
        0.0
        if successes == 0
        else float(scipy_stats.beta.ppf(alpha / 2, successes, n - successes + 1))
    )
    high = (
        1.0
        if successes == n
        else float(scipy_stats.beta.ppf(1 - alpha / 2, successes + 1, n - successes))
    )
    return Interval(low=low, high=high, method="clopper-pearson", confidence=confidence)


# --------------------------------------------------------------------------
# Correlation and distribution
# --------------------------------------------------------------------------


def spearman(x: Sequence[float], y: Sequence[float]) -> CorrelationResult:
    n = min(len(x), len(y))
    if n < 3:
        return CorrelationResult(n=n, reason="fewer than 3 paired observations")
    xa, ya = np.asarray(x[:n], dtype=float), np.asarray(y[:n], dtype=float)
    if np.all(xa == xa[0]) or np.all(ya == ya[0]):
        return CorrelationResult(n=n, reason="one variable has zero variance (constant)")
    res = scipy_stats.spearmanr(xa, ya)
    rho = float(res.statistic)
    p = float(res.pvalue)
    if math.isnan(rho):
        return CorrelationResult(n=n, reason="spearman returned NaN (degenerate ranks)")
    return CorrelationResult(n=n, rho=rho, p_value=p)


# --------------------------------------------------------------------------
# Paired significance tests
#
# Every design these cover scores the SAME cases under two conditions, so the
# observations are paired and an unpaired test would throw away the pairing --
# the main source of power at n=15-18. Wilcoxon rather than a paired t-test
# because 1-5 rubric scores are ordinal and small-n normality is not credible.
# --------------------------------------------------------------------------


def wilcoxon_paired(
    before: Sequence[float], after: Sequence[float], *, label: str = ""
) -> PairedTestResult:
    """Wilcoxon signed-rank over per-case deltas.

    Zero differences are dropped, which is the standard ("pratt" would keep
    them); with all-zero differences the test is undefined and says so rather
    than returning a p-value that looks like evidence of no effect.
    """
    n = min(len(before), len(after))
    if n < 3:
        return PairedTestResult(test="wilcoxon-signed-rank", n=n, reason="fewer than 3 pairs")
    diffs = [float(a) - float(b) for b, a in zip(before[:n], after[:n], strict=True)]
    non_zero = [d for d in diffs if d != 0.0]
    if not non_zero:
        return PairedTestResult(
            test="wilcoxon-signed-rank",
            n=0,
            effect=0.0,
            reason="every paired difference is exactly zero, so the test is undefined",
        )
    if len(non_zero) < 3:
        return PairedTestResult(
            test="wilcoxon-signed-rank",
            n=len(non_zero),
            effect=float(np.median(non_zero)),
            reason=f"only {len(non_zero)} non-tied pairs; too few for a meaningful test",
        )
    res = scipy_stats.wilcoxon(non_zero, alternative="two-sided")
    return PairedTestResult(
        test="wilcoxon-signed-rank" + (f" ({label})" if label else ""),
        n=len(non_zero),
        statistic=float(res.statistic),
        p_value=float(res.pvalue),
        effect=float(np.median(non_zero)),
    )


def mcnemar_exact(a_wins: int, b_wins: int, *, label: str = "") -> PairedTestResult:
    """Exact McNemar on the discordant pairs of a paired win/loss design.

    Only pairs where the two conditions disagree carry information; concordant
    pairs cancel. With n discordant, the null is a fair binomial split, so the
    exact two-sided binomial test is the right call at these sample sizes rather
    than the chi-square approximation.
    """
    n = a_wins + b_wins
    if n == 0:
        return PairedTestResult(
            test="mcnemar-exact", n=0, reason="no discordant pairs; the two conditions never differ"
        )
    res = scipy_stats.binomtest(b_wins, n=n, p=0.5, alternative="two-sided")
    return PairedTestResult(
        test="mcnemar-exact" + (f" ({label})" if label else ""),
        n=n,
        statistic=float(b_wins),
        p_value=float(res.pvalue),
        effect=float(b_wins / n),
    )


def holm_bonferroni(results: Sequence[PairedTestResult], *, alpha: float = 0.05) -> None:
    """Adjust a family of p-values in place, step-down Holm.

    Without this, ~15 comparisons at alpha=0.05 carry a family-wise error rate
    near 54%: with enough bias rows and probe categories, something crosses 0.05
    by chance alone. The honest outcome at these sample sizes is usually that
    most effects stop being significant, which is a finding rather than a
    weakness -- and stating it is what separates a measurement from a claim.
    """
    testable = [r for r in results if r.p_value is not None]
    if not testable:
        return
    m = len(testable)
    running = 0.0
    for rank, result in enumerate(sorted(testable, key=lambda r: r.p_value or 1.0)):
        adjusted = min(1.0, (m - rank) * (result.p_value or 1.0))
        # Step-down: adjusted p-values must be non-decreasing down the ranking.
        running = max(running, adjusted)
        result.adjusted_p_value = running
        # `<=`, not `<`: Holm rejects at adjusted p *at or below* alpha. With a
        # strict comparison an adjusted p of exactly 0.05 at alpha=0.05 reads as
        # non-significant, which is the wrong side of the convention and lands
        # exactly on the value most likely to appear at these sample sizes.
        result.significant = running <= alpha


def score_distribution(
    scores: Sequence[int], *, labels: Sequence[int] = DEFAULT_LABELS
) -> ScoreDistribution:
    """The score-clustering measurement: spread, mode dominance and entropy."""
    label_list = list(labels)
    # Scores outside the declared label set are dropped rather than silently
    # counted. Keeping them made the entropy probabilities sum to less than 1,
    # because the histogram admitted them but the entropy sum ranged over
    # `labels` only -- an inconsistency that is unreachable through the
    # pydantic Score type but not through a direct call.
    known = set(label_list)
    vals = [int(s) for s in scores if int(s) in known]
    n = len(vals)
    hist = {str(label): 0 for label in label_list}
    for v in vals:
        hist[str(v)] += 1
    if n == 0:
        return ScoreDistribution(n=0, histogram=hist)
    counts = Counter(vals)
    # Lowest label wins a tie, so the modal value does not depend on iteration
    # order when two levels are equally common.
    modal_count = max(counts.values())
    modal_value = min(v for v, c in counts.items() if c == modal_count)
    probs = [c / n for c in (hist[str(label)] for label in label_list) if c > 0]
    entropy_bits = -sum(p * math.log2(p) for p in probs)
    arr = np.asarray(vals, dtype=float)
    # The top two *declared* labels, not a hardcoded (4, 5): the scale is a
    # parameter, and a 1-7 rubric would otherwise report the share scoring 4-or-5
    # under a "top two" heading.
    top_two = sum(hist[str(label)] for label in label_list[-2:]) / n
    return ScoreDistribution(
        n=n,
        mean=float(arr.mean()),
        std_dev=float(arr.std(ddof=1)) if n > 1 else 0.0,
        modal_value=int(modal_value),
        modal_share=modal_count / n,
        entropy_bits=entropy_bits,
        top_two_share=top_two,
        histogram=hist,
    )


# --------------------------------------------------------------------------
# Agreement
# --------------------------------------------------------------------------


class KappaResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    n: int
    labels: list[int]
    kappa_quadratic: float | None = None
    kappa_ci: Interval | None = None
    percent_agreement: float | None = None
    within_one_agreement: float | None = None
    spearman: CorrelationResult | None = None
    confusion_matrix: list[list[int]] = Field(default_factory=list)
    human_marginals: dict[str, int] = Field(default_factory=dict)
    judge_marginals: dict[str, int] = Field(default_factory=dict)
    max_marginal_share: float | None = None
    degenerate: bool = False
    reason: str | None = None
    interpretation: str = ""
    case_ids: list[str] = Field(default_factory=list)


def _marginals(values: Sequence[int], labels: Sequence[int]) -> dict[str, int]:
    counts = Counter(int(v) for v in values)
    return {str(label): int(counts.get(label, 0)) for label in labels}


def _kappa_or_none(h: Sequence[int], j: Sequence[int], labels: Sequence[int]) -> float | None:
    # `or`, not `and`: with `labels` pinned, sklearn returns a hard 0.0 when
    # *one* rater is constant (only both-constant yields NaN). Under `and`
    # those resamples were folded into the bootstrap as real zeros, biasing
    # the CI downward -- and the "too few valid resamples" guard below could
    # never fire, because it required both raters constant in >=90% of draws.
    if len(set(h)) < 2 or len(set(j)) < 2:
        return None
    value = cohen_kappa_score(list(h), list(j), labels=list(labels), weights="quadratic")
    return None if (value is None or math.isnan(float(value))) else float(value)


def interpret_kappa(
    kappa: float | None,
    percent_agreement: float | None,
    rho: float | None,
    max_marginal_share: float | None,
) -> str:
    """Pre-committed interpretation rule, written before any number existed.

    Deciding after the fact what a low kappa "really means" is how a validation
    section turns into an apology, so the mapping is fixed here.
    """
    if kappa is None:
        return (
            "Kappa is undefined for this sample (a rater used fewer than two distinct labels). "
            "Report the confusion matrix and percent agreement instead; do not report 0.0."
        )
    concentrated = (max_marginal_share or 0.0) >= 0.60
    high_po = (percent_agreement or 0.0) >= 0.70
    if kappa < 0.40 and high_po and concentrated:
        return (
            "Low kappa with high raw agreement and concentrated marginals: this is marginal "
            "skew (the kappa paradox), not judge failure. Read the confusion matrix, not the "
            "coefficient."
        )
    if kappa < 0.40 and (rho or 0.0) >= 0.60:
        return (
            "Low kappa with high rank correlation: the judge is MIS-CALIBRATED, not unreliable. "
            "It orders cases correctly but sits at the wrong absolute level -- fixable by moving "
            "the anchors or the pass threshold."
        )
    if kappa < 0.40:
        return (
            "Low kappa with low rank correlation: the judge is UNRELIABLE on this sample. This "
            "is not fixable by recalibration."
        )
    if kappa < 0.60:
        return "Moderate agreement. Usable as a signal, not as a sole release gate at this n."
    return (
        "Substantial agreement, subject to the confidence interval below, which at this n is wide."
    )


def calculate_kappa(
    pairs: Sequence[tuple[str, int, int]],
    *,
    labels: Sequence[int] = DEFAULT_LABELS,
    bootstrap_resamples: int = 2000,
    seed: int = 20260726,
    confidence: float = 0.95,
) -> KappaResult:
    """Quadratic-weighted Cohen's kappa with its companion statistics.

    ``pairs`` is ``(case_id, human_score, judge_score)``. Sorted internally, so
    shuffling the input yields a bit-identical result including the bootstrap CI.
    """
    ordered = sorted(pairs, key=lambda t: str(t[0]))
    case_ids = [str(t[0]) for t in ordered]
    human = [int(t[1]) for t in ordered]
    judge = [int(t[2]) for t in ordered]
    n = len(ordered)
    label_list = [int(x) for x in labels]

    result = KappaResult(
        n=n,
        labels=label_list,
        case_ids=case_ids,
        human_marginals=_marginals(human, label_list),
        judge_marginals=_marginals(judge, label_list),
    )
    if n == 0:
        result.degenerate = True
        result.reason = "no paired observations"
        result.interpretation = interpret_kappa(None, None, None, None)
        return result

    result.percent_agreement = sum(1 for a, b in zip(human, judge, strict=True) if a == b) / n
    result.within_one_agreement = (
        sum(1 for a, b in zip(human, judge, strict=True) if abs(a - b) <= 1) / n
    )
    result.confusion_matrix = confusion_matrix(human, judge, labels=label_list).tolist()
    result.spearman = spearman(human, judge)
    result.max_marginal_share = max(
        max(result.human_marginals.values()) / n,
        max(result.judge_marginals.values()) / n,
    )

    if len(set(human)) < 2 or len(set(judge)) < 2:
        result.degenerate = True
        result.reason = (
            "at least one rater used fewer than two distinct labels "
            f"(human distinct={sorted(set(human))}, judge distinct={sorted(set(judge))}); "
            "quadratic-weighted kappa is undefined or meaningless here"
        )
        result.interpretation = interpret_kappa(None, result.percent_agreement, None, None)
        return result

    result.kappa_quadratic = _kappa_or_none(human, judge, label_list)

    # Percentile bootstrap over cases. Zero API cost: it resamples stored scores.
    if result.kappa_quadratic is not None and n >= 3 and bootstrap_resamples > 0:
        rng = np.random.default_rng(seed)
        draws: list[float] = []
        idx = np.arange(n)
        for _ in range(bootstrap_resamples):
            sample = rng.choice(idx, size=n, replace=True)
            h = [human[i] for i in sample]
            j = [judge[i] for i in sample]
            val = _kappa_or_none(h, j, label_list)
            if val is not None:
                draws.append(val)
        if len(draws) >= max(100, bootstrap_resamples // 10):
            # Derive the percentiles from `confidence`; hardcoding 2.5/97.5
            # stamped a 95% interval with whatever label the caller asked for.
            tail = (1.0 - confidence) / 2.0 * 100.0
            lo, hi = np.percentile(draws, [tail, 100.0 - tail])
            result.kappa_ci = Interval(
                low=float(lo),
                high=float(hi),
                method=f"percentile bootstrap ({len(draws)}/{bootstrap_resamples} valid resamples)",
                confidence=confidence,
            )
        else:
            result.reason = (
                f"bootstrap CI omitted: only {len(draws)}/{bootstrap_resamples} resamples were "
                "non-degenerate, which means the sample is too concentrated for a stable interval"
            )

    result.interpretation = interpret_kappa(
        result.kappa_quadratic,
        result.percent_agreement,
        result.spearman.rho if result.spearman else None,
        result.max_marginal_share,
    )
    return result


# --------------------------------------------------------------------------
# Test-retest reliability
# --------------------------------------------------------------------------


class ICCResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    n_subjects: int
    n_raters: int
    icc_1_1: float | None = None
    mean_within_subject_sd: float | None = None
    reason: str | None = None


def icc_1_1(matrix: Sequence[Sequence[float]]) -> ICCResult:
    """One-way random-effects ICC(1,1): the share of total variance that is
    between-case. This is *the* reliability coefficient for "how much of what
    the judge reports is signal", and it is reported instead of a rounded flip
    rate because flip rate on a thresholded pass/fail is a threshold artifact --
    a judge with tiny noise sitting on the pass boundary shows a huge flip rate,
    and a judge with large noise far from the boundary shows zero.
    """
    arr = np.asarray([[float(v) for v in row] for row in matrix], dtype=float)
    if arr.ndim != 2 or arr.shape[0] < 2 or arr.shape[1] < 2:
        return ICCResult(
            n_subjects=int(arr.shape[0]) if arr.ndim == 2 else 0,
            n_raters=int(arr.shape[1]) if arr.ndim == 2 else 0,
            reason="need at least 2 subjects and 2 runs",
        )
    n, k = arr.shape
    subject_means = arr.mean(axis=1)
    grand = arr.mean()
    ss_between = k * float(((subject_means - grand) ** 2).sum())
    ss_within = float(((arr - subject_means[:, None]) ** 2).sum())
    ms_between = ss_between / (n - 1)
    ms_within = ss_within / (n * (k - 1))
    denom = ms_between + (k - 1) * ms_within
    within_sd = float(np.mean(arr.std(axis=1, ddof=1))) if k > 1 else 0.0
    if denom == 0:
        return ICCResult(
            n_subjects=n,
            n_raters=k,
            mean_within_subject_sd=within_sd,
            reason="zero total variance: every run returned the same score for every case",
        )
    return ICCResult(
        n_subjects=n,
        n_raters=k,
        icc_1_1=float((ms_between - ms_within) / denom),
        mean_within_subject_sd=within_sd,
    )
