"""Statistics: kappa (formula, alignment, degeneracy, labels), intervals, ICC."""

from __future__ import annotations

import math
import random

import pytest
from sklearn.metrics import cohen_kappa_score

from src.stats import (
    calculate_kappa,
    clopper_pearson_interval,
    icc_1_1,
    interpret_kappa,
    score_distribution,
    spearman,
    wilson_interval,
)

# --------------------------------------------------------------------------
# Kappa: formula
# --------------------------------------------------------------------------


def test_perfect_agreement_is_one():
    pairs = [(f"c{i}", s, s) for i, s in enumerate([1, 2, 3, 4, 5, 5, 4, 3])]
    res = calculate_kappa(pairs, bootstrap_resamples=200)
    assert res.kappa_quadratic == pytest.approx(1.0)
    assert res.percent_agreement == 1.0


def test_matches_sklearn_on_a_hand_built_pair():
    human = [1, 2, 3, 4, 5, 3, 2, 4, 5, 1]
    judge = [2, 2, 3, 5, 5, 3, 1, 4, 4, 2]
    pairs = [(f"c{i}", h, j) for i, (h, j) in enumerate(zip(human, judge, strict=True))]
    expected = cohen_kappa_score(human, judge, labels=[1, 2, 3, 4, 5], weights="quadratic")
    res = calculate_kappa(pairs, bootstrap_resamples=200)
    assert res.kappa_quadratic == pytest.approx(float(expected))
    assert res.percent_agreement == pytest.approx(0.5)
    assert res.within_one_agreement == pytest.approx(1.0)


def test_labels_are_always_passed_explicitly_so_kappa_stays_comparable():
    """sklearn weights on the POSITION of a label in the label array, not on its
    value. Let it infer the classes from the observed union and the gaps between
    scores collapse: with observed {1, 2, 5} the distance from 2 to 5 becomes 1
    instead of 3, so a two-point disagreement is charged as a one-point one.

    Note the effect is narrower than it is usually stated. When the observed set
    is evenly spaced (say {3, 4, 5}) the compression is a uniform rescale and it
    cancels out of the ratio, leaving kappa unchanged. It bites exactly when the
    observed set is unevenly spaced -- which is what score clustering produces,
    and which differs from run to run, so runs stop being comparable.
    """
    human = [1, 2, 5, 2, 1, 5, 2, 1]
    judge = [2, 2, 5, 5, 1, 5, 2, 2]
    pairs = [(f"c{i}", h, j) for i, (h, j) in enumerate(zip(human, judge, strict=True))]
    ours = calculate_kappa(pairs, bootstrap_resamples=0).kappa_quadratic
    with_labels = cohen_kappa_score(human, judge, labels=[1, 2, 3, 4, 5], weights="quadratic")
    without_labels = cohen_kappa_score(human, judge, weights="quadratic")

    assert ours == pytest.approx(float(with_labels))
    assert not math.isclose(float(with_labels), float(without_labels)), (
        "this fixture exists precisely because the two differ; if it stops differing the test "
        "has lost its point"
    )
    assert calculate_kappa(pairs, bootstrap_resamples=0).labels == [1, 2, 3, 4, 5]


def test_evenly_spaced_observed_labels_do_not_change_kappa():
    """The honest companion to the test above: the usual blanket claim that
    omitting `labels` always distorts kappa is false, and it is worth knowing
    which case actually matters."""
    human = [3, 4, 5, 4, 3, 5, 4, 3]
    judge = [4, 4, 5, 5, 3, 5, 4, 4]
    with_labels = cohen_kappa_score(human, judge, labels=[1, 2, 3, 4, 5], weights="quadratic")
    without_labels = cohen_kappa_score(human, judge, weights="quadratic")
    assert math.isclose(float(with_labels), float(without_labels))


# --------------------------------------------------------------------------
# Kappa: alignment
# --------------------------------------------------------------------------


def test_shuffling_the_input_yields_a_bit_identical_result():
    """The signature takes (case_id, human, judge) triples precisely so that a
    misalignment bug cannot hide behind a plausible-looking number. Sorting
    internally makes the bootstrap CI reproducible too, not just the estimate."""
    pairs = [(f"c{i:02d}", (i % 5) + 1, ((i * 3) % 5) + 1) for i in range(20)]
    shuffled = pairs[:]
    random.Random(1).shuffle(shuffled)

    a = calculate_kappa(pairs, bootstrap_resamples=500, seed=7)
    b = calculate_kappa(shuffled, bootstrap_resamples=500, seed=7)
    assert a.kappa_quadratic == pytest.approx(b.kappa_quadratic)
    assert a.model_dump() == b.model_dump()


def test_a_misaligned_pairing_produces_a_different_kappa():
    """Sanity check on the test above: if alignment did not matter, the guard
    would be pointless."""
    aligned = [(f"c{i}", (i % 5) + 1, (i % 5) + 1) for i in range(15)]
    misaligned = [(f"c{i}", (i % 5) + 1, ((i + 1) % 5) + 1) for i in range(15)]
    assert calculate_kappa(aligned, bootstrap_resamples=0).kappa_quadratic == pytest.approx(1.0)
    assert calculate_kappa(misaligned, bootstrap_resamples=0).kappa_quadratic < 0.9


# --------------------------------------------------------------------------
# Kappa: degeneracy guards
# --------------------------------------------------------------------------


def test_single_label_rater_yields_null_kappa_with_a_stated_reason_not_nan():
    pairs = [(f"c{i}", 4, 4) for i in range(10)]
    res = calculate_kappa(pairs, bootstrap_resamples=0)
    assert res.kappa_quadratic is None
    assert res.degenerate is True
    assert "fewer than two distinct labels" in res.reason
    assert res.percent_agreement == 1.0, "raw agreement is still reportable and still informative"
    assert "undefined" in res.interpretation


def test_perfect_deterministic_offset_gives_zero_kappa_and_is_explained():
    """Judge all-4s against human all-5s is a perfect mapping and kappa 0.0 --
    the classic reading of that as catastrophic failure is what the companion
    statistics exist to prevent."""
    pairs = [(f"c{i}", 5, 4) for i in range(10)]
    res = calculate_kappa(pairs, bootstrap_resamples=0)
    assert res.kappa_quadratic is None or res.kappa_quadratic == pytest.approx(0.0)
    assert res.percent_agreement == 0.0
    assert res.within_one_agreement == 1.0


def test_empty_input_does_not_explode():
    res = calculate_kappa([], bootstrap_resamples=0)
    assert res.n == 0 and res.kappa_quadratic is None and res.degenerate


def test_companion_statistics_are_always_present():
    pairs = [(f"c{i}", (i % 5) + 1, ((i + 1) % 5) + 1) for i in range(20)]
    res = calculate_kappa(pairs, bootstrap_resamples=300, seed=3)
    assert len(res.confusion_matrix) == 5 and len(res.confusion_matrix[0]) == 5
    assert sum(sum(row) for row in res.confusion_matrix) == 20
    assert set(res.human_marginals) == {"1", "2", "3", "4", "5"}
    assert res.spearman is not None
    assert res.kappa_ci is not None and res.kappa_ci.low <= res.kappa_quadratic <= res.kappa_ci.high
    assert "bootstrap" in res.kappa_ci.method


def test_interpretation_rule_distinguishes_miscalibration_from_unreliability():
    assert "MIS-CALIBRATED" in interpret_kappa(0.2, 0.3, 0.8, 0.3)
    assert "UNRELIABLE" in interpret_kappa(0.2, 0.3, 0.1, 0.3)
    assert "marginal skew" in interpret_kappa(0.2, 0.85, 0.4, 0.8)
    assert "undefined" in interpret_kappa(None, None, None, None)


# --------------------------------------------------------------------------
# Intervals
# --------------------------------------------------------------------------


def test_wilson_interval_is_sane_at_the_boundaries():
    full = wilson_interval(5, 5)
    assert full.high == pytest.approx(1.0, abs=1e-9) or full.high < 1.0
    assert full.low < 1.0, "5/5 must not report a lower bound of 1.0 at n=5"
    zero = wilson_interval(0, 5)
    assert zero.low == pytest.approx(0.0, abs=1e-9) or zero.low > 0
    assert zero.high > 0.0
    assert wilson_interval(0, 0) is None


def test_clopper_pearson_brackets_the_estimate_and_is_wide_at_small_n():
    ci = clopper_pearson_interval(14, 20)
    assert ci.low < 0.7 < ci.high
    assert ci.high - ci.low > 0.3, "at n=20 the interval should be visibly uninformative"
    assert clopper_pearson_interval(20, 20).high == pytest.approx(1.0)
    assert clopper_pearson_interval(0, 20).low == pytest.approx(0.0)


# --------------------------------------------------------------------------
# Distributions, correlation, ICC
# --------------------------------------------------------------------------


def test_score_distribution_counts_every_level_even_the_unused_ones():
    d = score_distribution([4, 4, 4, 5])
    assert d.histogram == {"1": 0, "2": 0, "3": 0, "4": 3, "5": 1}
    assert d.modal_value == 4 and d.modal_share == 0.75
    assert d.top_two_share == 1.0
    assert d.entropy_bits == pytest.approx(-(0.75 * math.log2(0.75) + 0.25 * math.log2(0.25)))


def test_spearman_guards_degenerate_input():
    assert spearman([1, 2], [1, 2]).rho is None
    assert "zero variance" in spearman([1, 1, 1, 1], [1, 2, 3, 4]).reason
    assert spearman([1, 2, 3, 4], [1, 2, 3, 4]).rho == pytest.approx(1.0)
    assert spearman([1, 2, 3, 4], [4, 3, 2, 1]).rho == pytest.approx(-1.0)


def test_icc_is_high_when_between_case_variance_dominates():
    stable = [[5, 5, 5], [1, 1, 1], [3, 3, 3], [4, 4, 4]]
    noisy = [[5, 1, 3], [1, 5, 3], [3, 1, 5], [4, 2, 1]]
    assert icc_1_1(stable).icc_1_1 == pytest.approx(1.0)
    assert icc_1_1(noisy).icc_1_1 < 0.5
    assert icc_1_1(stable).mean_within_subject_sd == pytest.approx(0.0)


def test_icc_reports_a_reason_instead_of_dividing_by_zero():
    flat = icc_1_1([[4, 4], [4, 4]])
    assert flat.icc_1_1 is None and "zero total variance" in flat.reason
    assert icc_1_1([[4, 4]]).reason.startswith("need at least")
