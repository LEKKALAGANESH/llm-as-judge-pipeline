"""Aggregator arithmetic against hand-computed fixtures.

This is the reason ``runner`` and ``aggregator`` are separate modules: every
number behind the two largest rubric lines is checked here with no network, no
API key and no mock -- just a list of verdicts and a expected answer worked out
by hand in the assertion.
"""

from __future__ import annotations

import pytest

from src.aggregator import (
    DECISION_RULE_TEXT,
    aggregate,
    decide_winner,
    mean_scores_by_criterion,
    select_mode,
    tally_pairwise,
)
from src.errors import PipelineError
from src.schema import (
    CallAccounting,
    CriterionScore,
    PairwiseCaseResult,
    PointwiseCaseResult,
    PointwiseVerdict,
)

CRITERIA = ["correctness", "safety"]


def pw(case_id: str, scores: dict[str, int], overall: int, length: int = 100, **kw):
    return PointwiseCaseResult(
        case_id=case_id,
        verdict=PointwiseVerdict(
            criteria_breakdown={
                k: CriterionScore(rationale="r", score=v) for k, v in scores.items()
            },
            overall_rationale="r",
            overall_score=overall,
        ),
        output_length=length,
        accounting=CallAccounting(api_calls=1, attempts=1, prompt_tokens=100, completion_tokens=20),
        **kw,
    )


def prs(case_id: str, winner, *, consistent=True, incomplete=False, **kw):
    return PairwiseCaseResult(
        case_id=case_id,
        final_winner=winner,
        position_consistent=None if incomplete else consistent,
        incomplete=incomplete,
        accounting=CallAccounting(api_calls=2, attempts=2),
        **kw,
    )


# --------------------------------------------------------------------------
# Mode selection -- one rule, one place, four combinations
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "comparing,expected_output,expected",
    [
        (True, None, "pairwise"),
        (True, "ref", "pairwise"),
        (False, "ref", "pointwise_reference_based"),
        (False, None, "pointwise_reference_free"),
    ],
)
def test_mode_selection_rule(comparing, expected_output, expected):
    assert select_mode(comparing_two_configs=comparing, expected_output=expected_output) == expected


# --------------------------------------------------------------------------
# Pointwise arithmetic
# --------------------------------------------------------------------------


def test_pass_rate_uses_both_thresholds_and_is_recomputed_from_config():
    verdicts = [
        pw("a", {"correctness": 5, "safety": 5}, 5),  # passes
        pw("b", {"correctness": 5, "safety": 2}, 5),  # fails: weak criterion
        pw("c", {"correctness": 3, "safety": 3}, 3),  # fails: overall below bar
        pw("d", {"correctness": 4, "safety": 3}, 4),  # passes
    ]
    report = aggregate(
        verdicts,
        run_id="r",
        generated_at="t",
        pass_threshold=4,
        min_criterion_score=3,
    )
    assert report.n_scored == 4
    assert report.n_passed == 2
    assert report.pass_rate == 0.5

    # Re-tunable without re-spending the suite: that is why it is recomputed here
    # rather than read off the stored verdict.
    relaxed = aggregate(
        verdicts, run_id="r", generated_at="t", pass_threshold=3, min_criterion_score=2
    )
    assert relaxed.n_passed == 4
    strict = aggregate(
        verdicts, run_id="r", generated_at="t", pass_threshold=5, min_criterion_score=5
    )
    assert strict.n_passed == 1


def test_mean_scores_by_criterion_is_per_criterion_not_pooled():
    verdicts = [
        pw("a", {"correctness": 5, "safety": 1}, 3),
        pw("b", {"correctness": 3, "safety": 3}, 3),
    ]
    means = mean_scores_by_criterion(verdicts)
    assert means == {"correctness": 4.0, "safety": 2.0}


def test_failures_are_counted_and_excluded_never_silently_dropped():
    verdicts = [
        pw("a", {"correctness": 5, "safety": 5}, 5),
        PointwiseCaseResult(case_id="b", failure="judge_parse_error", error="bad json"),
        PointwiseCaseResult(case_id="c", failure="judge_call_failed", error="429"),
    ]
    report = aggregate(verdicts, run_id="r", generated_at="t")
    assert report.n_cases == 3
    assert report.n_scored == 1
    assert report.n_parse_errors == 1
    assert report.n_call_failures == 1
    assert sorted(report.failed_case_ids) == ["b", "c"]
    assert report.pass_rate == 1.0, "the rate is over scored cases, and the exclusions are visible"


def test_score_distribution_measures_clustering():
    clustered = [pw(f"c{i}", {"correctness": 4, "safety": 4}, 4 + (i % 2)) for i in range(10)]
    spread = [pw(f"s{i}", {"correctness": 3, "safety": 3}, 1 + (i % 5)) for i in range(10)]

    c = aggregate(clustered, run_id="r", generated_at="t").overall_score_distribution
    s = aggregate(spread, run_id="r", generated_at="t").overall_score_distribution

    assert c.top_two_share == 1.0 and s.top_two_share == pytest.approx(0.4)
    assert c.entropy_bits < s.entropy_bits
    assert c.std_dev < s.std_dev
    assert c.modal_share >= s.modal_share
    assert sum(c.histogram.values()) == 10


def test_length_vs_score_correlation_is_computed_from_stored_lengths():
    rising = [
        pw(f"c{i}", {"correctness": 3, "safety": 3}, 1 + i, length=100 * (i + 1)) for i in range(5)
    ]
    report = aggregate(rising, run_id="r", generated_at="t")
    assert report.length_vs_score_spearman.rho == pytest.approx(1.0)
    assert report.length_vs_score_spearman.n == 5


def test_accounting_is_summed_across_cases_and_costed():
    verdicts = [
        pw("a", {"correctness": 4, "safety": 4}, 4),
        pw("b", {"correctness": 4, "safety": 4}, 4),
    ]
    report = aggregate(verdicts, run_id="r", generated_at="t", price_per_million=(1.0, 10.0))
    assert report.total_judge_calls == 2
    assert report.total_prompt_tokens == 200
    assert report.total_completion_tokens == 40
    assert report.estimated_cost_usd == pytest.approx((200 * 1.0 + 40 * 10.0) / 1_000_000)


# --------------------------------------------------------------------------
# Pairwise arithmetic
# --------------------------------------------------------------------------


def test_tally_denominators_are_exactly_as_specified():
    verdicts = [
        prs("1", "Model_A"),
        prs("2", "Model_B"),
        prs("3", "Model_B"),
        prs("4", "Tie"),
        prs("5", "Tie (Position Inconsistency)", consistent=False),
        prs("6", None, incomplete=True),
    ]
    t = tally_pairwise(verdicts)
    assert (t.wins_a, t.wins_b, t.ties) == (1, 2, 1)
    assert t.inconsistent_pairs == 1
    assert t.incomplete_pairs == 1
    # completed excludes only the incomplete pair; resolved additionally excludes
    # ties and position-inconsistent pairs.
    assert t.completed_pairs == 5
    assert t.resolved_pairs == 3
    assert t.flip_rate == pytest.approx(1 / 5)
    assert t.win_rate_b == pytest.approx(2 / 3)
    assert t.win_rate_b_ci is not None and t.win_rate_b_ci.method == "clopper-pearson"


def test_incomplete_pairs_are_excluded_from_the_flip_rate_denominator():
    with_incomplete = tally_pairwise(
        [
            prs("1", "Tie (Position Inconsistency)", consistent=False),
            prs("2", None, incomplete=True),
        ]
    )
    assert with_incomplete.flip_rate == 1.0, "1 inconsistent / 1 completed, not 1/2"


def test_decision_rule_short_circuits_on_the_validity_gate():
    t = tally_pairwise(
        [prs(str(i), "Model_B") for i in range(6)]
        + [prs(f"x{i}", "Tie (Position Inconsistency)", consistent=False) for i in range(4)]
    )
    assert t.flip_rate == pytest.approx(0.4)
    winner, trace = decide_winner(t, flip_rate_threshold=0.20, win_rate_threshold=0.55)
    assert winner == "Judge unreliable on this suite - no winner declared"
    assert "short-circuit" in " ".join(trace)
    assert "Step 2" not in " ".join(trace), "the effect estimate must not be evaluated at all"


def test_coverage_gate_blocks_a_winner_declared_from_a_handful_of_survivors():
    """The failure that actually shipped: 3 of 15 pairs resolved, 12 lost to rate
    limiting, and a clean 0/3 flip rate sailed through the validity gate.

    The flip rate is computed over *completed* pairs, so it cannot see what is
    missing -- and the missing pairs are not a random sample, because the largest
    prompts fail first. Coverage therefore has to be checked before validity.
    """
    resolved = [prs(str(i), "Model_B") for i in range(3)]
    incomplete = [prs(f"x{i}", None, incomplete=True) for i in range(12)]
    t = tally_pairwise(resolved + incomplete)

    assert t.completed_pairs == 3
    assert t.flip_rate == 0.0, "the survivors look perfectly consistent"

    winner, trace = decide_winner(t, flip_rate_threshold=0.20, win_rate_threshold=0.55)

    assert winner == "Insufficient coverage - no winner declared"
    joined = " ".join(trace)
    assert "Step 0 coverage gate" in joined
    assert "Step 1" not in joined, "validity must not be reached on an unrepresentative sample"


def test_coverage_gate_allows_a_run_that_completed_most_pairs():
    t = tally_pairwise(
        [prs(str(i), "Model_B") for i in range(8)] + [prs("x", None, incomplete=True)]
    )
    winner, trace = decide_winner(
        t, flip_rate_threshold=0.20, win_rate_threshold=0.55, config_b_label="prompt_v2"
    )
    assert winner == "prompt_v2"
    assert "Coverage gate passed" in " ".join(trace)


def test_decision_rule_declares_a_winner_above_the_threshold():
    t = tally_pairwise([prs(str(i), "Model_B") for i in range(8)] + [prs("a", "Model_A")])
    winner, trace = decide_winner(
        t, flip_rate_threshold=0.20, win_rate_threshold=0.55, config_b_label="prompt_v2"
    )
    assert winner == "prompt_v2"
    assert "Gate passed" in " ".join(trace)


def test_decision_rule_is_inconclusive_when_the_margin_is_thin():
    # 6/11 = 0.545, just under the 0.55 bar. The threshold is a real bar, not a
    # rounding suggestion.
    t = tally_pairwise(
        [prs(str(i), "Model_B") for i in range(6)] + [prs(f"a{i}", "Model_A") for i in range(5)]
    )
    assert t.win_rate_b == pytest.approx(6 / 11)
    winner, _ = decide_winner(t, flip_rate_threshold=0.20, win_rate_threshold=0.55)
    assert winner == "Inconclusive - no significant difference"


def test_all_ties_is_inconclusive_not_a_win_for_either_side():
    t = tally_pairwise([prs(str(i), "Tie") for i in range(5)])
    winner, _ = decide_winner(t, flip_rate_threshold=0.20, win_rate_threshold=0.55)
    assert winner == "Inconclusive - no pair resolved to a winner"


def test_no_completed_pairs_is_distinguished_from_inconclusive():
    t = tally_pairwise([prs("1", None, incomplete=True)])
    winner, _ = decide_winner(t, flip_rate_threshold=0.20, win_rate_threshold=0.55)
    assert winner == "No winner declared - no completed pairs"


def test_decision_rule_text_has_no_mean_score_term():
    """A pairwise-only run carries no numeric score, so a mean-score criterion
    would not be computable from the run it is applied to."""
    assert "no mean-score term" in DECISION_RULE_TEXT.lower()
    assert "mean_score >" not in DECISION_RULE_TEXT


def test_pairwise_report_carries_the_rule_and_a_recomputable_trace():
    report = aggregate(
        [prs("1", "Model_B"), prs("2", "Model_B"), prs("3", "Model_A")],
        run_id="r",
        generated_at="t",
        config_a_label="v1",
        config_b_label="v2",
    )
    assert report.mode == "pairwise"
    assert report.decision_rule == DECISION_RULE_TEXT
    assert report.decision_trace
    assert report.tally.wins_b == 2


# --------------------------------------------------------------------------
# Purity
# --------------------------------------------------------------------------


def test_aggregate_is_deterministic_and_side_effect_free(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    verdicts = [pw("a", {"correctness": 4, "safety": 4}, 4)]
    first = aggregate(verdicts, run_id="r", generated_at="t")
    second = aggregate(verdicts, run_id="r", generated_at="t")
    assert first == second
    assert list(tmp_path.iterdir()) == [], "the aggregator must not write anything"


def test_mixed_verdict_streams_are_refused():
    with pytest.raises(PipelineError):
        aggregate([pw("a", {"correctness": 4}, 4), prs("b", "Tie")], run_id="r", generated_at="t")
