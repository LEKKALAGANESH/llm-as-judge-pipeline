"""Position-bias protocol: remapping, resolution, and the one-order-failed rule."""

from __future__ import annotations

import pytest

from src.judge import JudgeOutcome
from src.mitigations import (
    POSITION_INCONSISTENT,
    build_confidently_wrong_probe,
    build_padded_probe,
    evaluate_pairwise_unbiased,
    remap_reverse_winner,
)
from src.schema import CallAccounting, PairwiseVerdict, TestCase


def _outcome(winner: str | None, *, failure: str | None = None) -> JudgeOutcome:
    verdict = PairwiseVerdict(rationale="because", winner=winner) if winner else None
    return JudgeOutcome(
        verdict=verdict,
        accounting=CallAccounting(api_calls=1, attempts=1, prompt_tokens=10, completion_tokens=5),
        failure=failure,  # type: ignore[arg-type]
        error=None if winner else "boom",
    )


def _fn(forward: str | None, reverse: str | None):
    calls = []

    def judge_pair(a: str, b: str, order: str) -> JudgeOutcome:
        calls.append((a, b, order))
        return _outcome(
            forward if order == "forward" else reverse,
            failure=None if (forward if order == "forward" else reverse) else "judge_parse_error",
        )

    return judge_pair, calls


@pytest.mark.parametrize(
    "reverse,expected",
    [("Model_A", "Model_B"), ("Model_B", "Model_A"), ("Tie", "Tie")],
)
def test_remap_translates_the_reversed_frame(reverse, expected):
    assert remap_reverse_winner(reverse) == expected


def test_agreement_resolves_to_that_winner():
    # forward says A wins; reverse (B shown first) says B wins -> remaps to A.
    fn, calls = _fn("Model_A", "Model_B")
    res = evaluate_pairwise_unbiased(fn, case_id="c1", output_a="AAA", output_b="BBB")
    assert res.forward_winner == "Model_A"
    assert res.reverse_winner == "Model_A"
    assert res.final_winner == "Model_A"
    assert res.position_consistent is True
    assert res.incomplete is False
    assert len(calls) == 2, "exactly two judge calls per pairwise case"
    assert calls[0][:2] == ("AAA", "BBB")
    assert calls[1][:2] == ("BBB", "AAA"), "the reverse call must actually swap the outputs"


def test_disagreement_resolves_to_position_inconsistency():
    # Both orders name whatever was shown first: the definition of position bias.
    fn, _ = _fn("Model_A", "Model_A")
    res = evaluate_pairwise_unbiased(fn, case_id="c2", output_a="AAA", output_b="BBB")
    assert res.reverse_winner == "Model_B"
    assert res.final_winner == POSITION_INCONSISTENT
    assert res.position_consistent is False
    assert res.incomplete is False


def test_tie_in_both_orders_is_a_consistent_tie():
    fn, _ = _fn("Tie", "Tie")
    res = evaluate_pairwise_unbiased(fn, case_id="c3", output_a="X", output_b="X")
    assert res.final_winner == "Tie"
    assert res.position_consistent is True


@pytest.mark.parametrize("forward,reverse", [("Model_A", None), (None, "Model_B"), (None, None)])
def test_one_order_failed_is_never_resolved_from_the_surviving_order(forward, reverse):
    fn, _ = _fn(forward, reverse)
    res = evaluate_pairwise_unbiased(fn, case_id="c4", output_a="AAA", output_b="BBB")
    assert res.incomplete is True
    assert res.final_winner is None, "resolving from one order reintroduces the bias being removed"
    assert res.position_consistent is None
    assert res.error and ("forward" in res.error or "reverse" in res.error)


def test_accounting_is_summed_across_both_orders():
    fn, _ = _fn("Model_A", "Model_B")
    res = evaluate_pairwise_unbiased(fn, case_id="c5", output_a="A", output_b="B")
    assert res.accounting.api_calls == 2
    assert res.accounting.prompt_tokens == 20


def test_order_swap_off_is_the_ablation_baseline_and_costs_one_call():
    fn, calls = _fn("Model_A", "Model_B")
    res = evaluate_pairwise_unbiased(
        fn, case_id="c6", output_a="AAA", output_b="BBB", order_swap=False
    )
    assert len(calls) == 1
    assert res.final_winner == "Model_A"
    assert res.position_consistent is None, "unknowable without the second order"
    assert res.order_swapped is False
    assert res.reverse_winner is None


def test_order_swap_off_still_reports_an_incomplete_pair_on_failure():
    fn, _ = _fn(None, None)
    res = evaluate_pairwise_unbiased(fn, case_id="c7", output_a="A", output_b="B", order_swap=False)
    assert res.incomplete is True
    assert res.final_winner is None


def test_lengths_are_recorded_for_the_verbosity_correlation():
    fn, _ = _fn("Tie", "Tie")
    res = evaluate_pairwise_unbiased(fn, case_id="c8", output_a="abc", output_b="abcdef")
    assert (res.length_a, res.length_b) == (3, 6)


# --------------------------------------------------------------------------
# Probe construction helpers
# --------------------------------------------------------------------------

BASE = TestCase(case_id="b1", input="q", model_output="The answer is 42.", tags=["x"])


def test_padded_probe_preserves_the_substance_and_adds_only_filler():
    padded = build_padded_probe(BASE)
    assert BASE.model_output in padded.model_output
    assert len(padded.model_output) > 3 * len(BASE.model_output)
    assert padded.case_id == "b1-padded"
    assert "probe:padded" in padded.tags


def test_confidently_wrong_probe_requires_the_false_claim_to_be_supplied():
    probe = build_confidently_wrong_probe(BASE, false_claim="The answer is 43.")
    assert "The answer is 43." in probe.model_output
    assert "no real debate" in probe.model_output
    assert "probe:confidently_wrong" in probe.tags
