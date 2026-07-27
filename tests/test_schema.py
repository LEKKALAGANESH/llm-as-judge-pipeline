"""Schema tests. The field-order assertions are the most important tests in this
repository: they are the only thing standing between the project and a silently
voided sycophancy mitigation."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.judge import criterion_object_schema, pairwise_wire_schema, pointwise_wire_schema
from src.schema import (
    CriterionScore,
    PairwiseVerdict,
    PointwiseVerdict,
    TestCase,
    VerdictAdapter,
    compute_passed,
)
from tests.fixtures import malformed_responses as mr

# --------------------------------------------------------------------------
# Field order IS the mitigation
# --------------------------------------------------------------------------


def test_criterion_score_field_order_is_rationale_before_score():
    assert list(CriterionScore.model_fields) == ["rationale", "score"]


def test_pointwise_verdict_puts_overall_rationale_before_overall_score():
    fields = list(PointwiseVerdict.model_fields)
    assert fields.index("overall_rationale") < fields.index("overall_score")


def test_pairwise_verdict_puts_rationale_before_winner():
    assert list(PairwiseVerdict.model_fields) == ["rationale", "winner"]


def test_serialised_json_preserves_rationale_before_score():
    """Order must survive serialisation, not just declaration -- what the model
    sees is the rendered schema, and what we store is the dumped JSON."""
    verdict = CriterionScore(rationale="cites 'signed 1919'", score=4)
    dumped = verdict.model_dump_json()
    assert dumped.index('"rationale"') < dumped.index('"score"')


def test_wire_schema_ordering_is_derived_from_the_models_not_hand_written():
    crit = criterion_object_schema()
    assert list(crit["properties"]) == list(CriterionScore.model_fields)
    assert crit["required"] == list(CriterionScore.model_fields)
    assert crit["propertyOrdering"] == list(CriterionScore.model_fields)

    pointwise = pointwise_wire_schema(["correctness", "safety"])
    assert list(pointwise["properties"]) == list(PointwiseVerdict.model_fields)
    assert pointwise["required"] == list(PointwiseVerdict.model_fields)

    pairwise = pairwise_wire_schema()
    assert list(pairwise["properties"]) == list(PairwiseVerdict.model_fields)


def test_wire_schema_lists_every_requested_criterion_and_nothing_else():
    schema = pointwise_wire_schema(mr.CRITERIA)
    breakdown = schema["properties"]["criteria_breakdown"]
    assert list(breakdown["properties"]) == mr.CRITERIA
    assert breakdown["required"] == mr.CRITERIA
    assert breakdown["additionalProperties"] is False


# --------------------------------------------------------------------------
# Round-trip and rejection
# --------------------------------------------------------------------------


def test_pointwise_round_trip():
    original = PointwiseVerdict.model_validate_json(mr.valid_pointwise(overall=3))
    again = PointwiseVerdict.model_validate_json(original.model_dump_json())
    assert again == original
    assert again.overall_score == 3


def test_pairwise_round_trip():
    original = PairwiseVerdict.model_validate_json(mr.valid_pairwise("Model_A"))
    again = PairwiseVerdict.model_validate_json(original.model_dump_json())
    assert again == original


def test_verdict_union_round_trips_through_the_adapter():
    from src.schema import PairwiseCaseResult, PointwiseCaseResult

    for verdict in (
        PointwiseCaseResult(case_id="a", output_length=10),
        PairwiseCaseResult(case_id="b", final_winner="Tie"),
    ):
        blob = VerdictAdapter.dump_json(verdict)
        assert VerdictAdapter.validate_json(blob) == verdict


@pytest.mark.parametrize(
    "payload",
    [mr.MISSING_OVERALL_SCORE, mr.WRONG_TYPE, mr.SCORE_OUT_OF_RANGE, mr.EMPTY_RATIONALE],
)
def test_invalid_payloads_are_rejected(payload):
    with pytest.raises(ValidationError):
        PointwiseVerdict.model_validate_json(payload)


def test_missing_criterion_is_rejected_only_when_the_required_set_is_supplied():
    # Without context this is structurally valid: five criteria is a fine dict.
    PointwiseVerdict.model_validate_json(mr.MISSING_CRITERION)
    with pytest.raises(ValidationError) as exc:
        PointwiseVerdict.model_validate_json(
            mr.MISSING_CRITERION, context={"required_criteria": mr.CRITERIA}
        )
    assert "missing criteria" in str(exc.value)


def test_extra_criterion_is_rejected_with_the_required_set():
    with pytest.raises(ValidationError) as exc:
        PointwiseVerdict.model_validate_json(
            mr.EXTRA_CRITERION, context={"required_criteria": mr.CRITERIA}
        )
    assert "unexpected criteria" in str(exc.value)


def test_bad_pairwise_winner_is_rejected():
    with pytest.raises(ValidationError):
        PairwiseVerdict.model_validate_json(mr.PAIRWISE_BAD_WINNER)


def test_test_case_forbids_unknown_fields():
    with pytest.raises(ValidationError):
        TestCase(case_id="x", input="i", model_output="o", typo_field="oops")


def test_duplicate_case_ids_are_rejected():
    from src.schema import TestSuite

    with pytest.raises(ValidationError) as exc:
        TestSuite(
            suite_id="s",
            cases=[
                TestCase(case_id="dup", input="a", model_output="a"),
                TestCase(case_id="dup", input="b", model_output="b"),
            ],
        )
    assert "duplicate case_id" in str(exc.value)


# --------------------------------------------------------------------------
# passed is computed, never emitted
# --------------------------------------------------------------------------


def test_verdict_schemas_have_no_passed_field():
    assert "passed" not in PointwiseVerdict.model_fields
    assert "passed" not in CriterionScore.model_fields
    assert "passed" not in PairwiseVerdict.model_fields
    assert "passed" not in str(pointwise_wire_schema(mr.CRITERIA))


@pytest.mark.parametrize(
    "overall,worst,expected",
    [
        (5, 5, True),
        (4, 3, True),
        (4, 2, False),  # a single weak criterion fails the case
        (3, 5, False),  # overall below the bar fails the case
        (5, 3, True),
    ],
)
def test_compute_passed_requires_both_conditions(overall, worst, expected):
    scores = dict.fromkeys(mr.CRITERIA, 5)
    scores[mr.CRITERIA[0]] = worst
    verdict = PointwiseVerdict.model_validate_json(
        mr.valid_pointwise(scores=scores, overall=overall)
    )
    assert compute_passed(verdict, pass_threshold=4, min_criterion_score=3) is expected


def test_mean_criterion_score_is_a_float_and_overall_is_an_int():
    verdict = PointwiseVerdict.model_validate_json(
        mr.valid_pointwise(scores={**dict.fromkeys(mr.CRITERIA, 4), "safety": 1}, overall=4)
    )
    assert isinstance(verdict.overall_score, int)
    assert verdict.min_criterion_score == 1
    assert verdict.mean_criterion_score == pytest.approx((4 * 5 + 1) / 6)
