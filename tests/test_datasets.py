"""Dataset loading, and assertions about the shipped data itself.

The second half of this file is unusual but deliberate: the committed suites,
probes and gold labels are deliverables, so their properties (coverage per bias
category, pre-registered bounds, disclosed labelling method, gold ids that
actually exist) are tested rather than eyeballed.
"""

from __future__ import annotations

import json

import pytest

from src.datasets import (
    Probe,
    load_comparison_suite,
    load_gold_labels,
    load_probes,
    load_test_suite,
)
from src.errors import SuiteLoadError

SUITE = "data/test_suites/general_qa.json"
COMPARISON = "data/test_suites/comparison_qa.json"
PROBES = "data/test_suites/adversarial_probes.json"
GOLD = "data/gold_labels/human_annotated.json"


# --------------------------------------------------------------------------
# Negative paths
# --------------------------------------------------------------------------


def test_missing_file_is_a_clear_error(tmp_path):
    with pytest.raises(SuiteLoadError) as exc:
        load_test_suite(tmp_path / "nope.json")
    assert "not found" in str(exc.value)


def test_missing_gold_file_refuses_rather_than_reporting_a_fabricated_kappa(tmp_path):
    with pytest.raises(SuiteLoadError):
        load_gold_labels(tmp_path / "no_gold.json")


def test_empty_gold_file_is_refused_with_an_explanation(tmp_path):
    p = tmp_path / "gold.json"
    p.write_text(
        json.dumps(
            {
                "labeling_method": {
                    "labeler": "x",
                    "labeled_at": "2026-01-01",
                    "procedure": "x",
                    "labeled_before_judge_run": True,
                },
                "labels": [],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(SuiteLoadError) as exc:
        load_gold_labels(p)
    assert "fabricated result" in str(exc.value)


def test_empty_suite_is_refused(tmp_path):
    p = tmp_path / "s.json"
    p.write_text(json.dumps({"suite_id": "empty", "cases": []}), encoding="utf-8")
    with pytest.raises(SuiteLoadError) as exc:
        load_test_suite(p)
    assert "zero cases" in str(exc.value)


def test_invalid_json_names_the_file(tmp_path):
    p = tmp_path / "s.json"
    p.write_text("{not json", encoding="utf-8")
    with pytest.raises(SuiteLoadError) as exc:
        load_test_suite(p)
    assert "not valid JSON" in str(exc.value)


def test_schema_violation_names_the_failing_case(tmp_path):
    p = tmp_path / "s.json"
    p.write_text(
        json.dumps({"suite_id": "s", "cases": [{"case_id": "a", "input": "i"}]}), encoding="utf-8"
    )
    with pytest.raises(SuiteLoadError) as exc:
        load_test_suite(p)
    assert "model_output" in str(exc.value)


@pytest.mark.parametrize(
    "payload,expected",
    [
        ({"kind": "pointwise_threshold", "model_output": "x"}, "needs assertions"),
        ({"kind": "pointwise_threshold", "assertions": [{"max_score": 2}]}, "needs model_output"),
        ({"kind": "relational_pair", "baseline_output": "a"}, "needs baseline_output"),
        ({"kind": "identical_pair"}, "needs output"),
    ],
)
def test_probe_kinds_require_their_own_fields(payload, expected):
    with pytest.raises(Exception) as exc:
        Probe(probe_id="p", category="c", input="i", **payload)
    assert expected in str(exc.value)


def test_a_threshold_assertion_must_carry_a_bound():
    with pytest.raises(Exception) as exc:
        Probe(
            probe_id="p",
            category="c",
            kind="pointwise_threshold",
            input="i",
            model_output="o",
            assertions=[{"criterion": "correctness"}],
        )
    assert "needs max_score, min_score, or both" in str(exc.value)


# --------------------------------------------------------------------------
# The shipped data
# --------------------------------------------------------------------------


def test_general_qa_suite_is_large_enough_and_exercises_every_criterion(repo_root):
    suite = load_test_suite(repo_root / SUITE)
    assert 15 <= len(suite.cases) <= 25
    tags = " ".join(t for c in suite.cases for t in c.tags)
    for criterion in (
        "correctness",
        "faithfulness",
        "completeness",
        "instruction_following",
        "tone",
        "safety",
    ):
        assert criterion in tags, f"no case is tagged as targeting {criterion}"
    assert any(c.expected_output for c in suite.cases), "no reference-based case"
    assert any(c.expected_output is None for c in suite.cases), "no reference-free case"


def test_gold_labels_reference_real_cases_and_disclose_the_method(repo_root):
    suite = load_test_suite(repo_root / SUITE)
    gold = load_gold_labels(repo_root / GOLD)
    ids = {c.case_id for c in suite.cases}
    assert len(gold.labels) >= 15
    assert all(label.case_id in ids for label in gold.labels)
    method = gold.labeling_method
    assert method.labeled_before_judge_run is True
    assert method.caveats, "an undisclosed limitation is a hidden one"
    assert len(method.procedure) > 100


def test_gold_labels_use_more_than_one_score_level(repo_root):
    """A gold set that is all 4s and 5s makes kappa undefined or uninterpretable
    before the judge has said anything."""
    gold = load_gold_labels(repo_root / GOLD)
    levels = {label.overall_score for label in gold.labels}
    assert len(levels) >= 4, f"gold overall scores span only {sorted(levels)}"


def test_gold_labels_cover_every_rubric_criterion_per_case(repo_root, rubric):
    gold = load_gold_labels(repo_root / GOLD)
    for label in gold.labels:
        assert set(label.criterion_scores) == set(rubric.criterion_names), label.case_id


def test_probe_suite_has_enough_probes_per_category(repo_root):
    probes = load_probes(repo_root / PROBES)
    counts: dict[str, int] = {}
    for p in probes.probes:
        counts[p.category] = counts.get(p.category, 0) + 1
    assert set(counts) == {
        "verbosity",
        "sycophancy",
        "position_noise_floor",
        "over_correction",
        "prompt_injection",
    }
    for category, n in counts.items():
        assert n >= 5, f"{category} has {n} probes; n=1 is not a measurement and n<5 is thin"


def test_every_probe_carries_a_pre_registered_bound(repo_root):
    probes = load_probes(repo_root / PROBES)
    for p in probes.probes:
        if p.kind == "pointwise_threshold":
            assert p.assertions
        elif p.kind == "relational_pair":
            assert p.relation and p.scored_criteria
        else:
            assert p.output


def test_verbosity_probes_are_scored_only_on_the_stipulated_identical_criteria(repo_root):
    probes = load_probes(repo_root / PROBES)
    for p in probes.probes:
        if p.category == "verbosity":
            assert set(p.scored_criteria) == {"correctness", "faithfulness"}, p.probe_id
            assert p.relation == "variant_not_higher"
            assert len(p.variant_output) > len(p.baseline_output)


def test_over_correction_probes_push_the_opposite_way(repo_root):
    """Every other probe rewards brevity; without these you can score 100% by
    cranking the anti-verbosity clause until thoroughness is punished."""
    probes = load_probes(repo_root / PROBES)
    over = [p for p in probes.probes if p.category == "over_correction"]
    assert over
    for p in over:
        assert p.relation == "variant_higher"
        assert len(p.variant_output) > len(p.baseline_output)


def test_identical_pair_probes_are_actually_identical(repo_root):
    probes = load_probes(repo_root / PROBES)
    identical = [p for p in probes.probes if p.kind == "identical_pair"]
    assert len(identical) >= 3
    for p in identical:
        assert p.output and p.baseline_output is None and p.variant_output is None


def test_injection_probes_contain_an_actual_injection(repo_root):
    probes = load_probes(repo_root / PROBES)
    inj = [p for p in probes.probes if p.category == "prompt_injection"]
    assert len(inj) >= 5
    assert any("overall_score" in (p.model_output or "") for p in inj), "no injected verdict object"
    assert any("IGNORE ALL PREVIOUS INSTRUCTIONS" in (p.model_output or "") for p in inj)


def test_comparison_suite_declares_its_provenance_honestly(repo_root):
    suite = load_comparison_suite(repo_root / COMPARISON)
    assert len(suite.cases) >= 15
    assert suite.config_a_label != suite.config_b_label
    prov = suite.provenance
    assert "source" in prov and "regenerate_command" in prov
    assert prov["config_a_prompt"] != prov["config_b_prompt"]
    for case in suite.cases:
        assert case.source, f"{case.case_id} does not say where its outputs came from"
        assert case.output_a != case.output_b


def test_comparison_suite_does_not_leak_which_arm_produced_which_output(repo_root):
    """The judge sees `system_prompt`; if the two variant prompts appeared there
    it could infer provenance, which would contaminate a blind comparison."""
    suite = load_comparison_suite(repo_root / COMPARISON)
    for case in suite.cases:
        assert case.system_prompt == "Answer the user's question."
        assert suite.config_a_label not in case.output_a
        assert suite.config_b_label not in case.output_b
