"""Config, family derivation and the cross-family invariant.

The negative tests here are the point: a mitigation enforced by an assertion that
is never tested is a mitigation enforced by hope.
"""

from __future__ import annotations

import pytest
import yaml

from src.errors import ConfigError, CrossFamilyViolation
from src.suite_config import (
    UNKNOWN_FAMILY,
    CandidateSpec,
    MitigationSettings,
    ModelSettings,
    PipelineConfig,
    derive_family,
    enforce_cross_family,
    load_config,
    load_rubric,
    model_portion,
    routing_provider,
    warn_if_deterministic_test_retest,
)

# --------------------------------------------------------------------------
# family derivation: model portion, never the routing prefix
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "model_id,expected",
    [
        ("groq/openai/gpt-oss-120b", "openai"),
        ("groq/openai/gpt-oss-20b", "openai"),
        ("gemini/gemini-3.5-flash", "google"),
        ("vertex_ai/gemini-3.5-flash-lite", "google"),
        ("bedrock/anthropic.claude-3-5-sonnet", "anthropic"),
        ("anthropic/claude-opus-4", "anthropic"),
        ("gpt-4o", "openai"),
        ("groq/meta-llama/llama-3.3-70b", "meta"),
        ("together_ai/mistralai/mixtral-8x7b", "mistral"),
        ("something/entirely-unheard-of-v9", UNKNOWN_FAMILY),
    ],
)
def test_family_is_derived_from_the_model_portion(model_id, expected):
    assert derive_family(model_id) == expected


def test_the_trap_case_a_groq_served_gemini_is_still_google():
    """A naive provider check would compare 'gemini' to 'groq', pass for the
    wrong reason, and keep passing after someone routed the judge through Groq."""
    assert routing_provider("groq/gemini-3.5-flash") == "groq"
    assert derive_family("groq/gemini-3.5-flash") == "google"
    assert model_portion("groq/openai/gpt-oss-120b") == "openai/gpt-oss-120b"


def _config(judge: str, candidates: list[str], generator_family: str | None = None):
    return PipelineConfig(
        judge=ModelSettings(model=judge),
        candidates=[CandidateSpec(label=f"c{i}", model=m) for i, m in enumerate(candidates)],
        generator_family=generator_family,
    )


def test_cross_family_passes_for_the_shipped_pairing():
    cfg = enforce_cross_family(
        _config("gemini/gemini-3.5-flash", ["groq/openai/gpt-oss-120b", "groq/openai/gpt-oss-20b"])
    )
    assert cfg.resolved_families["judge"] == "google"
    assert cfg.resolved_families["candidate:c0"] == "openai"
    assert cfg.cross_family_enforced is True


def test_cross_family_refuses_a_same_family_pairing():
    with pytest.raises(CrossFamilyViolation) as exc:
        enforce_cross_family(_config("groq/openai/gpt-oss-120b", ["groq/openai/gpt-oss-20b"]))
    msg = str(exc.value)
    assert "openai" in msg and "Refusing to run" in msg


def test_cross_family_catches_the_routing_prefix_trap():
    """Judge and candidate are routed by different providers but share a family."""
    with pytest.raises(CrossFamilyViolation):
        enforce_cross_family(_config("groq/gemini-3.5-flash", ["vertex_ai/gemini-3.5-flash-lite"]))


def test_empty_candidate_list_is_refused_rather_than_vacuously_passing():
    with pytest.raises(CrossFamilyViolation) as exc:
        enforce_cross_family(_config("gemini/gemini-3.5-flash", []))
    assert "vacuously true" in str(exc.value)


def test_explicit_generator_family_satisfies_the_non_empty_requirement():
    cfg = enforce_cross_family(_config("gemini/gemini-3.5-flash", [], generator_family="openai"))
    assert cfg.resolved_families["generator_family"] == "openai"


def test_explicit_generator_family_can_also_violate_the_invariant():
    with pytest.raises(CrossFamilyViolation):
        enforce_cross_family(_config("gemini/gemini-3.5-flash", [], generator_family="google"))


def test_unknown_family_is_a_failure_not_a_pass():
    with pytest.raises(CrossFamilyViolation) as exc:
        enforce_cross_family(_config("gemini/gemini-3.5-flash", ["mystery/unknown-model-x"]))
    assert "Cannot resolve the model family" in str(exc.value)


def test_allow_same_family_waives_the_check_and_records_the_waiver():
    cfg = enforce_cross_family(
        _config("groq/openai/gpt-oss-120b", ["groq/openai/gpt-oss-20b"]), allow_same_family=True
    )
    assert cfg.cross_family_enforced is False


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------


def test_shipped_config_and_rubric_load_and_satisfy_the_invariant(repo_root):
    cfg = load_config(repo_root / "config" / "suite_config.yaml")
    # Assert the invariant, not the vendor: which family judges is a free-tier
    # quota decision that has already changed once (Gemini -> Groq-hosted
    # llama). What must never change is that it differs from the generators'.
    assert cfg.generator_family == "openai"
    assert cfg.judge.family != cfg.generator_family
    assert cfg.cross_family_enforced is True

    rubric = load_rubric(repo_root / "config" / "rubric.yaml")
    assert rubric.criterion_names == [
        "correctness",
        "faithfulness",
        "completeness",
        "instruction_following",
        "tone",
        "safety",
    ]
    for criterion in rubric.criteria:
        assert set(criterion.anchors) == {"1", "3", "5"}, criterion.name
        assert criterion.rationale, f"{criterion.name} has no documented rationale"
    assert rubric.clauses.anti_verbosity and rubric.clauses.grounding
    assert rubric.clauses.injection_defense


def test_loader_uses_safe_load_and_will_not_construct_python_objects(tmp_path):
    """`yaml.load` with an unsafe loader would happily build an arbitrary object
    here. safe_load raises instead, which is the behaviour we want on a file a
    user was invited to receive from someone else."""
    bad = tmp_path / "evil.yaml"
    bad.write_text("judge: !!python/object/apply:os.system ['echo pwned']\n", encoding="utf-8")
    with pytest.raises(yaml.YAMLError):
        load_config(bad)


def test_missing_config_file_is_a_clear_error(tmp_path):
    with pytest.raises(ConfigError) as exc:
        load_config(tmp_path / "nope.yaml")
    assert "not found" in str(exc.value)


def test_invalid_config_names_the_failing_field(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        "judge:\n  model: gemini/gemini-3.5-flash\n  temperature: 99\n", encoding="utf-8"
    )
    with pytest.raises(ConfigError) as exc:
        load_config(bad)
    assert "temperature" in str(exc.value)


def test_rubric_subset_selects_and_validates(repo_root):
    rubric = load_rubric(repo_root / "config" / "rubric.yaml")
    subset = rubric.subset(["correctness", "safety"])
    assert subset.criterion_names == ["correctness", "safety"]
    assert rubric.subset(None) is rubric
    with pytest.raises(ConfigError):
        rubric.subset(["not_a_criterion"])


# --------------------------------------------------------------------------
# T=0 guard and mitigation switches
# --------------------------------------------------------------------------


def test_test_retest_at_zero_temperature_is_refused_unless_acknowledged():
    with pytest.raises(ConfigError) as exc:
        warn_if_deterministic_test_retest(0.0, acknowledged=False)
    assert "not a validation result" in str(exc.value)

    msg = warn_if_deterministic_test_retest(0.0, acknowledged=True)
    assert msg is not None and msg.startswith("ACKNOWLEDGED")
    assert warn_if_deterministic_test_retest(0.7, acknowledged=False) is None


def test_mitigations_all_off_leaves_rationale_first_alone():
    off = MitigationSettings.all_off()
    assert off.order_swap is False
    assert off.anti_verbosity_clause is False
    assert off.few_shot_anchors is False
    # Not flag-ablatable: it is the schema's field order.
    assert off.rationale_first is True
