"""Live-API tests. Deselected in CI (`pytest -m "not live"`), run manually.

These exist because the two things most likely to be silently wrong on this stack
cannot be caught by a mock: whether the model ids still resolve (both providers
deprecated models within the last quarter, and litellm's model table is stale for
both), and whether the provider actually honours the structured-output request.

    pytest -m live -q

Costs roughly 4 judge calls. Requires GEMINI_API_KEY and GROQ_API_KEY.
"""

from __future__ import annotations

import os

import pytest

from src.judge import JudgePromptBuilder, judge_pointwise
from src.llm_client import LLMClient
from src.schema import TestCase
from src.suite_config import MitigationSettings

pytestmark = pytest.mark.live

CASE = TestCase(
    case_id="live-001",
    input="When was the Treaty of Versailles signed?",
    system_prompt="Be concise.",
    model_output="The Treaty of Versailles was signed on 28 June 1919.",
    expected_output="28 June 1919.",
)


def _skip_without(env: str) -> None:
    if not os.environ.get(env):
        pytest.skip(f"{env} is not set")


def test_judge_model_id_still_resolves_and_returns_usage(config):
    _skip_without("GEMINI_API_KEY")
    client = LLMClient(cache_path=None, enable_cache=False, max_calls=2)
    resp = client.complete(
        model=config.judge.model,
        messages=[{"role": "user", "content": "Reply with the single word: ready"}],
        temperature=0.0,
        max_tokens=16,
    )
    assert resp.text.strip()
    assert resp.prompt_tokens > 0, "usage must be read off the response, not estimated"
    assert resp.latency_ms > 0


def test_candidate_model_id_still_resolves(config):
    _skip_without("GROQ_API_KEY")
    probe = config.self_enhancement_probe_judge
    client = LLMClient(cache_path=None, enable_cache=False, max_calls=2)
    resp = client.complete(
        model=probe.model,
        messages=[{"role": "user", "content": "Reply with the single word: ready"}],
        temperature=0.0,
        max_tokens=16,
    )
    assert resp.text.strip()


def test_real_judge_returns_a_parseable_verdict_in_schema_order(config, rubric):
    _skip_without("GEMINI_API_KEY")
    client = LLMClient(cache_path=None, enable_cache=False, max_calls=4)
    outcome = judge_pointwise(
        case=CASE,
        rubric=rubric,
        builder=JudgePromptBuilder(rubric, MitigationSettings()),
        judge=config.judge,
        client=client,
    )
    assert outcome.verdict is not None, (
        f"judge failed: {outcome.error}\nraw: {outcome.raw_response}"
    )
    assert set(outcome.verdict.criteria_breakdown) == set(rubric.criterion_names)
    raw = outcome.raw_response or ""
    assert raw.index('"rationale"') < raw.index('"score"'), (
        "the model emitted score before rationale: constrained decoding is not honouring the "
        "schema order, which voids the grounding mitigation"
    )
