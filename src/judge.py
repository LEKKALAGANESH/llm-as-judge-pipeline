"""The only module that calls the judge LLM.

Three things in here are load-bearing beyond "build a prompt, call a model":

**The wire schema is generated from the Pydantic models.** ``criterion_object_schema``
reads ``list(CriterionScore.model_fields)`` rather than hard-coding key names, so
the schema advertised to the provider, the schema pasted into the prompt, and the
schema used to parse the reply cannot drift apart -- and the rationale-before-score
ordering propagates to all three from one place.

**The repair loop is not a retry decorator.** ``tenacity`` re-invokes with
identical arguments; re-sending a prompt that just produced malformed JSON is a
re-issue, not a repair. The loop here appends the malformed output *and* the
validation error to the message list before asking again, which is what makes the
second request materially different from the first. Transient-error retries live
one layer down in ``llm_client``.

**Every attempt is logged before the function returns**, including attempts that
failed to parse and attempts that failed to complete, and tokens are summed
across all of them. Logging after a successful parse would leave exactly the
failures a reviewer wants to audit with no trace.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import ValidationError

from .audit_log import AuditLogger, redact_env_values
from .errors import FatalProviderError, QuotaExceeded
from .llm_client import TRANSIENT_ERRORS, LLMClient
from .schema import (
    CallAccounting,
    CriterionScore,
    FailureKind,
    PairwiseVerdict,
    PointwiseVerdict,
    TestCase,
)
from .suite_config import (
    MitigationSettings,
    ModelSettings,
    Rubric,
    routing_provider,
)

JudgeMode = Literal["pointwise", "pairwise"]


# --------------------------------------------------------------------------
# Wire schema (generated from the Pydantic models -- never hand-written)
# --------------------------------------------------------------------------

_CRITERION_FIELD_DOCS = {
    "rationale": (
        "Evidence for this criterion. Quote the exact span of the judged output that "
        "supports your judgment, or name precisely what is missing. Write this BEFORE "
        "you decide the score."
    ),
    "score": "Integer 1-5 for this criterion, following from the rationale you just wrote.",
}


def criterion_object_schema() -> dict[str, Any]:
    """One criterion's wire schema, in the model's declared field order."""
    order = list(CriterionScore.model_fields)
    props: dict[str, Any] = {}
    for name in order:
        if name == "score":
            props[name] = {
                "type": "integer",
                "minimum": 1,
                "maximum": 5,
                "description": _CRITERION_FIELD_DOCS[name],
            }
        else:
            props[name] = {"type": "string", "description": _CRITERION_FIELD_DOCS[name]}
    return {
        "type": "object",
        "properties": props,
        "required": order,
        # Gemini honours propertyOrdering; it is a no-op elsewhere. Ordering is
        # the sycophancy mitigation, so it is asserted in three places.
        "propertyOrdering": order,
        "additionalProperties": False,
    }


def pointwise_wire_schema(criteria: list[str]) -> dict[str, Any]:
    order = list(PointwiseVerdict.model_fields)
    crit = criterion_object_schema()
    props: dict[str, Any] = {
        "criteria_breakdown": {
            "type": "object",
            "properties": {name: crit for name in criteria},
            "required": list(criteria),
            "propertyOrdering": list(criteria),
            "additionalProperties": False,
        },
        "overall_rationale": {
            "type": "string",
            "description": (
                "Two or three sentences synthesising the per-criterion evidence above. "
                "Written BEFORE the overall score."
            ),
        },
        "overall_score": {
            "type": "integer",
            "minimum": 1,
            "maximum": 5,
            "description": "Integer 1-5 holistic score, consistent with the per-criterion scores.",
        },
    }
    return {
        "type": "object",
        "properties": {name: props[name] for name in order},
        "required": order,
        "propertyOrdering": order,
        "additionalProperties": False,
    }


def pairwise_wire_schema() -> dict[str, Any]:
    order = list(PairwiseVerdict.model_fields)
    props: dict[str, Any] = {
        "rationale": {
            "type": "string",
            "description": (
                "Criterion-by-criterion comparison citing specific spans from each output. "
                "Written BEFORE the winner is named."
            ),
        },
        "winner": {
            "type": "string",
            "enum": ["Model_A", "Model_B", "Tie"],
            "description": "The better output, or Tie when they are of equivalent quality.",
        },
    }
    return {
        "type": "object",
        "properties": {name: props[name] for name in order},
        "required": order,
        "propertyOrdering": order,
        "additionalProperties": False,
    }


def response_format_for(model: str, name: str, schema: dict[str, Any]) -> dict[str, Any]:
    """Groq's `strict: true` is genuine constrained decoding with guaranteed
    conformance; Gemini's responseSchema docs stop short of promising that,
    which is exactly why the repair loop below is a real defense here and not a
    vestigial one."""
    strict = routing_provider(model) == "groq"
    return {
        "type": "json_schema",
        "json_schema": {"name": name, "schema": schema, "strict": strict},
    }


# --------------------------------------------------------------------------
# Prompt construction
# --------------------------------------------------------------------------


NONCE_SALT = os.environ.get("JUDGE_NONCE_SALT", "llm-as-judge-pipeline/delimiter/v1")


def content_nonce(*parts: str) -> str:
    """Per-request delimiter nonce, derived from a salt and the request content.

    Deriving it (rather than drawing a fresh random value) is deliberate. A
    random nonce makes every prompt unique, which silently defeats the disk
    cache -- on a free tier that is the difference between iterating freely and
    burning a day's quota on debugging. Hashing content plus a salt keeps the
    nonce unguessable-in-practice from the content side (an attacker would need
    a preimage, and the salt is overridable per deployment via JUDGE_NONCE_SALT)
    while keeping identical requests cacheable.

    The nonce is defence in depth in any case: the load-bearing step is stripping
    it from the content before wrapping, which is what makes early termination of
    the block impossible even if the delimiter leaked.
    """
    h = hashlib.sha256(NONCE_SALT.encode("utf-8"))
    for part in parts:
        h.update(b"\x00")
        h.update((part or "").encode("utf-8"))
    return h.hexdigest()[:16]


def wrap_untrusted(content: str, nonce: str, label: str) -> str:
    """Delimit judged content as data, stripping the nonce from it first.

    Without the strip, a candidate output containing the delimiter could close
    the block early and have the remainder of itself read as instructions.
    """
    cleaned = (content or "").replace(nonce, "")
    return f"<<<BEGIN {label} {nonce}>>>\n{cleaned}\n<<<END {label} {nonce}>>>"


class JudgePromptBuilder:
    """Renders the rubric into a system prompt once per (criteria, mode).

    The rubric text is identical across every call in a run, so it is built once
    and reused -- a trivial win that also keeps the stable, cacheable content at
    the front of the prompt where provider-side prompt caching wants it.
    """

    def __init__(self, rubric: Rubric, mitigations: MitigationSettings) -> None:
        self.rubric = rubric
        self.mitigations = mitigations
        self._cache: dict[tuple[str, tuple[str, ...]], str] = {}

    # -- shared pieces -----------------------------------------------------

    def _criteria_block(self, criteria: list[str]) -> str:
        by_name = {c.name: c for c in self.rubric.criteria}
        chunks: list[str] = []
        for name in criteria:
            crit = by_name[name]
            block = [f"### {crit.name}", crit.description.strip()]
            if self.mitigations.few_shot_anchors and crit.anchors:
                block.append("Calibration anchors (use the whole 1-5 scale):")
                for level in sorted(crit.anchors, key=lambda k: int(k)):
                    block.append(f"  score {level}: {crit.anchors[level].strip()}")
            chunks.append("\n".join(block))
        return "\n\n".join(chunks)

    def _clauses_block(self) -> str:
        parts = [self.rubric.clauses.grounding.strip()]
        if self.mitigations.anti_verbosity_clause and self.rubric.clauses.anti_verbosity:
            parts.append(self.rubric.clauses.anti_verbosity.strip())
        if self.rubric.clauses.injection_defense:
            parts.append(self.rubric.clauses.injection_defense.strip())
        return "\n\n".join(p for p in parts if p)

    # -- system prompts ----------------------------------------------------

    def pointwise_system(self, criteria: list[str]) -> str:
        key = ("pointwise", tuple(criteria))
        if key in self._cache:
            return self._cache[key]
        schema = pointwise_wire_schema(criteria)
        text = f"""You are an evaluation judge. You score one model output against a fixed rubric.

## Scale
{self.rubric.scale.description.strip()}

## Rubric (version {self.rubric.version})
Score every one of these {len(criteria)} criteria. Do not add criteria, do not omit any.

{self._criteria_block(criteria)}

## How to judge
{self._clauses_block()}

Do NOT output a pass/fail decision. Pass/fail is computed downstream from your
scores against thresholds you are not given, and a self-declared verdict would
be an uncalibrated extra judgment.

## Output format
Reply with a single JSON object and nothing else -- no prose, no markdown fence.
It must conform exactly to this JSON Schema, including the order of the fields:

{json.dumps(schema, indent=2)}
"""
        self._cache[key] = text
        return text

    def pairwise_system(self) -> str:
        key = ("pairwise", ())
        if key in self._cache:
            return self._cache[key]
        criteria = self.rubric.criterion_names
        schema = pairwise_wire_schema()
        text = f"""You are an evaluation judge. Two model outputs answer the same request. \
Decide which is better, or declare a tie.

## Criteria (version {self.rubric.version})
Weigh these, in this order of importance, and say so in your rationale:

{self._criteria_block(criteria)}

## How to judge
{self._clauses_block()}

The labels "Model_A" and "Model_B" are arbitrary presentation order and carry no
information about quality, provenance, or which output was produced first. Judge
the content only. If the two outputs are of equivalent quality -- including when
they are identical or near-identical -- return "Tie". Do not break a genuine tie
to appear decisive.

## Output format
Reply with a single JSON object and nothing else -- no prose, no markdown fence.
It must conform exactly to this JSON Schema, including the order of the fields:

{json.dumps(schema, indent=2)}
"""
        self._cache[key] = text
        return text

    # -- user prompts ------------------------------------------------------

    def pointwise_user(self, case: TestCase, nonce: str) -> str:
        parts = [
            "Evaluate the OUTPUT below against the rubric.",
            "",
            "Original request given to the model:",
            wrap_untrusted(case.input, nonce, "REQUEST"),
        ]
        if case.system_prompt:
            parts += [
                "",
                "System prompt the model was operating under:",
                wrap_untrusted(case.system_prompt, nonce, "SYSTEM_PROMPT_UNDER_TEST"),
            ]
        if case.expected_output is not None:
            parts += [
                "",
                "Reference answer (this case is reference-based; judge correctness and "
                "faithfulness against this reference, but do not require identical wording -- "
                "a different valid phrasing is not an error):",
                wrap_untrusted(case.expected_output, nonce, "REFERENCE"),
            ]
        else:
            parts += [
                "",
                "No reference answer exists for this case (reference-free). Judge against the "
                "request, the system prompt and your own knowledge.",
            ]
        parts += [
            "",
            "Output to judge:",
            wrap_untrusted(case.model_output, nonce, "OUTPUT"),
            "",
            "Return the JSON verdict now.",
        ]
        return "\n".join(parts)

    def pairwise_user(
        self,
        case_input: str,
        system_prompt: str,
        output_a: str,
        output_b: str,
        nonce: str,
        expected_output: str | None = None,
    ) -> str:
        parts = [
            "Compare the two outputs below.",
            "",
            "Original request given to both models:",
            wrap_untrusted(case_input, nonce, "REQUEST"),
        ]
        if system_prompt:
            parts += [
                "",
                "System prompt both models were operating under:",
                wrap_untrusted(system_prompt, nonce, "SYSTEM_PROMPT_UNDER_TEST"),
            ]
        if expected_output is not None:
            parts += [
                "",
                "Reference answer:",
                wrap_untrusted(expected_output, nonce, "REFERENCE"),
            ]
        parts += [
            "",
            "Model_A output:",
            wrap_untrusted(output_a, nonce, "OUTPUT_MODEL_A"),
            "",
            "Model_B output:",
            wrap_untrusted(output_b, nonce, "OUTPUT_MODEL_B"),
            "",
            "Return the JSON verdict now.",
        ]
        return "\n".join(parts)


# --------------------------------------------------------------------------
# Parsing
# --------------------------------------------------------------------------


def extract_json_object(text: str) -> str:
    """Recover the JSON object from a reply that may be wrapped in prose or a
    markdown fence. Raises ValueError when nothing object-shaped is present."""
    if text is None:
        raise ValueError("empty response: the model returned no content")
    stripped = text.strip()
    if not stripped:
        raise ValueError("empty response: the model returned no content")

    try:
        json.loads(stripped)
        return stripped
    except json.JSONDecodeError:
        pass

    fence = stripped
    if "```" in fence:
        segments = fence.split("```")
        for seg in segments[1:]:
            body = seg
            if body.lower().startswith("json"):
                body = body[4:]
            body = body.strip()
            if body.startswith("{"):
                try:
                    json.loads(body)
                    return body
                except json.JSONDecodeError:
                    continue

    start = stripped.find("{")
    if start == -1:
        raise ValueError("no JSON object found in the response (no '{' present)")
    depth = 0
    in_string = False
    escaped = False
    for i in range(start, len(stripped)):
        ch = stripped[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                candidate = stripped[start : i + 1]
                json.loads(candidate)  # raises JSONDecodeError -> caller repairs
                return candidate
    raise ValueError("JSON object is truncated: no matching closing brace")


def parse_pointwise(raw: str, required_criteria: list[str]) -> PointwiseVerdict:
    payload = extract_json_object(raw)
    return PointwiseVerdict.model_validate_json(
        payload, context={"required_criteria": required_criteria}
    )


def parse_pairwise(raw: str) -> PairwiseVerdict:
    payload = extract_json_object(raw)
    return PairwiseVerdict.model_validate_json(payload)


REPAIR_TEMPLATE = """Your previous reply could not be parsed into the required verdict schema.

The exact text you returned was:
---BEGIN YOUR PREVIOUS REPLY---
{malformed}
---END YOUR PREVIOUS REPLY---

The parser reported this error:
---BEGIN PARSER ERROR---
{error}
---END PARSER ERROR---

Fix precisely that problem and reply again with the corrected verdict. Rules:
- Output a single JSON object and nothing else. No prose, no markdown fence.
- Keep the field order given in the schema: every rationale comes before its score.
- Every required criterion must be present exactly once.
- Scores are integers from 1 to 5.
- Do not change your judgment to make the JSON easier; only fix the format.

Required schema:
{schema}
"""


TRUNCATION_REPAIR_TEMPLATE = """Your previous reply was cut off before it finished \
(the provider reported finish_reason="length"), so it could not be parsed.

This is a length problem, not a formatting one. Reply again with the same judgment, \
but keep every rationale to a single sentence quoting only the shortest span that \
supports the score. Output one JSON object and nothing else, with every required \
criterion present exactly once and each rationale before its score.

Do not soften or change any score to make the reply shorter."""

#: Ceiling for the doubling retry on truncation. Two doublings from the 2048
#: default is already an unusually long verdict; beyond that the likely cause is
#: a runaway generation rather than a genuinely long rationale, and continuing
#: to double just spends quota.
MAX_REPAIR_TOKEN_BUDGET = 8192


# --------------------------------------------------------------------------
# The call
# --------------------------------------------------------------------------


@dataclass
class JudgeOutcome:
    verdict: PointwiseVerdict | PairwiseVerdict | None = None
    raw_response: str | None = None
    accounting: CallAccounting = field(default_factory=CallAccounting)
    failure: FailureKind | None = None
    error: str | None = None
    messages_sent: list[list[dict[str, str]]] = field(default_factory=list)


def judge_call(
    *,
    mode: JudgeMode,
    case_id: str,
    system_prompt: str,
    user_prompt: str,
    wire_schema: dict[str, Any],
    schema_name: str,
    required_criteria: list[str] | None,
    judge: ModelSettings,
    client: LLMClient,
    logger: AuditLogger | None = None,
    stage: str = "run",
    extra_log: dict[str, Any] | None = None,
    bypass_cache: bool = False,
) -> JudgeOutcome:
    """One judged item: up to ``judge.max_repair_attempts`` structured-parse
    attempts, every one of them logged, tokens summed across all of them."""
    messages: list[dict[str, str]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    response_format = (
        response_format_for(judge.model, schema_name, wire_schema)
        if judge.structured_output
        else None
    )

    accounting = CallAccounting()
    outcome = JudgeOutcome(accounting=accounting)
    last_error: str | None = None
    token_budget = judge.max_tokens

    for attempt in range(1, judge.max_repair_attempts + 1):
        sent = [dict(m) for m in messages]
        outcome.messages_sent.append(sent)
        try:
            response = client.complete(
                model=judge.model,
                messages=messages,
                temperature=judge.temperature,
                max_tokens=token_budget,
                response_format=response_format,
                timeout=judge.request_timeout_s,
                max_transient_retries=judge.max_transient_retries,
                bypass_cache=bypass_cache,
            )
        except QuotaExceeded:
            raise  # circuit breaker: abort the run, do not swallow
        except (FatalProviderError, *TRANSIENT_ERRORS) as exc:
            accounting.attempts += 1
            last_error = f"{type(exc).__name__}: {exc}"
            if logger:
                logger.log_call(
                    stage=stage,
                    case_id=case_id,
                    attempt=attempt,
                    model=judge.model,
                    temperature=judge.temperature,
                    messages=_redact(sent),
                    raw_response=None,
                    parse_ok=False,
                    error=last_error,
                    prompt_tokens=0,
                    completion_tokens=0,
                    latency_ms=0.0,
                    cached=False,
                    extra={"mode": mode, "failure": "judge_call_failed", **(extra_log or {})},
                )
            outcome.failure = "judge_call_failed"
            outcome.error = last_error
            return outcome

        accounting.attempts += 1
        accounting.prompt_tokens += response.prompt_tokens
        accounting.completion_tokens += response.completion_tokens
        accounting.latency_ms += response.latency_ms
        if response.cached:
            accounting.cached_calls += 1
        else:
            accounting.api_calls += 1
        outcome.raw_response = response.text

        parse_error: str | None = None
        verdict: PointwiseVerdict | PairwiseVerdict | None = None
        try:
            if mode == "pointwise":
                verdict = parse_pointwise(response.text, required_criteria or [])
            else:
                verdict = parse_pairwise(response.text)
        except (ValidationError, ValueError, json.JSONDecodeError) as exc:
            parse_error = f"{type(exc).__name__}: {exc}"

        if logger:
            logger.log_call(
                stage=stage,
                case_id=case_id,
                attempt=attempt,
                model=judge.model,
                temperature=judge.temperature,
                messages=_redact(sent),
                raw_response=response.text,
                parse_ok=parse_error is None,
                error=parse_error,
                prompt_tokens=response.prompt_tokens,
                completion_tokens=response.completion_tokens,
                latency_ms=response.latency_ms,
                cached=response.cached,
                extra={
                    "mode": mode,
                    "structured_output": response.structured_output_used,
                    "finish_reason": response.finish_reason,
                    **(extra_log or {}),
                },
            )

        if parse_error is None:
            outcome.verdict = verdict
            return outcome

        last_error = parse_error

        if response.finish_reason == "length":
            # Truncation is not a formatting mistake, and treating it as one is
            # self-defeating: echoing the truncated text back makes the prompt
            # LONGER while the output budget stays the same, so every remaining
            # attempt is guaranteed to truncate too. Raise the budget and ask for
            # shorter rationales instead. The bias here matters -- the cases that
            # truncate are the verbose ones, so silently dropping them biases the
            # pass rate toward short answers.
            token_budget = min(token_budget * 2, MAX_REPAIR_TOKEN_BUDGET)
            messages.append({"role": "assistant", "content": response.text})
            messages.append({"role": "user", "content": TRUNCATION_REPAIR_TEMPLATE})
            continue

        # THE REPAIR STEP: the next request differs from this one. It carries the
        # malformed text and the parser's complaint, which is the whole point --
        # re-sending the original messages would be a re-issue, not a repair.
        messages.append({"role": "assistant", "content": response.text})
        messages.append(
            {
                "role": "user",
                "content": REPAIR_TEMPLATE.format(
                    malformed=response.text,
                    error=parse_error,
                    schema=json.dumps(wire_schema, indent=2),
                ),
            }
        )

    outcome.failure = "judge_parse_error"
    outcome.error = last_error
    return outcome


def _redact(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    return [{**m, "content": redact_env_values(m.get("content", ""))} for m in messages]


# --------------------------------------------------------------------------
# Convenience wrappers used by runner/mitigations
# --------------------------------------------------------------------------


def judge_pointwise(
    *,
    case: TestCase,
    rubric: Rubric,
    builder: JudgePromptBuilder,
    judge: ModelSettings,
    client: LLMClient,
    logger: AuditLogger | None = None,
    stage: str = "run",
    bypass_cache: bool = False,
) -> JudgeOutcome:
    effective = rubric.subset(case.criteria)
    criteria = effective.criterion_names
    nonce = content_nonce(
        "pointwise",
        case.case_id,
        case.input,
        case.system_prompt,
        case.model_output,
        case.expected_output or "",
    )
    return judge_call(
        mode="pointwise",
        case_id=case.case_id,
        system_prompt=builder.pointwise_system(criteria),
        user_prompt=builder.pointwise_user(case, nonce),
        wire_schema=pointwise_wire_schema(criteria),
        schema_name="pointwise_verdict",
        required_criteria=criteria,
        judge=judge,
        client=client,
        logger=logger,
        stage=stage,
        bypass_cache=bypass_cache,
        extra_log={"reference_based": case.expected_output is not None, "nonce": nonce},
    )


def judge_pairwise(
    *,
    case_id: str,
    case_input: str,
    system_prompt: str,
    output_a: str,
    output_b: str,
    builder: JudgePromptBuilder,
    judge: ModelSettings,
    client: LLMClient,
    logger: AuditLogger | None = None,
    stage: str = "run",
    expected_output: str | None = None,
    order: str = "forward",
    bypass_cache: bool = False,
) -> JudgeOutcome:
    nonce = content_nonce(
        "pairwise", case_id, case_input, system_prompt, output_a, output_b, expected_output or ""
    )
    return judge_call(
        mode="pairwise",
        case_id=case_id,
        system_prompt=builder.pairwise_system(),
        user_prompt=builder.pairwise_user(
            case_input, system_prompt, output_a, output_b, nonce, expected_output
        ),
        wire_schema=pairwise_wire_schema(),
        schema_name="pairwise_verdict",
        required_criteria=None,
        judge=judge,
        client=client,
        logger=logger,
        stage=stage,
        bypass_cache=bypass_cache,
        extra_log={"order": order, "nonce": nonce},
    )
