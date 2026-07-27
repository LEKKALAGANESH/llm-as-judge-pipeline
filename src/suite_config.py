"""Configuration and rubric loading, plus the cross-family invariant.

Both YAML files are read with ``yaml.safe_load``. Never ``yaml.load`` and never
``FullLoader``: a suite/rubric file is exactly the artifact a user is invited to
receive from someone else and run, which makes arbitrary-object construction on
load a real code-execution path rather than a theoretical one.

The interesting logic here is :func:`derive_family`. Under litellm the routing
prefix is not the model lineage: ``groq/openai/gpt-oss-120b`` is *routed* by
Groq but is family **openai**, and ``bedrock/anthropic.claude-3`` is routed by
Bedrock but is family **anthropic**. A cross-family check written against the
routing prefix would compare "gemini" to "groq", pass for the wrong reason, and
keep passing after someone swapped the judge to a Groq-served Gemini -- exactly
the pairing the invariant exists to prevent.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from .errors import ConfigError, CrossFamilyViolation

# Ordered longest/most-specific first. Matched against the MODEL portion of the
# id, never the routing prefix.
_FAMILY_PATTERNS: tuple[tuple[str, str], ...] = (
    ("gpt-oss", "openai"),
    ("gpt", "openai"),
    ("o3", "openai"),
    ("o4", "openai"),
    ("gemini", "google"),
    ("gemma", "google"),
    ("claude", "anthropic"),
    ("llama", "meta"),
    ("mixtral", "mistral"),
    ("mistral", "mistral"),
    ("qwen", "alibaba"),
    ("deepseek", "deepseek"),
    ("grok", "xai"),
    ("command-r", "cohere"),
)

UNKNOWN_FAMILY = "unknown"

_PROVIDER_KEY_ENV: dict[str, str] = {
    "gemini": "GEMINI_API_KEY",
    "google": "GEMINI_API_KEY",
    "vertex_ai": "GOOGLE_APPLICATION_CREDENTIALS",
    "groq": "GROQ_API_KEY",
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
}


def model_portion(model_id: str) -> str:
    """Strip the litellm routing prefix, leaving the part that names the model.

    ``groq/openai/gpt-oss-120b`` -> ``openai/gpt-oss-120b``
    ``gemini/gemini-3.5-flash``  -> ``gemini-3.5-flash``
    ``gpt-4o``                   -> ``gpt-4o``
    """
    parts = model_id.strip().split("/")
    if len(parts) == 1:
        return parts[0].lower()
    return "/".join(parts[1:]).lower()


def routing_provider(model_id: str) -> str:
    """The litellm routing prefix. Useful for credentials and provider quirks --
    and explicitly NOT used for the family invariant."""
    parts = model_id.strip().split("/")
    return parts[0].lower() if len(parts) > 1 else "openai"


def derive_family(model_id: str) -> str:
    """Model lineage, derived from the model portion of the id.

    Returns :data:`UNKNOWN_FAMILY` when no pattern matches. Unknown is treated
    as a *failure* by the invariant check rather than a pass, because a family
    that cannot be resolved cannot be asserted to differ from anything.
    """
    portion = model_portion(model_id)
    for needle, family in _FAMILY_PATTERNS:
        if needle in portion:
            return family
    return UNKNOWN_FAMILY


def api_key_env_for(model_id: str) -> str:
    return _PROVIDER_KEY_ENV.get(routing_provider(model_id), "OPENAI_API_KEY")


# --------------------------------------------------------------------------
# Config models
# --------------------------------------------------------------------------


class RunSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    results_dir: str = "results"
    logs_dir: str = "logs"
    cache_path: str = ".cache/llm_responses"
    max_calls_per_run: int = Field(default=400, ge=1)
    max_cost_usd: float = Field(default=5.0, ge=0.0)
    max_concurrent_judge_calls: int = Field(default=2, ge=1)
    #: Provider TPM ceiling for the judge model, paced client-side before each
    #: request. ``None`` disables pacing.
    tokens_per_minute: int | None = Field(default=None, ge=1)


class ModelSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_tokens: int = Field(default=2048, ge=64)
    max_repair_attempts: int = Field(default=3, ge=1)
    max_transient_retries: int = Field(default=3, ge=1)
    request_timeout_s: int = Field(default=120, ge=1)
    structured_output: bool = True

    @property
    def family(self) -> str:
        return derive_family(self.model)

    @property
    def provider(self) -> str:
        return routing_provider(self.model)

    @property
    def api_key_env(self) -> str:
        return api_key_env_for(self.model)

    def with_temperature(self, temperature: float) -> ModelSettings:
        return self.model_copy(update={"temperature": temperature})


class CandidateSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    label: str
    model: str
    prompt_variant: str | None = None

    @property
    def family(self) -> str:
        return derive_family(self.model)


class ScoringSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    pass_threshold: int = Field(default=4, ge=1, le=5)
    min_criterion_score: int = Field(default=3, ge=1, le=5)


class ComparisonSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    flip_rate_threshold: float = Field(default=0.20, ge=0.0, le=1.0)
    win_rate_threshold: float = Field(default=0.55, ge=0.5, le=1.0)


class ValidationSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    gold_labels_path: str = "data/gold_labels/human_annotated.json"
    probes_path: str = "data/test_suites/adversarial_probes.json"
    test_retest_runs: int = Field(default=5, ge=2)
    test_retest_temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    test_retest_also_at_judge_temperature: bool = True
    bootstrap_resamples: int = Field(default=2000, ge=100)
    bootstrap_seed: int = 20260726
    kappa_labels: list[int] = Field(default_factory=lambda: [1, 2, 3, 4, 5])


class MitigationSettings(BaseModel):
    """The three ablatable switches, plus one that is structural.

    ``rationale_first`` is the verdict schema's field order. It cannot be turned
    off without shipping a second schema, so `--mitigations off` does not touch
    it; the README says so plainly rather than implying an ablation that did not
    happen. Its effect is measured by the sycophancy probes instead.
    """

    model_config = ConfigDict(extra="forbid")

    order_swap: bool = True
    anti_verbosity_clause: bool = True
    few_shot_anchors: bool = True
    rationale_first: bool = True

    @classmethod
    def all_off(cls) -> MitigationSettings:
        return cls(
            order_swap=False,
            anti_verbosity_clause=False,
            few_shot_anchors=False,
            rationale_first=True,
        )

    def as_dict(self) -> dict[str, bool]:
        return self.model_dump()


class TokenPrice(BaseModel):
    model_config = ConfigDict(extra="forbid")

    input: float = 0.0
    output: float = 0.0


class PipelineConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: str = "1.0"
    run: RunSettings = Field(default_factory=RunSettings)
    judge: ModelSettings
    self_enhancement_probe_judge: ModelSettings | None = None
    candidates: list[CandidateSpec] = Field(default_factory=list)
    generator_family: str | None = None
    scoring: ScoringSettings = Field(default_factory=ScoringSettings)
    comparison: ComparisonSettings = Field(default_factory=ComparisonSettings)
    validation: ValidationSettings = Field(default_factory=ValidationSettings)
    mitigations: MitigationSettings = Field(default_factory=MitigationSettings)
    pricing_usd_per_million_tokens: dict[str, TokenPrice] = Field(default_factory=dict)

    # Populated by enforce_cross_family; echoed into every report so the claim
    # is auditable rather than asserted.
    resolved_families: dict[str, str] = Field(default_factory=dict)
    cross_family_enforced: bool = True

    @field_validator("generator_family")
    @classmethod
    def _normalise_family(cls, v: str | None) -> str | None:
        return v.strip().lower() if isinstance(v, str) else v

    def price_for(self, model_id: str) -> TokenPrice:
        return self.pricing_usd_per_million_tokens.get(model_id, TokenPrice())

    def estimate_cost_usd(self, model_id: str, prompt_tokens: int, completion_tokens: int) -> float:
        price = self.price_for(model_id)
        return (prompt_tokens * price.input + completion_tokens * price.output) / 1_000_000


def declared_generator_families(config: PipelineConfig) -> dict[str, str]:
    """Every generator family this run claims to be judging, keyed by label."""
    families: dict[str, str] = {}
    for cand in config.candidates:
        families[f"candidate:{cand.label}"] = cand.family
    if config.generator_family:
        families["generator_family"] = config.generator_family
    return families


def enforce_cross_family(
    config: PipelineConfig,
    *,
    allow_same_family: bool = False,
) -> PipelineConfig:
    """Assert the self-enhancement mitigation, or refuse to run.

    Two holes are guarded explicitly:

    * ``all(judge.family != c.family for c in candidates)`` is vacuously True
      over an empty candidate list, and in the default flow (pre-generated
      ``model_output``) no candidate model need be configured at all. So a
      non-empty ``candidates`` list OR an explicit ``generator_family`` is
      required before the mitigation may be claimed.
    * An unresolvable family is a failure, not a pass.
    """
    judge_family = config.judge.family
    generators = declared_generator_families(config)

    config.resolved_families = {"judge": judge_family, **generators}
    config.cross_family_enforced = not allow_same_family

    if allow_same_family:
        return config

    if not generators:
        raise CrossFamilyViolation(
            "Cannot verify the self-enhancement mitigation: no candidates are configured and "
            "`generator_family` is unset, which makes the cross-family assertion vacuously true. "
            "Set `generator_family:` in suite_config.yaml to the family that produced the "
            "model_output values, or configure `candidates:`."
        )

    unknown = [k for k, v in {"judge": judge_family, **generators}.items() if v == UNKNOWN_FAMILY]
    if unknown:
        raise CrossFamilyViolation(
            "Cannot resolve the model family for: "
            + ", ".join(unknown)
            + ". The cross-family invariant cannot be asserted for a model whose lineage is "
            "unknown. Add a pattern to _FAMILY_PATTERNS in src/suite_config.py, or set "
            "`generator_family:` explicitly."
        )

    conflicts = [f"{label} ({fam})" for label, fam in generators.items() if fam == judge_family]
    if conflicts:
        raise CrossFamilyViolation(
            f"Self-enhancement bias invariant violated: judge {config.judge.model!r} resolves to "
            f"family {judge_family!r}, which is shared by {', '.join(conflicts)}. "
            "A judge must not share a lineage with any model it grades. Refusing to run. "
            "(Note the routing prefix is irrelevant here -- family is derived from the model "
            "portion of the id, so groq/openai/gpt-oss-120b is family 'openai', not 'groq'.) "
            "If this run is the deliberate same-family self-enhancement measurement, pass "
            "--allow-same-family."
        )
    return config


# --------------------------------------------------------------------------
# Rubric models
# --------------------------------------------------------------------------


class ScaleSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    min: int = 1
    max: int = 5
    description: str = ""


class RubricCriterion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    description: str
    rationale: str = ""
    anchors: dict[str, str] = Field(default_factory=dict)

    @field_validator("anchors", mode="before")
    @classmethod
    def _stringify_anchor_keys(cls, v: Any) -> Any:
        if isinstance(v, dict):
            return {str(k): val for k, val in v.items()}
        return v


class RubricClauses(BaseModel):
    model_config = ConfigDict(extra="forbid")

    anti_verbosity: str = ""
    grounding: str = ""
    injection_defense: str = ""


class Rubric(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: str = "1.0"
    scale: ScaleSpec = Field(default_factory=ScaleSpec)
    criteria: list[RubricCriterion]
    clauses: RubricClauses = Field(default_factory=RubricClauses)

    @property
    def criterion_names(self) -> list[str]:
        return [c.name for c in self.criteria]

    def subset(self, names: list[str] | None) -> Rubric:
        """Per-case ``criteria`` override -- a subset of the rubric, validated."""
        if not names:
            return self
        by_name = {c.name: c for c in self.criteria}
        missing = [n for n in names if n not in by_name]
        if missing:
            raise ConfigError(
                f"test case requests criteria not present in the rubric: {missing}. "
                f"Available: {self.criterion_names}"
            )
        return self.model_copy(update={"criteria": [by_name[n] for n in names]})


# --------------------------------------------------------------------------
# Loaders
# --------------------------------------------------------------------------


def _read_yaml(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.is_file():
        raise ConfigError(f"config file not found: {p}")
    with p.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)  # safe_load only -- see module docstring
    if not isinstance(data, dict):
        raise ConfigError(f"{p} must contain a YAML mapping at the top level, got {type(data)}")
    return data


def load_config(
    path: str | Path = "config/suite_config.yaml",
    *,
    allow_same_family: bool = False,
    enforce: bool = True,
) -> PipelineConfig:
    raw = _read_yaml(path)
    try:
        config = PipelineConfig.model_validate(raw)
    except ValidationError as exc:
        raise ConfigError(f"{path} failed validation:\n{exc}") from exc
    if enforce:
        config = enforce_cross_family(config, allow_same_family=allow_same_family)
    else:
        config.resolved_families = {
            "judge": config.judge.family,
            **declared_generator_families(config),
        }
    return config


def load_rubric(path: str | Path = "config/rubric.yaml") -> Rubric:
    raw = _read_yaml(path)
    try:
        return Rubric.model_validate(raw)
    except ValidationError as exc:
        raise ConfigError(f"{path} failed validation:\n{exc}") from exc


def check_api_key(model_id: str) -> None:
    """Fail fast, naming the exact environment variable, before any call."""
    env = api_key_env_for(model_id)
    if not os.environ.get(env):
        raise ConfigError(
            f"{env} is not set, and it is required to call {model_id!r}. "
            "Copy .env.example to .env and fill it in."
        )


def warn_if_deterministic_test_retest(temperature: float, *, acknowledged: bool) -> str | None:
    """Test-retest at T=0 measures almost nothing.

    Returns a warning string (the caller decides whether to print or raise).
    Note the premise is softer than it is usually stated: modern serving stacks
    are not bit-deterministic, so even T=0 produces a small non-zero flip rate.
    A flip rate of *exactly* 0.0 is the real red flag -- it suggests the
    temperature setting never took effect at all, which is why the validator
    also diffs the raw rationale strings when that happens.
    """
    if temperature > 0:
        return None
    msg = (
        "test-retest is configured at temperature 0. Consistency will be close to trivial and "
        "the resulting number is not a validation result. Set validation.test_retest_temperature "
        "to >= 0.7, or re-run with --acknowledge-t0 to record the caveat explicitly."
    )
    if acknowledged:
        return "ACKNOWLEDGED: " + msg
    raise ConfigError(msg)
