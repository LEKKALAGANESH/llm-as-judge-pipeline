"""Single source of truth for every structured shape in the pipeline.

This module has zero internal imports by design (architecture invariant): it is
a leaf, so nothing here can create a cycle and everything else may depend on it.

Two field-ordering decisions in this file are mitigations, not style:

1. ``CriterionScore`` declares ``rationale`` before ``score``, and
   ``PointwiseVerdict`` declares ``overall_rationale`` before ``overall_score``.
   Under constrained / structured decoding the model emits JSON fields in
   schema order. A schema that put ``score`` first would make the judge commit
   to a number before generating any reasoning, turning the rationale into
   post-hoc rationalisation and silently voiding the "force per-criterion
   grounding before the score" sycophancy mitigation this project claims. There
   is a unit test asserting the order, and the wire schema sent to the provider
   is generated from these models so the two cannot drift.

2. There is no ``passed`` field on any verdict. Pass/fail is computed in code
   from ``suite_config.scoring`` (see :func:`compute_passed`). A judge-declared
   boolean would be an uncalibrated fourth judgment with no anchors, and it
   would make the pass bar un-retunable without re-spending the whole suite.

``overall_score`` is an **integer** 1-5. That is deliberate: Cohen's kappa is a
categorical/ordinal statistic, and computing it over floats makes sklearn treat
every distinct float as its own class, returning a plausible-looking number over
arbitrary observed values. The mean of the six criteria is available as the
derived float :attr:`PointwiseVerdict.mean_criterion_score`, which is used for
correlation-style statistics (Spearman, ICC) where a continuous quantity is
correct.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    TypeAdapter,
    ValidationInfo,
    model_validator,
)

SCORE_MIN = 1
SCORE_MAX = 5

Score = Annotated[int, Field(ge=SCORE_MIN, le=SCORE_MAX)]


# --------------------------------------------------------------------------
# Inputs
# --------------------------------------------------------------------------


class TestCase(BaseModel):
    """One pointwise case: the shape the assignment specifies, typed."""

    # Not a pytest class despite the name; stops pytest trying to collect it.
    __test__ = False

    model_config = ConfigDict(extra="forbid")

    case_id: str
    input: str
    system_prompt: str = ""
    model_output: str
    expected_output: str | None = None
    criteria: list[str] | None = Field(
        default=None,
        description="Optional per-case override of the rubric criteria subset to score.",
    )
    tags: list[str] = Field(default_factory=list)
    notes: str | None = None


class TestSuite(BaseModel):
    __test__ = False

    model_config = ConfigDict(extra="forbid")

    suite_id: str
    description: str = ""
    provenance: dict[str, Any] = Field(default_factory=dict)
    cases: list[TestCase]

    @model_validator(mode="after")
    def _unique_case_ids(self) -> TestSuite:
        seen: set[str] = set()
        for i, case in enumerate(self.cases):
            if case.case_id in seen:
                raise ValueError(f"duplicate case_id {case.case_id!r} at cases[{i}]")
            seen.add(case.case_id)
        return self


class ComparisonCase(BaseModel):
    """One pairwise A/B case. ``output_a``/``output_b`` come from two configs."""

    model_config = ConfigDict(extra="forbid")

    case_id: str
    input: str
    system_prompt: str = ""
    output_a: str
    output_b: str
    config_a_label: str
    config_b_label: str
    expected_output: str | None = None
    tags: list[str] = Field(default_factory=list)
    source: str | None = Field(
        default=None,
        description="Provenance of the two outputs: which model/prompt produced them, and when.",
    )


class ComparisonSuite(BaseModel):
    model_config = ConfigDict(extra="forbid")

    suite_id: str
    description: str = ""
    config_a_label: str
    config_b_label: str
    provenance: dict[str, Any] = Field(default_factory=dict)
    cases: list[ComparisonCase]

    @model_validator(mode="after")
    def _unique_case_ids(self) -> ComparisonSuite:
        seen: set[str] = set()
        for i, case in enumerate(self.cases):
            if case.case_id in seen:
                raise ValueError(f"duplicate case_id {case.case_id!r} at cases[{i}]")
            seen.add(case.case_id)
        return self


# --------------------------------------------------------------------------
# Verdicts (what the judge is asked to emit)
# --------------------------------------------------------------------------


class CriterionScore(BaseModel):
    """Per-criterion judgment.

    FIELD ORDER IS THE MITIGATION. ``rationale`` must precede ``score``.
    See the module docstring and ``tests/test_schema.py::test_field_order_*``.
    """

    model_config = ConfigDict(extra="forbid")

    rationale: str = Field(min_length=1)
    score: Score


class PointwiseVerdict(BaseModel):
    """Absolute score of one output against the rubric.

    Note the absence of ``passed`` -- see :func:`compute_passed`.
    """

    model_config = ConfigDict(extra="forbid")

    criteria_breakdown: dict[str, CriterionScore]
    overall_rationale: str = Field(min_length=1)
    overall_score: Score

    @model_validator(mode="after")
    def _breakdown_covers_required_criteria(self, info: ValidationInfo) -> PointwiseVerdict:
        """Every rubric criterion must be present, not a convenient subset.

        The required set is supplied through pydantic's validation context so
        that this module stays free of internal imports:

            PointwiseVerdict.model_validate_json(
                raw, context={"required_criteria": [...]}
            )

        With no context supplied (e.g. reading back a stored verdict) the check
        is skipped, which keeps round-trip deserialisation cheap and total.
        """
        ctx = info.context or {}
        required = ctx.get("required_criteria")
        if not required:
            return self
        missing = [c for c in required if c not in self.criteria_breakdown]
        extra = [c for c in self.criteria_breakdown if c not in required]
        problems = []
        if missing:
            problems.append(f"missing criteria: {missing}")
        if extra:
            problems.append(f"unexpected criteria: {extra}")
        if problems:
            raise ValueError("; ".join(problems))
        return self

    @property
    def mean_criterion_score(self) -> float:
        scores = [c.score for c in self.criteria_breakdown.values()]
        return sum(scores) / len(scores) if scores else 0.0

    @property
    def min_criterion_score(self) -> int:
        scores = [c.score for c in self.criteria_breakdown.values()]
        return min(scores) if scores else 0


PairwiseWinner = Literal["Model_A", "Model_B", "Tie"]


class PairwiseVerdict(BaseModel):
    """A-vs-B judgment. ``rationale`` precedes ``winner`` for the same reason
    ``rationale`` precedes ``score``: the reasoning must not be post-hoc."""

    model_config = ConfigDict(extra="forbid")

    rationale: str = Field(min_length=1)
    winner: PairwiseWinner


def compute_passed(
    verdict: PointwiseVerdict,
    *,
    pass_threshold: int,
    min_criterion_score: int,
) -> bool:
    """The pass rule, in code, from config -- never emitted by the judge."""
    return (
        verdict.overall_score >= pass_threshold
        and verdict.min_criterion_score >= min_criterion_score
    )


# --------------------------------------------------------------------------
# Per-case results (what the runner emits; what the aggregator consumes)
# --------------------------------------------------------------------------


class CallAccounting(BaseModel):
    """Token/latency/attempt totals, summed across repair and transient retries
    so that retried calls are never invisible to cost accounting."""

    model_config = ConfigDict(extra="forbid")

    api_calls: int = 0
    cached_calls: int = 0
    attempts: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0

    def merged(self, other: CallAccounting) -> CallAccounting:
        return CallAccounting(
            api_calls=self.api_calls + other.api_calls,
            cached_calls=self.cached_calls + other.cached_calls,
            attempts=self.attempts + other.attempts,
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            latency_ms=self.latency_ms + other.latency_ms,
        )


FailureKind = Literal["judge_parse_error", "judge_call_failed"]


class PointwiseCaseResult(BaseModel):
    """One pointwise case, judged. ``verdict`` is None when the case failed;
    failures are counted and reported, never silently dropped."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["pointwise"] = "pointwise"
    case_id: str
    verdict: PointwiseVerdict | None = None
    passed: bool | None = None
    failure: FailureKind | None = None
    error: str | None = None
    output_length: int = Field(
        default=0,
        description="len(model_output) in characters -- the x-axis of the verbosity correlation.",
    )
    reference_based: bool = False
    judge_model: str = ""
    judge_temperature: float = 0.0
    accounting: CallAccounting = Field(default_factory=CallAccounting)
    run_id: str = ""
    tags: list[str] = Field(default_factory=list)


class PairwiseCaseResult(BaseModel):
    """One pairwise case, judged in both orders (unless order-swap is ablated).

    ``final_winner`` is None exactly when ``incomplete`` is True. A pair where
    one order failed after retries is excluded from the win/loss/tie counts AND
    from the flip-rate denominator, and surfaced separately as an incomplete
    pair -- resolving it from the single successful order would silently
    reintroduce the exact position bias the protocol exists to remove.
    """

    model_config = ConfigDict(extra="forbid")

    kind: Literal["pairwise"] = "pairwise"
    case_id: str
    forward_winner: PairwiseWinner | None = None
    reverse_winner: PairwiseWinner | None = Field(
        default=None,
        description="Already remapped into the original (A, B) frame.",
    )
    final_winner: Literal["Model_A", "Model_B", "Tie", "Tie (Position Inconsistency)"] | None = None
    position_consistent: bool | None = Field(
        default=None,
        description="None when order-swap is disabled (ablation) or the pair is incomplete.",
    )
    order_swapped: bool = True
    incomplete: bool = False
    failure: FailureKind | None = None
    error: str | None = None
    forward_rationale: str | None = None
    reverse_rationale: str | None = None
    length_a: int = 0
    length_b: int = 0
    judge_model: str = ""
    judge_temperature: float = 0.0
    accounting: CallAccounting = Field(default_factory=CallAccounting)
    run_id: str = ""
    tags: list[str] = Field(default_factory=list)


Verdict = Annotated[
    PointwiseCaseResult | PairwiseCaseResult,
    Field(discriminator="kind"),
]

VerdictAdapter: TypeAdapter[Verdict] = TypeAdapter(Verdict)
VerdictListAdapter: TypeAdapter[list[Verdict]] = TypeAdapter(list[Verdict])


# --------------------------------------------------------------------------
# Aggregated report
# --------------------------------------------------------------------------


class ScoreDistribution(BaseModel):
    """Score-clustering measurement. "We added anchors" is not a measurement;
    "scores collapsed to 4-or-5 on 71% of cases without anchors, 38% with" is."""

    model_config = ConfigDict(extra="forbid")

    n: int
    mean: float | None = None
    std_dev: float | None = None
    modal_value: int | None = None
    modal_share: float | None = None
    entropy_bits: float | None = Field(
        default=None, description="Shannon entropy over the 5 score levels, in bits (max log2(5))."
    )
    top_two_share: float | None = Field(
        default=None, description="Share of scores at 4 or 5 -- the clustering symptom."
    )
    histogram: dict[str, int] = Field(default_factory=dict)


class CorrelationResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    n: int
    rho: float | None = None
    p_value: float | None = None
    reason: str | None = Field(
        default=None, description="Set when rho is None (too few points, or zero variance)."
    )


class Interval(BaseModel):
    model_config = ConfigDict(extra="forbid")

    low: float
    high: float
    method: str
    confidence: float = 0.95


class PairedTestResult(BaseModel):
    """A significance test on a *paired* design.

    Ablation before/after, and the two self-enhancement judges, both score the
    same cases. Reporting only the raw difference leaves the reader unable to
    tell a real effect from resampling noise at n=15-18, which is precisely the
    gap the rubric line "empirical proof, not assertions" is about. The paired
    test is the standard answer and costs no API calls: the data is on disk.
    """

    model_config = ConfigDict(extra="forbid")

    test: str = Field(description="wilcoxon-signed-rank | mcnemar-exact")
    n: int = Field(description="Pairs contributing to the test (ties are excluded by Wilcoxon).")
    statistic: float | None = None
    p_value: float | None = None
    effect: float | None = Field(
        default=None, description="Median paired difference, or the discordant-pair split."
    )
    adjusted_p_value: float | None = Field(
        default=None, description="Holm-Bonferroni adjusted across the reported family."
    )
    significant: bool | None = Field(
        default=None, description="Against alpha after adjustment, when an adjustment was applied."
    )
    reason: str | None = Field(default=None, description="Set when the test could not be run.")


class PairwiseTally(BaseModel):
    model_config = ConfigDict(extra="forbid")

    n_cases: int = 0
    wins_a: int = 0
    wins_b: int = 0
    ties: int = 0
    inconsistent_pairs: int = 0
    incomplete_pairs: int = 0
    completed_pairs: int = Field(
        default=0, description="wins_a + wins_b + ties + inconsistent (the flip-rate denominator)."
    )
    resolved_pairs: int = Field(
        default=0, description="wins_a + wins_b -- excludes ties AND position-inconsistent pairs."
    )
    win_rate_a: float | None = None
    win_rate_b: float | None = None
    flip_rate: float | None = None
    win_rate_b_ci: Interval | None = None


class SuiteReport(BaseModel):
    """Pure function of a list of verdicts plus run metadata."""

    model_config = ConfigDict(extra="forbid")

    run_id: str
    generated_at: str
    mode: Literal["pointwise", "pairwise"]
    suite_id: str = ""
    suite_path: str = ""
    judge_model: str = ""
    judge_temperature: float = 0.0
    mitigations: dict[str, bool] = Field(default_factory=dict)
    resolved_families: dict[str, str] = Field(default_factory=dict)
    cross_family_enforced: bool = True

    n_cases: int = 0
    n_scored: int = 0
    n_parse_errors: int = 0
    n_call_failures: int = 0
    failed_case_ids: list[str] = Field(default_factory=list)

    # pointwise
    pass_threshold: int | None = None
    min_criterion_score: int | None = None
    pass_rate: float | None = None
    n_passed: int | None = None
    mean_overall_score: float | None = None
    mean_scores_by_criterion: dict[str, float] = Field(default_factory=dict)
    overall_score_distribution: ScoreDistribution | None = None
    criterion_score_distributions: dict[str, ScoreDistribution] = Field(default_factory=dict)
    length_vs_score_spearman: CorrelationResult | None = None

    # pairwise
    tally: PairwiseTally | None = None
    config_a_label: str | None = None
    config_b_label: str | None = None
    winner: str | None = None
    decision_rule: str | None = None
    decision_trace: list[str] = Field(default_factory=list)

    # accounting
    total_judge_calls: int = 0
    total_cached_calls: int = 0
    total_attempts: int = 0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_latency_ms: float = 0.0
    estimated_cost_usd: float = 0.0
    cost_note: str = ""
