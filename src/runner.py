"""Orchestration: iterate cases, apply the mode rule, call the judge, emit verdicts.

Deliberately separate from ``aggregator.py``. The runner does all the I/O and all
the LLM calls and produces ``list[Verdict]``; the aggregator is a pure reduction
over that list. Keeping them in one module -- which is the natural thing to do --
is what makes the pass-rate/win-rate/flip-rate arithmetic untestable without a
network, and that arithmetic sits behind the two largest rubric lines.

Concurrency is bounded by ``run.max_concurrent_judge_calls`` and defaults to 1.
That is not timidity, and the reason is worth stating precisely because it is
the one most people get wrong: **tokens per minute binds before requests per
minute**. Measured on the free tier, llama-3.3-70b-versatile allows 12,000 TPM
while one judge call costs ~4,600 tokens -- about 2.6 calls/minute, regardless
of the 30 RPM the same tier advertises. A concurrent driver simply requests two
calls' worth of tokens at once and 429s on the first one.

The same measurement is why the judge is not on Gemini: that free tier metered
gemini-3.5-flash at 20 requests per *day*
(``GenerateRequestsPerDayPerProjectPerModel-FreeTier``), against a validation
suite needing ~370 calls.

The two orders of a single pairwise case always run sequentially, so the audit
log reads in a sensible order per case.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import TypeVar

from .aggregator import select_mode
from .audit_log import AuditLogger, new_run_id
from .judge import JudgeOutcome, JudgePromptBuilder, judge_pairwise, judge_pointwise
from .llm_client import LLMClient
from .mitigations import evaluate_pairwise_unbiased
from .schema import (
    ComparisonCase,
    PairwiseCaseResult,
    PointwiseCaseResult,
    TestCase,
    Verdict,
    compute_passed,
)
from .suite_config import MitigationSettings, ModelSettings, PipelineConfig, Rubric

T = TypeVar("T")
R = TypeVar("R")

__all__ = [
    "RunContext",
    "run_pointwise_cases",
    "run_pairwise_cases",
    "select_mode",
]


@dataclass
class RunContext:
    """Everything one run needs, assembled once."""

    config: PipelineConfig
    rubric: Rubric
    client: LLMClient
    logger: AuditLogger
    judge: ModelSettings
    mitigations: MitigationSettings
    run_id: str
    bypass_cache: bool = False
    """Set for test-retest and noise-floor runs. Without it the disk cache would
    return byte-identical responses for a repeated prompt and manufacture a
    perfect 0% flip rate -- the exact meaningless validation number the protocol
    exists to avoid."""
    builder: JudgePromptBuilder = field(init=False)

    def __post_init__(self) -> None:
        self.builder = JudgePromptBuilder(self.rubric, self.mitigations)

    @classmethod
    def build(
        cls,
        config: PipelineConfig,
        rubric: Rubric,
        *,
        judge: ModelSettings | None = None,
        mitigations: MitigationSettings | None = None,
        client: LLMClient | None = None,
        logger: AuditLogger | None = None,
        run_id: str | None = None,
        run_prefix: str = "run",
        bypass_cache: bool = False,
    ) -> RunContext:
        rid = run_id or new_run_id(run_prefix)
        return cls(
            config=config,
            rubric=rubric,
            client=client
            or LLMClient(
                cache_path=config.run.cache_path,
                max_calls=config.run.max_calls_per_run,
                max_cost_usd=config.run.max_cost_usd,
                price_lookup=lambda m: (
                    config.price_for(m).input,
                    config.price_for(m).output,
                ),
            ),
            logger=logger or AuditLogger(config.run.logs_dir, run_id=rid),
            judge=judge or config.judge,
            mitigations=mitigations or config.mitigations,
            run_id=rid,
            bypass_cache=bypass_cache,
        )


def _map_bounded(fn: Callable[[T], R], items: Sequence[T], workers: int) -> list[R]:
    """Order-preserving bounded map. Sequential when workers <= 1."""
    if workers <= 1 or len(items) <= 1:
        return [fn(item) for item in items]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(fn, items))


# --------------------------------------------------------------------------
# Pointwise
# --------------------------------------------------------------------------


def run_pointwise_cases(
    cases: Sequence[TestCase],
    ctx: RunContext,
    *,
    stage: str = "run",
    resume_from: Sequence[Verdict] = (),
) -> list[Verdict]:
    """Judge each case once and emit one verdict per case, failures included."""
    done = {v.case_id: v for v in resume_from if isinstance(v, PointwiseCaseResult)}
    pending = [c for c in cases if c.case_id not in done]

    def _one(case: TestCase) -> PointwiseCaseResult:
        outcome: JudgeOutcome = judge_pointwise(
            case=case,
            rubric=ctx.rubric,
            builder=ctx.builder,
            judge=ctx.judge,
            client=ctx.client,
            logger=ctx.logger,
            stage=stage,
            bypass_cache=ctx.bypass_cache,
        )
        verdict = outcome.verdict
        result = PointwiseCaseResult(
            case_id=case.case_id,
            verdict=verdict,  # type: ignore[arg-type]
            passed=(
                compute_passed(
                    verdict,  # type: ignore[arg-type]
                    pass_threshold=ctx.config.scoring.pass_threshold,
                    min_criterion_score=ctx.config.scoring.min_criterion_score,
                )
                if verdict is not None
                else None
            ),
            failure=outcome.failure,
            error=outcome.error,
            output_length=len(case.model_output),
            reference_based=case.expected_output is not None,
            judge_model=ctx.judge.model,
            judge_temperature=ctx.judge.temperature,
            accounting=outcome.accounting,
            run_id=ctx.run_id,
            tags=list(case.tags),
        )
        ctx.logger.log_verdict(result, stage=stage, mitigations=ctx.mitigations.as_dict())
        return result

    fresh = _map_bounded(_one, pending, ctx.config.run.max_concurrent_judge_calls)
    by_id = {**done, **{v.case_id: v for v in fresh}}
    return [by_id[c.case_id] for c in cases if c.case_id in by_id]


# --------------------------------------------------------------------------
# Pairwise
# --------------------------------------------------------------------------


def run_pairwise_cases(
    cases: Sequence[ComparisonCase],
    ctx: RunContext,
    *,
    stage: str = "compare",
    resume_from: Sequence[Verdict] = (),
) -> list[Verdict]:
    """Judge each pair in both orders (unless order-swap is ablated off)."""
    done = {v.case_id: v for v in resume_from if isinstance(v, PairwiseCaseResult)}
    pending = [c for c in cases if c.case_id not in done]

    def _one(case: ComparisonCase) -> PairwiseCaseResult:
        def _judge_pair(first: str, second: str, order: str) -> JudgeOutcome:
            return judge_pairwise(
                case_id=case.case_id,
                case_input=case.input,
                system_prompt=case.system_prompt,
                output_a=first,
                output_b=second,
                builder=ctx.builder,
                judge=ctx.judge,
                client=ctx.client,
                logger=ctx.logger,
                stage=stage,
                expected_output=case.expected_output,
                order=order,
                bypass_cache=ctx.bypass_cache,
            )

        result = evaluate_pairwise_unbiased(
            _judge_pair,
            case_id=case.case_id,
            output_a=case.output_a,
            output_b=case.output_b,
            order_swap=ctx.mitigations.order_swap,
            judge_model=ctx.judge.model,
            judge_temperature=ctx.judge.temperature,
            run_id=ctx.run_id,
            tags=list(case.tags),
        )
        ctx.logger.log_verdict(result, stage=stage, mitigations=ctx.mitigations.as_dict())
        return result

    fresh = _map_bounded(_one, pending, ctx.config.run.max_concurrent_judge_calls)
    by_id = {**done, **{v.case_id: v for v in fresh}}
    return [by_id[c.case_id] for c in cases if c.case_id in by_id]
