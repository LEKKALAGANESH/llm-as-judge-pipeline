"""Pure reduction: ``list[Verdict] -> SuiteReport``. No I/O, no LLM calls.

The split between this module and ``runner.py`` exists so that the arithmetic
behind the two largest rubric lines -- pass rate, mean per criterion, win rate,
flip rate, and the A/B decision -- is unit-testable against hand-computed
fixtures without a network, an API key, or a mock. Everything in here is a
function of its arguments.

One deliberate choice: ``passed`` is recomputed here from the thresholds rather
than read off the stored verdict. That is what makes the pass bar re-tunable
without re-spending the whole suite, and it means ``replay`` can answer "what
would the pass rate have been at threshold 3?" from the log alone.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

from .errors import PipelineError
from .schema import (
    PairwiseCaseResult,
    PairwiseTally,
    PointwiseCaseResult,
    ScoreDistribution,
    SuiteReport,
    Verdict,
    compute_passed,
)
from .stats import clopper_pearson_interval, score_distribution, spearman

Mode = Literal["pairwise", "pointwise_reference_based", "pointwise_reference_free"]

DECISION_RULE_TEXT = (
    "Step 0 (coverage gate, evaluated first and short-circuiting): if "
    "completed_pairs/n_cases < min_completion_rate, declare insufficient coverage and stop. "
    "This precedes the validity gate because the flip rate is computed over completed pairs "
    "only: a run that lost most of its pairs to rate limiting reports a clean flip rate on the "
    "survivors and would otherwise pass a gate that cannot see what is missing. The lost pairs "
    "are also not a random sample -- the largest prompts fail first. "
    "Step 1 (validity gate, short-circuiting): if "
    "flip_rate > flip_rate_threshold, declare the judge unreliable on this suite and stop -- "
    "no winner. This is a precondition on the instrument, not a third criterion on the effect; "
    "ANDing it with the effect terms would collapse 'the judge is untrustworthy' and 'the two "
    "configs are equivalent' into one output, which are operationally opposite conclusions. "
    "Step 2 (effect): resolved = wins_a + wins_b, which excludes ties AND position-inconsistent "
    "pairs. If wins_b/resolved > win_rate_threshold the winner is config B; if wins_a/resolved > "
    "win_rate_threshold the winner is config A; otherwise inconclusive. There is no mean-score "
    "term: a comparison run is pairwise-only and PairwiseVerdict carries no numeric score, so a "
    "mean-score criterion would not be computable from the run it is applied to. Pairs where "
    "either order failed after retries are excluded from both the win counts and the flip-rate "
    "denominator and are reported as incomplete_pairs."
)


def select_mode(*, comparing_two_configs: bool, expected_output: str | None) -> Mode:
    """The mode-selection rule, implemented in exactly one place.

    This single rule is what lets the pipeline support all four modes named in
    the brief without four code paths: reference-based and reference-free are a
    prompt-construction parameter of one pointwise call, and pairwise is the one
    structurally distinct second call.
    """
    if comparing_two_configs:
        return "pairwise"
    if expected_output is not None:
        return "pointwise_reference_based"
    return "pointwise_reference_free"


# --------------------------------------------------------------------------
# Pointwise arithmetic
# --------------------------------------------------------------------------


def mean_scores_by_criterion(
    verdicts: Sequence[PointwiseCaseResult],
    *,
    order: Sequence[str] = (),
) -> dict[str, float]:
    """Mean score per criterion. ``order`` (the rubric's criterion order) fixes
    the key order so two reports can be diffed line by line."""
    sums: dict[str, float] = {}
    counts: dict[str, int] = {}
    for v in verdicts:
        if v.verdict is None:
            continue
        for name, cs in v.verdict.criteria_breakdown.items():
            sums[name] = sums.get(name, 0.0) + cs.score
            counts[name] = counts.get(name, 0) + 1
    names = [n for n in order if n in sums] + [n for n in sums if n not in set(order)]
    return {name: sums[name] / counts[name] for name in names if counts[name]}


def criterion_distributions(
    verdicts: Sequence[PointwiseCaseResult],
) -> dict[str, ScoreDistribution]:
    buckets: dict[str, list[int]] = {}
    for v in verdicts:
        if v.verdict is None:
            continue
        for name, cs in v.verdict.criteria_breakdown.items():
            buckets.setdefault(name, []).append(cs.score)
    return {name: score_distribution(scores) for name, scores in buckets.items()}


# --------------------------------------------------------------------------
# Pairwise arithmetic
# --------------------------------------------------------------------------


def tally_pairwise(verdicts: Sequence[PairwiseCaseResult]) -> PairwiseTally:
    tally = PairwiseTally(n_cases=len(verdicts))
    for v in verdicts:
        if v.incomplete or v.final_winner is None:
            tally.incomplete_pairs += 1
            continue
        if v.final_winner == "Model_A":
            tally.wins_a += 1
        elif v.final_winner == "Model_B":
            tally.wins_b += 1
        elif v.final_winner == "Tie":
            tally.ties += 1
        else:  # "Tie (Position Inconsistency)"
            tally.inconsistent_pairs += 1

    tally.completed_pairs = tally.wins_a + tally.wins_b + tally.ties + tally.inconsistent_pairs
    tally.resolved_pairs = tally.wins_a + tally.wins_b
    if tally.completed_pairs:
        tally.flip_rate = tally.inconsistent_pairs / tally.completed_pairs
    if tally.resolved_pairs:
        tally.win_rate_a = tally.wins_a / tally.resolved_pairs
        tally.win_rate_b = tally.wins_b / tally.resolved_pairs
        tally.win_rate_b_ci = clopper_pearson_interval(tally.wins_b, tally.resolved_pairs)
    return tally


def decide_winner(
    tally: PairwiseTally,
    *,
    flip_rate_threshold: float,
    win_rate_threshold: float,
    min_completion_rate: float = 0.5,
    config_a_label: str = "Config A",
    config_b_label: str = "Config B",
) -> tuple[str, list[str]]:
    """Coverage gate, then validity gate, then the win rate. Nothing else."""
    trace: list[str] = []

    if tally.completed_pairs == 0:
        trace.append("No pair completed in both orders; there is nothing to decide.")
        return "No winner declared - no completed pairs", trace

    # Coverage gate, before anything else. Every downstream statistic is
    # conditioned on the pairs that survived, and the ones that died are not a
    # random sample -- rate limiting kills the largest prompts first. Worse, the
    # flip rate is computed over completed pairs only, so a run that lost 12 of
    # 15 reports a clean 0/3 and sails through the validity gate. The failure
    # mode that actually occurs is therefore the one the gate cannot see unless
    # coverage is checked first.
    completion_rate = tally.completed_pairs / tally.n_cases if tally.n_cases else 0.0
    trace.append(
        f"Step 0 coverage gate: completed {tally.completed_pairs}/{tally.n_cases} pairs "
        f"= {completion_rate:.1%} vs minimum {min_completion_rate:.0%} "
        f"({tally.incomplete_pairs} incomplete)."
    )
    if completion_rate < min_completion_rate:
        trace.append(
            "Gate FAILED - too few pairs completed for the remainder to represent the suite. "
            "Re-run the missing pairs before reading any comparison from this report."
        )
        return "Insufficient coverage - no winner declared", trace
    trace.append("Coverage gate passed.")

    flip = tally.flip_rate or 0.0
    trace.append(
        f"Step 1 validity gate: flip_rate = {tally.inconsistent_pairs}/{tally.completed_pairs} "
        f"= {flip:.3f} vs threshold {flip_rate_threshold:.2f}."
    )
    if flip > flip_rate_threshold:
        trace.append("Gate FAILED - short-circuit, no winner is declared.")
        return "Judge unreliable on this suite - no winner declared", trace
    trace.append("Gate passed.")

    if tally.resolved_pairs == 0:
        trace.append(
            "Step 2: resolved = wins_a + wins_b = 0 (every pair tied or was inconsistent)."
        )
        return "Inconclusive - no pair resolved to a winner", trace

    rate_b = tally.win_rate_b or 0.0
    rate_a = tally.win_rate_a or 0.0
    trace.append(
        f"Step 2 effect: resolved = wins_a + wins_b = {tally.wins_a} + {tally.wins_b} "
        f"= {tally.resolved_pairs} (excludes {tally.ties} ties and "
        f"{tally.inconsistent_pairs} position-inconsistent pairs). "
        f"win_rate_a = {rate_a:.3f}, win_rate_b = {rate_b:.3f}, threshold {win_rate_threshold:.2f}."
    )
    if rate_b > win_rate_threshold:
        return f"{config_b_label}", trace
    if rate_a > win_rate_threshold:
        return f"{config_a_label}", trace
    return "Inconclusive - no significant difference", trace


# --------------------------------------------------------------------------
# The reduction
# --------------------------------------------------------------------------


def aggregate(
    verdicts: Sequence[Verdict],
    *,
    run_id: str,
    generated_at: str,
    # The rubric's criterion order; used only to fix key order in the report so
    # two reports (e.g. the ablation's before/after pair) diff line by line.
    criteria: Sequence[str] = (),
    pass_threshold: int = 4,
    min_criterion_score: int = 3,
    flip_rate_threshold: float = 0.20,
    win_rate_threshold: float = 0.55,
    suite_id: str = "",
    suite_path: str = "",
    judge_model: str = "",
    judge_temperature: float = 0.0,
    mitigations: dict[str, bool] | None = None,
    resolved_families: dict[str, str] | None = None,
    cross_family_enforced: bool = True,
    config_a_label: str | None = None,
    config_b_label: str | None = None,
    price_per_million: tuple[float, float] | None = None,
    cost_note: str = "",
) -> SuiteReport:
    """Reduce verdicts to a report. Pure: same inputs, same output, always."""
    kinds = {v.kind for v in verdicts}
    if len(kinds) > 1:
        raise PipelineError(f"cannot aggregate a mixed verdict stream: {sorted(kinds)}")
    mode: Literal["pointwise", "pairwise"] = "pairwise" if kinds == {"pairwise"} else "pointwise"

    report = SuiteReport(
        run_id=run_id,
        generated_at=generated_at,
        mode=mode,
        suite_id=suite_id,
        suite_path=suite_path,
        judge_model=judge_model,
        judge_temperature=judge_temperature,
        mitigations=dict(mitigations or {}),
        resolved_families=dict(resolved_families or {}),
        cross_family_enforced=cross_family_enforced,
        n_cases=len(verdicts),
        decision_rule=DECISION_RULE_TEXT if mode == "pairwise" else None,
    )

    for v in verdicts:
        report.total_judge_calls += v.accounting.api_calls
        report.total_cached_calls += v.accounting.cached_calls
        report.total_attempts += v.accounting.attempts
        report.total_prompt_tokens += v.accounting.prompt_tokens
        report.total_completion_tokens += v.accounting.completion_tokens
        report.total_latency_ms += v.accounting.latency_ms
        if v.failure == "judge_parse_error":
            report.n_parse_errors += 1
            report.failed_case_ids.append(v.case_id)
        elif v.failure == "judge_call_failed":
            report.n_call_failures += 1
            report.failed_case_ids.append(v.case_id)

    if price_per_million is not None:
        in_price, out_price = price_per_million
        report.estimated_cost_usd = (
            report.total_prompt_tokens * in_price + report.total_completion_tokens * out_price
        ) / 1_000_000
    report.cost_note = cost_note

    if mode == "pointwise":
        pw = [v for v in verdicts if isinstance(v, PointwiseCaseResult)]
        scored = [v for v in pw if v.verdict is not None]
        report.n_scored = len(scored)
        report.pass_threshold = pass_threshold
        report.min_criterion_score = min_criterion_score
        if scored:
            passed = [
                compute_passed(
                    v.verdict,  # type: ignore[arg-type]
                    pass_threshold=pass_threshold,
                    min_criterion_score=min_criterion_score,
                )
                for v in scored
            ]
            report.n_passed = sum(passed)
            report.pass_rate = report.n_passed / len(scored)
            overall = [v.verdict.overall_score for v in scored]  # type: ignore[union-attr]
            report.mean_overall_score = sum(overall) / len(overall)
            report.mean_scores_by_criterion = mean_scores_by_criterion(scored, order=criteria)
            report.overall_score_distribution = score_distribution(overall)
            report.criterion_score_distributions = criterion_distributions(scored)
            # Verbosity, measured at suite scale and at zero extra API cost:
            # the lengths and the scores are already on disk.
            report.length_vs_score_spearman = spearman([v.output_length for v in scored], overall)
        else:
            report.pass_rate = None
    else:
        pr = [v for v in verdicts if isinstance(v, PairwiseCaseResult)]
        tally = tally_pairwise(pr)
        report.tally = tally
        report.n_scored = tally.completed_pairs
        report.config_a_label = config_a_label
        report.config_b_label = config_b_label
        winner, trace = decide_winner(
            tally,
            flip_rate_threshold=flip_rate_threshold,
            win_rate_threshold=win_rate_threshold,
            config_a_label=config_a_label or "Config A",
            config_b_label=config_b_label or "Config B",
        )
        report.winner = winner
        report.decision_trace = trace

    return report
