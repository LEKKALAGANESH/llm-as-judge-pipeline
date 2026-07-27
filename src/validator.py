"""Judge validation: the module that answers "can we trust this judge".

Three independent numbers, not one:

* **Agreement with a disclosed gold set** -- quadratic-weighted Cohen's kappa
  with a bootstrap CI, plus the companion statistics that make a low kappa
  interpretable rather than alarming.
* **Test-retest reliability** at T > 0 -- ICC(1,1) on the unrounded overall
  score as the primary number, with the pass/fail flip rate reported second and
  labelled as the threshold artifact it is.
* **Adversarial probes** -- pass rate per bias category with Wilson intervals,
  because a per-category rate over one probe is 0% or 100% and is not a
  measurement.

Plus two measurements that exist to keep the bias table honest: the same-order
repeat flip rate (which turns the raw position-flip rate into a noise-corrected
estimate) and the same-family judge run (which turns self-enhancement bias from
an asserted mitigation into a reported number).
"""

from __future__ import annotations

from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict, Field

from .aggregator import tally_pairwise
from .datasets import GoldLabelSet, LabelingMethod, Probe, ProbeSuite
from .runner import RunContext, run_pairwise_cases, run_pointwise_cases
from .schema import (
    ComparisonCase,
    Interval,
    PairedTestResult,
    PairwiseCaseResult,
    PairwiseTally,
    PointwiseCaseResult,
    TestCase,
    compute_passed,
)
from .stats import (
    ICCResult,
    KappaResult,
    calculate_kappa,
    icc_1_1,
    mcnemar_exact,
    spearman,
    wilson_interval,
)

# --------------------------------------------------------------------------
# Gold-set agreement
# --------------------------------------------------------------------------


class GoldValidation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    n_labelled: int
    n_judged: int
    n_failed: int
    labeling_method: LabelingMethod
    overall_kappa: KappaResult
    pooled_criterion_kappa: KappaResult | None = None
    intra_rater_kappa: KappaResult | None = None
    intra_rater_note: str = ""
    unit_of_analysis: str = ""


UNIT_OF_ANALYSIS_NOTE = (
    "Two units are reported. `overall_kappa` is one observation per case (n = number of gold "
    "cases) on the integer overall_score -- this is the headline number. "
    "`pooled_criterion_kappa` pools every (case, criterion) pair (n = cases x criteria); it has "
    "more observations but they are clustered within case, so its confidence interval is "
    "anti-conservative and it should be read as a secondary, descriptive figure only."
)


def validate_against_gold(
    cases: Sequence[TestCase],
    gold: GoldLabelSet,
    ctx: RunContext,
    *,
    bootstrap_resamples: int = 2000,
    seed: int = 20260726,
    labels: Sequence[int] = (1, 2, 3, 4, 5),
    stage: str = "gold",
) -> tuple[GoldValidation, list[PointwiseCaseResult]]:
    by_case = gold.by_case
    target = [c for c in cases if c.case_id in by_case]
    missing = sorted(set(by_case) - {c.case_id for c in target})
    if missing:
        raise ValueError(
            f"gold labels reference case ids that are not in the suite: {missing}. "
            "Refusing to compute an agreement number over a partially matched set."
        )

    results = [
        r
        for r in run_pointwise_cases(target, ctx, stage=stage)
        if isinstance(r, PointwiseCaseResult)
    ]
    scored = [r for r in results if r.verdict is not None]

    overall_pairs = [
        (r.case_id, by_case[r.case_id].overall_score, r.verdict.overall_score)  # type: ignore[union-attr]
        for r in scored
    ]
    overall = calculate_kappa(
        overall_pairs, labels=labels, bootstrap_resamples=bootstrap_resamples, seed=seed
    )

    criterion_pairs: list[tuple[str, int, int]] = []
    for r in scored:
        human = by_case[r.case_id].criterion_scores
        for name, cs in r.verdict.criteria_breakdown.items():  # type: ignore[union-attr]
            if name in human:
                criterion_pairs.append((f"{r.case_id}::{name}", human[name], cs.score))
    pooled = (
        calculate_kappa(
            criterion_pairs, labels=labels, bootstrap_resamples=bootstrap_resamples, seed=seed
        )
        if criterion_pairs
        else None
    )

    relabel_pairs = [
        (label.case_id, label.overall_score, label.overall_score_relabel)
        for label in gold.labels
        if label.overall_score_relabel is not None
    ]
    intra: KappaResult | None = None
    note = (
        "Not performed. A second, blind labelling pass after a washout would establish the "
        "CEILING on judge-human agreement: kappa 0.55 against a labeller whose own self-agreement "
        "is 0.60 is a completely different result from 0.55 against a labeller at 0.95, and "
        "without the second pass this design cannot tell those apart. The code path is live -- "
        "populate `overall_score_relabel` in the gold file and it is computed automatically."
    )
    if len(relabel_pairs) >= 2:
        intra = calculate_kappa(
            [(cid, a, b) for cid, a, b in relabel_pairs if b is not None],
            labels=labels,
            bootstrap_resamples=bootstrap_resamples,
            seed=seed,
        )
        note = "Second blind labelling pass present; this is the ceiling on judge-human agreement."

    validation = GoldValidation(
        n_labelled=len(gold.labels),
        n_judged=len(scored),
        n_failed=len(results) - len(scored),
        labeling_method=gold.labeling_method,
        overall_kappa=overall,
        pooled_criterion_kappa=pooled,
        intra_rater_kappa=intra,
        intra_rater_note=note,
        unit_of_analysis=UNIT_OF_ANALYSIS_NOTE,
    )
    return validation, results


# --------------------------------------------------------------------------
# Test-retest
# --------------------------------------------------------------------------


class TestRetestResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    runs: int
    temperature: float
    n_cases: int
    icc: ICCResult
    overall_score_flip_rate: float | None = Field(
        default=None,
        description="Share of cases whose integer overall_score was not identical across all runs.",
    )
    passed_flip_rate: float | None = Field(
        default=None,
        description="Secondary and explicitly threshold-dependent: a judge with tiny noise whose "
        "scores sit on the pass boundary shows a huge flip rate, and one with large noise far "
        "from the boundary shows zero.",
    )
    mean_within_case_sd: float | None = None
    sampling_warning: bool = False
    identical_rationale_share: float | None = None
    note: str = ""


def test_retest_consistency(
    cases: Sequence[TestCase],
    ctx: RunContext,
    *,
    runs: int = 5,
    pass_threshold: int = 4,
    min_criterion_score: int = 3,
    stage_prefix: str = "retest",
) -> TestRetestResult:
    """Run the same suite k times and measure the judge's own noise.

    The cache is bypassed for every call here. Without that, a disk cache keyed
    on the prompt would return byte-identical responses and manufacture a
    perfect 0% flip rate -- the exact meaningless "validation" number this
    protocol exists to avoid.
    """
    per_run: list[dict[str, PointwiseCaseResult]] = []
    for i in range(runs):
        run_ctx = RunContext(
            config=ctx.config,
            rubric=ctx.rubric,
            client=ctx.client,
            logger=ctx.logger,
            judge=ctx.judge,
            mitigations=ctx.mitigations,
            run_id=ctx.run_id,
            bypass_cache=True,
        )
        results = run_pointwise_cases(cases, run_ctx, stage=f"{stage_prefix}-{i + 1}")
        per_run.append(
            {r.case_id: r for r in results if isinstance(r, PointwiseCaseResult) and r.verdict}
        )

    common = set(per_run[0])
    for run in per_run[1:]:
        common &= set(run)
    ordered_ids = [c.case_id for c in cases if c.case_id in common]

    matrix: list[list[float]] = []
    score_flips = 0
    passed_flips = 0
    identical_rationales = 0
    for cid in ordered_ids:
        scores = [run[cid].verdict.overall_score for run in per_run]  # type: ignore[union-attr]
        matrix.append([float(s) for s in scores])
        if len(set(scores)) > 1:
            score_flips += 1
        passes = [
            compute_passed(
                run[cid].verdict,  # type: ignore[arg-type]
                pass_threshold=pass_threshold,
                min_criterion_score=min_criterion_score,
            )
            for run in per_run
        ]
        if len(set(passes)) > 1:
            passed_flips += 1
        rationales = {run[cid].verdict.overall_rationale for run in per_run}  # type: ignore[union-attr]
        if len(rationales) == 1:
            identical_rationales += 1

    n = len(ordered_ids)
    result = TestRetestResult(
        runs=runs,
        temperature=ctx.judge.temperature,
        n_cases=n,
        icc=icc_1_1(matrix) if matrix else ICCResult(n_subjects=0, n_raters=0, reason="no cases"),
    )
    if n:
        result.overall_score_flip_rate = score_flips / n
        result.passed_flip_rate = passed_flips / n
        result.identical_rationale_share = identical_rationales / n
        result.mean_within_case_sd = result.icc.mean_within_subject_sd

    # A *literal* 0.0 is the red flag, not a low number: modern serving stacks
    # are not bit-deterministic, so even T=0 normally produces a small non-zero
    # flip rate. Byte-identical rationales across every run prove that sampling
    # never happened -- i.e. the temperature setting did not take effect.
    if n and result.overall_score_flip_rate == 0.0:
        result.sampling_warning = True
        result.note = (
            "Flip rate is exactly 0.0 across all runs. "
            f"{identical_rationales}/{n} cases produced byte-identical rationales in every run. "
            "If that share is high, the temperature setting did not take effect and this is not "
            "a reliability result."
        )
    return result


# --------------------------------------------------------------------------
# Adversarial probes
# --------------------------------------------------------------------------


class ProbeOutcome(BaseModel):
    model_config = ConfigDict(extra="forbid")

    probe_id: str
    category: str
    kind: str
    passed: bool | None = Field(
        default=None, description="None when the probe could not be judged (excluded from rates)."
    )
    detail: str = ""
    measured: dict[str, float] = Field(default_factory=dict)


class ProbeCategoryResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    category: str
    n_probes: int
    n_evaluated: int
    n_passed: int
    pass_rate: float | None = None
    wilson_ci: Interval | None = None


class ProbeValidation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    n_probes: int
    n_evaluated: int
    n_passed: int
    overall_pass_rate: float | None = None
    by_category: list[ProbeCategoryResult] = Field(default_factory=list)
    outcomes: list[ProbeOutcome] = Field(default_factory=list)
    noise_floor_non_tie_rate: float | None = Field(
        default=None,
        description="Non-Tie rate over byte-identical output pairs. An assumption-free estimate "
        "of pure position + sampling bias -- the cleanest control in the design.",
    )


def _probe_case(probe: Probe, output: str, suffix: str = "") -> TestCase:
    return TestCase(
        case_id=f"{probe.probe_id}{suffix}",
        input=probe.input,
        system_prompt=probe.system_prompt,
        model_output=output,
        expected_output=probe.expected_output,
        tags=[f"probe:{probe.category}"],
    )


def _mean_of(result: PointwiseCaseResult, criteria: Sequence[str]) -> float | None:
    if result.verdict is None:
        return None
    scores = [
        result.verdict.criteria_breakdown[name].score
        for name in criteria
        if name in result.verdict.criteria_breakdown
    ]
    return sum(scores) / len(scores) if scores else None


def run_adversarial_probes(
    probes: ProbeSuite,
    ctx: RunContext,
    *,
    stage: str = "probe",
) -> ProbeValidation:
    outcomes: list[ProbeOutcome] = []

    pointwise_probes = [p for p in probes.probes if p.kind == "pointwise_threshold"]
    relational = [p for p in probes.probes if p.kind == "relational_pair"]
    identical = [p for p in probes.probes if p.kind == "identical_pair"]

    # --- pointwise threshold probes ---------------------------------------
    cases = [_probe_case(p, p.model_output or "") for p in pointwise_probes]
    results = {
        r.case_id: r
        for r in run_pointwise_cases(cases, ctx, stage=stage)
        if isinstance(r, PointwiseCaseResult)
    }
    for probe in pointwise_probes:
        res = results.get(probe.probe_id)
        if res is None or res.verdict is None:
            outcomes.append(
                ProbeOutcome(
                    probe_id=probe.probe_id,
                    category=probe.category,
                    kind=probe.kind,
                    passed=None,
                    detail=f"judge failed: {getattr(res, 'error', 'no result')}",
                )
            )
            continue
        checks: list[str] = []
        ok = True
        measured: dict[str, float] = {"overall_score": float(res.verdict.overall_score)}
        for assertion in probe.assertions:
            if assertion.criterion is None:
                actual = res.verdict.overall_score
                name = "overall_score"
            else:
                cs = res.verdict.criteria_breakdown.get(assertion.criterion)
                if cs is None:
                    ok = False
                    checks.append(f"{assertion.criterion} missing from verdict")
                    continue
                actual = cs.score
                name = assertion.criterion
            measured[name] = float(actual)
            if assertion.max_score is not None and actual > assertion.max_score:
                ok = False
                checks.append(f"{name}={actual} > max {assertion.max_score}")
            if assertion.min_score is not None and actual < assertion.min_score:
                ok = False
                checks.append(f"{name}={actual} < min {assertion.min_score}")
        outcomes.append(
            ProbeOutcome(
                probe_id=probe.probe_id,
                category=probe.category,
                kind=probe.kind,
                passed=ok,
                detail="; ".join(checks) if checks else "all pre-registered bounds met",
                measured=measured,
            )
        )

    # --- relational pair probes -------------------------------------------
    pair_cases: list[TestCase] = []
    for p in relational:
        pair_cases.append(_probe_case(p, p.baseline_output or "", "::baseline"))
        pair_cases.append(_probe_case(p, p.variant_output or "", "::variant"))
    pair_results = {
        r.case_id: r
        for r in run_pointwise_cases(pair_cases, ctx, stage=stage)
        if isinstance(r, PointwiseCaseResult)
    }
    for probe in relational:
        base = pair_results.get(f"{probe.probe_id}::baseline")
        var = pair_results.get(f"{probe.probe_id}::variant")
        base_mean = _mean_of(base, probe.scored_criteria) if base else None
        var_mean = _mean_of(var, probe.scored_criteria) if var else None
        if base_mean is None or var_mean is None:
            outcomes.append(
                ProbeOutcome(
                    probe_id=probe.probe_id,
                    category=probe.category,
                    kind=probe.kind,
                    passed=None,
                    detail="judge failed on one or both sides of the pair",
                )
            )
            continue
        delta = var_mean - base_mean
        if probe.relation == "variant_not_higher":
            ok = delta <= 0.0
            rule = "pre-registered: delta <= 0"
        else:
            ok = delta > 0.0
            rule = "pre-registered: delta > 0"
        outcomes.append(
            ProbeOutcome(
                probe_id=probe.probe_id,
                category=probe.category,
                kind=probe.kind,
                passed=ok,
                detail=(
                    f"{rule}; scored on {probe.scored_criteria}; "
                    f"baseline={base_mean:.2f} variant={var_mean:.2f} delta={delta:+.2f}"
                ),
                measured={"baseline": base_mean, "variant": var_mean, "delta": delta},
            )
        )

    # --- identical output pairs (noise floor) ------------------------------
    non_tie_verdicts = 0
    total_verdicts = 0
    identical_cases = [
        ComparisonCase(
            case_id=p.probe_id,
            input=p.input,
            system_prompt=p.system_prompt,
            output_a=p.output or "",
            output_b=p.output or "",
            config_a_label="identical",
            config_b_label="identical",
            tags=[f"probe:{p.category}"],
        )
        for p in identical
    ]
    if identical_cases:
        # The two outputs are byte-identical, so the forward and reverse prompts
        # are byte-identical too and the second call is a guaranteed cache hit.
        # Without the bypass this control observes one verdict, reports it as
        # two, and the remap turns it into a manufactured "position flip".
        noise_floor_ctx = RunContext(
            config=ctx.config,
            rubric=ctx.rubric,
            client=ctx.client,
            logger=ctx.logger,
            judge=ctx.judge,
            mitigations=ctx.mitigations,
            run_id=ctx.run_id,
            bypass_cache=True,
        )
        pair_out = {
            r.case_id: r
            for r in run_pairwise_cases(identical_cases, noise_floor_ctx, stage=stage)
            if isinstance(r, PairwiseCaseResult)
        }
        for probe in identical:
            res = pair_out.get(probe.probe_id)
            if res is None or res.incomplete:
                outcomes.append(
                    ProbeOutcome(
                        probe_id=probe.probe_id,
                        category=probe.category,
                        kind=probe.kind,
                        passed=None,
                        detail=f"judge failed: {getattr(res, 'error', 'no result')}",
                    )
                )
                continue
            winners = [w for w in (res.forward_winner, res.reverse_winner) if w is not None]
            total_verdicts += len(winners)
            non_tie = sum(1 for w in winners if w != "Tie")
            non_tie_verdicts += non_tie
            outcomes.append(
                ProbeOutcome(
                    probe_id=probe.probe_id,
                    category=probe.category,
                    kind=probe.kind,
                    passed=non_tie == 0,
                    detail=(
                        f"byte-identical outputs; forward={res.forward_winner} "
                        f"reverse(remapped)={res.reverse_winner}; a non-Tie here is pure "
                        "position or sampling noise"
                    ),
                    measured={"non_tie_verdicts": float(non_tie)},
                )
            )

    # --- aggregate ---------------------------------------------------------
    validation = ProbeValidation(
        n_probes=len(probes.probes),
        n_evaluated=sum(1 for o in outcomes if o.passed is not None),
        n_passed=sum(1 for o in outcomes if o.passed),
        outcomes=outcomes,
    )
    if validation.n_evaluated:
        validation.overall_pass_rate = validation.n_passed / validation.n_evaluated
    if total_verdicts:
        validation.noise_floor_non_tie_rate = non_tie_verdicts / total_verdicts

    for category in probes.categories:
        cat = [o for o in outcomes if o.category == category]
        evaluated = [o for o in cat if o.passed is not None]
        n_passed = sum(1 for o in evaluated if o.passed)
        validation.by_category.append(
            ProbeCategoryResult(
                category=category,
                n_probes=len(cat),
                n_evaluated=len(evaluated),
                n_passed=n_passed,
                pass_rate=(n_passed / len(evaluated)) if evaluated else None,
                wilson_ci=wilson_interval(n_passed, len(evaluated)) if evaluated else None,
            )
        )
    return validation


# --------------------------------------------------------------------------
# Position-bias noise floor and self-enhancement
# --------------------------------------------------------------------------


class NoiseFloorResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    n_pairs: int
    same_order_flip_rate: float | None = None
    swapped_flip_rate: float | None = None
    noise_corrected_position_bias: float | None = None
    note: str = (
        "A forward/reverse disagreement at T>0 could be pure sampling noise, so the raw "
        "order-swap flip rate is an UPPER BOUND on true position bias. Repeating the forward "
        "order without swapping isolates that noise; the difference is the corrected estimate. "
        "A negative corrected value means the measured position effect is within noise."
    )


def same_order_repeat_flip_rate(
    cases: Sequence[ComparisonCase],
    ctx: RunContext,
    baseline: Sequence[PairwiseCaseResult],
    *,
    swapped_flip_rate: float | None = None,
    stage: str = "noise-floor",
) -> NoiseFloorResult:
    """Re-judge the SAME order and compare, to separate noise from position bias."""
    repeat_ctx = RunContext(
        config=ctx.config,
        rubric=ctx.rubric,
        client=ctx.client,
        logger=ctx.logger,
        judge=ctx.judge,
        mitigations=ctx.mitigations.model_copy(update={"order_swap": False}),
        run_id=ctx.run_id,
        bypass_cache=True,
    )
    repeat = {
        r.case_id: r
        for r in run_pairwise_cases(cases, repeat_ctx, stage=stage)
        if isinstance(r, PairwiseCaseResult)
    }
    base = {r.case_id: r for r in baseline}
    compared = 0
    flips = 0
    for cid, first in base.items():
        second = repeat.get(cid)
        if second is None or first.forward_winner is None or second.forward_winner is None:
            continue
        compared += 1
        if first.forward_winner != second.forward_winner:
            flips += 1
    result = NoiseFloorResult(n_pairs=compared, swapped_flip_rate=swapped_flip_rate)
    if compared:
        result.same_order_flip_rate = flips / compared
        if swapped_flip_rate is not None:
            result.noise_corrected_position_bias = swapped_flip_rate - result.same_order_flip_rate
    return result


class SelfEnhancementResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cross_family_judge: str
    cross_family_judge_family: str
    same_family_judge: str
    same_family_judge_family: str
    candidate_family: str
    cross_family_tally: PairwiseTally
    same_family_tally: PairwiseTally
    win_rate_b_delta: float | None = None
    significance: PairedTestResult | None = Field(
        default=None,
        description="Exact McNemar over the pairs where the two judges disagree. The same cases "
        "are judged twice, so the design is paired and a bare win-rate delta cannot be "
        "distinguished from noise at this sample size.",
    )
    interpretation: str = ""


def measure_self_enhancement(
    cases: Sequence[ComparisonCase],
    cross_ctx: RunContext,
    same_ctx: RunContext,
    *,
    candidate_family: str,
    cross_results: Sequence[PairwiseCaseResult] | None = None,
) -> SelfEnhancementResult:
    """Judge the same pairwise suite cross-family and same-family.

    The win-rate delta is the self-enhancement estimate. Under a purely
    cross-family design this bias is structurally unmeasurable -- the invariant
    avoids it but yields no estimate of it -- and on a paid stack measuring it
    means buying a second judge. Here the second judge is free, so the weakest
    row of the bias matrix becomes a reported number.
    """
    cross = (
        list(cross_results)
        if cross_results
        else [
            r
            for r in run_pairwise_cases(cases, cross_ctx, stage="self-enh-cross")
            if isinstance(r, PairwiseCaseResult)
        ]
    )
    same = [
        r
        for r in run_pairwise_cases(cases, same_ctx, stage="self-enh-same")
        if isinstance(r, PairwiseCaseResult)
    ]
    cross_tally = tally_pairwise(cross)
    same_tally = tally_pairwise(same)
    delta = None
    if cross_tally.win_rate_b is not None and same_tally.win_rate_b is not None:
        delta = same_tally.win_rate_b - cross_tally.win_rate_b

    # McNemar over the discordant pairs: cases the two judges resolved to
    # *different* winners. Concordant pairs carry no information about which
    # judge favours which candidate, and counting them would understate the
    # effect by diluting the denominator with agreement.
    cross_by_id = {r.case_id: r.final_winner for r in cross if not r.incomplete}
    same_by_id = {r.case_id: r.final_winner for r in same if not r.incomplete}
    shifted_to_b = shifted_to_a = 0
    for case_id, cross_winner in cross_by_id.items():
        same_winner = same_by_id.get(case_id)
        if same_winner is None or same_winner == cross_winner:
            continue
        if same_winner == "Model_B":
            shifted_to_b += 1
        elif same_winner == "Model_A":
            shifted_to_a += 1
    significance = mcnemar_exact(shifted_to_a, shifted_to_b, label="same-family judge shift")

    interpretation = (
        "Both judges evaluated identical content. Any difference in win rate is attributable to "
        "the judge, and the same-family judge shares a lineage with both candidates, so a "
        "systematic shift is the self-enhancement signal. Note the confound to state plainly: "
        "the two judges differ in capability as well as in family, so this delta is an upper "
        "bound on self-enhancement rather than a clean estimate of it."
    )
    return SelfEnhancementResult(
        cross_family_judge=cross_ctx.judge.model,
        cross_family_judge_family=cross_ctx.judge.family,
        same_family_judge=same_ctx.judge.model,
        same_family_judge_family=same_ctx.judge.family,
        candidate_family=candidate_family,
        cross_family_tally=cross_tally,
        same_family_tally=same_tally,
        win_rate_b_delta=delta,
        significance=significance,
        interpretation=interpretation,
    )


# --------------------------------------------------------------------------
# Report assembly
# --------------------------------------------------------------------------


class LengthQualityBaseline(BaseModel):
    """How much length/score correlation is *real* on this suite.

    The suite-scale verbosity metric is Spearman(output length, judge score).
    Read alone it is not a bias estimate: if longer answers genuinely are better
    here, a perfectly calibrated judge scores strongly positive and looks biased.
    Correlating length against the HUMAN labels gives the baseline a fair judge
    should reproduce, so the corrected estimate is rho_judge - rho_gold.

    Costs nothing: both vectors are already on disk.
    """

    model_config = ConfigDict(extra="forbid")

    rho_gold: float | None = None
    p_value: float | None = None
    n: int = 0
    reason: str | None = None
    interpretation: str = ""


def length_quality_baseline(cases: Sequence[TestCase], gold: GoldLabelSet) -> LengthQualityBaseline:
    by_case = gold.by_case
    lengths: list[float] = []
    scores: list[float] = []
    for case in cases:
        label = by_case.get(case.case_id)
        if label is not None:
            lengths.append(float(len(case.model_output or "")))
            scores.append(float(label.overall_score))
    result = spearman(lengths, scores)
    if result.rho is None:
        return LengthQualityBaseline(n=result.n, reason=result.reason)
    return LengthQualityBaseline(
        rho_gold=result.rho,
        p_value=result.p_value,
        n=result.n,
        interpretation=(
            f"On this suite, length and *human-judged* quality correlate at rho={result.rho:+.3f}. "
            "A well-calibrated judge should therefore land near that value, not near zero. "
            "Subtract this from the suite's length_vs_score_spearman for the verbosity-bias "
            "estimate; reading the raw figure as bias would penalise the judge for tracking a "
            "real property of the data."
        ),
    )


class JudgeValidationReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str
    generated_at: str
    judge_model: str
    judge_temperature: float
    resolved_families: dict[str, str] = Field(default_factory=dict)
    cross_family_enforced: bool = True
    mitigations: dict[str, bool] = Field(default_factory=dict)

    gold: GoldValidation | None = None
    length_quality_baseline: LengthQualityBaseline | None = None
    test_retest: TestRetestResult | None = None
    test_retest_at_judge_temperature: TestRetestResult | None = None
    probes: ProbeValidation | None = None
    noise_floor: NoiseFloorResult | None = None
    self_enhancement: SelfEnhancementResult | None = None

    total_judge_calls: int = 0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    estimated_cost_usd: float = 0.0
    caveats: list[str] = Field(default_factory=list)
