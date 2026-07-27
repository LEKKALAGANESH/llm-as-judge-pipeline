"""The `--mitigations off` ablation: the only source of "how biased before vs after".

Nothing else in the pipeline produces the "before" half. Mitigations built and
described, with no baseline to compare them against, is a Discussion written from
assertion; this module is what turns it into evidence.

Three of the five mitigations are ablatable and are switched off here: the
order-swap, the anti-verbosity rubric clause, and the few-shot anchors. Two are
not, and the report says so rather than implying an ablation that did not happen:

* **rationale-first** is the verdict schema's field order. Disabling it would
  mean shipping a second schema; its effect is measured by the sycophancy probes.
* **cross-family judging** is which models you call. Its "before" condition is
  the same-family judge run (`measure-self-enhancement`), not a flag.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from .aggregator import aggregate
from .audit_log import AuditLogger, utc_now_iso
from .datasets import ProbeSuite
from .llm_client import LLMClient
from .reports import COST_NOTE
from .runner import RunContext, run_pairwise_cases, run_pointwise_cases
from .schema import ComparisonCase, PairedTestResult, SuiteReport, TestCase
from .stats import holm_bonferroni, mcnemar_exact, wilcoxon_paired
from .suite_config import MitigationSettings, PipelineConfig, Rubric
from .validator import ProbeValidation, SelfEnhancementResult, run_adversarial_probes

ABLATABLE_PROBE_CATEGORIES = ("verbosity", "over_correction")


class BiasRow(BaseModel):
    model_config = ConfigDict(extra="forbid")

    bias: str
    mitigation: str
    ablatable: bool
    measurement: str
    before: dict[str, Any] = Field(default_factory=dict)
    after: dict[str, Any] = Field(default_factory=dict)
    delta: dict[str, Any] = Field(default_factory=dict)
    significance: PairedTestResult | None = Field(
        default=None,
        description="Paired test on the same cases under both conditions. A raw delta at "
        "n=15-18 cannot be distinguished from noise by eye, and this rubric line asks for "
        "empirical proof rather than a direction.",
    )
    note: str = ""


class BiasAblationReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str
    generated_at: str
    judge_model: str
    judge_temperature: float
    rows: list[BiasRow] = Field(default_factory=list)
    pointwise_with_mitigations: SuiteReport | None = None
    pointwise_without_mitigations: SuiteReport | None = None
    pairwise_with_order_swap: SuiteReport | None = None
    pairwise_without_order_swap: SuiteReport | None = None
    probes_with_clause: ProbeValidation | None = None
    probes_without_clause: ProbeValidation | None = None
    self_enhancement: SelfEnhancementResult | None = None
    caveats: list[str] = Field(default_factory=list)


def _ctx(
    config: PipelineConfig,
    rubric: Rubric,
    client: LLMClient,
    logger: AuditLogger,
    run_id: str,
    mitigations: MitigationSettings,
) -> RunContext:
    return RunContext(
        config=config,
        rubric=rubric,
        client=client,
        logger=logger,
        judge=config.judge,
        mitigations=mitigations,
        run_id=run_id,
    )


def _aggregate(
    verdicts: Sequence[Any],
    config: PipelineConfig,
    *,
    run_id: str,
    mitigations: MitigationSettings,
    suite_id: str,
    suite_path: str,
    criteria: Sequence[str] = (),
    config_a_label: str | None = None,
    config_b_label: str | None = None,
) -> SuiteReport:
    price = config.price_for(config.judge.model)
    return aggregate(
        verdicts,
        run_id=run_id,
        generated_at=utc_now_iso(),
        # Same key order in both conditions so the before/after pair diffs cleanly.
        criteria=criteria,
        pass_threshold=config.scoring.pass_threshold,
        min_criterion_score=config.scoring.min_criterion_score,
        flip_rate_threshold=config.comparison.flip_rate_threshold,
        win_rate_threshold=config.comparison.win_rate_threshold,
        suite_id=suite_id,
        suite_path=suite_path,
        judge_model=config.judge.model,
        judge_temperature=config.judge.temperature,
        mitigations=mitigations.as_dict(),
        resolved_families=config.resolved_families,
        cross_family_enforced=config.cross_family_enforced,
        config_a_label=config_a_label,
        config_b_label=config_b_label,
        price_per_million=(price.input, price.output),
        cost_note=COST_NOTE,
    )


def _dist(report: SuiteReport | None) -> dict[str, Any]:
    d = report.overall_score_distribution if report else None
    if d is None:
        return {}
    return {
        "std_dev": d.std_dev,
        "modal_value": d.modal_value,
        "modal_share": d.modal_share,
        "entropy_bits": d.entropy_bits,
        "share_scoring_4_or_5": d.top_two_share,
        "histogram": d.histogram,
    }


def _rho(report: SuiteReport | None) -> dict[str, Any]:
    c = report.length_vs_score_spearman if report else None
    if c is None:
        return {}
    return {"spearman_rho": c.rho, "p_value": c.p_value, "n": c.n, "reason": c.reason}


def _probe_rate(validation: ProbeValidation | None, category: str) -> dict[str, Any]:
    if validation is None:
        return {}
    for row in validation.by_category:
        if row.category == category:
            return {
                "pass_rate": row.pass_rate,
                "n_evaluated": row.n_evaluated,
                "n_passed": row.n_passed,
                "wilson_ci": ([row.wilson_ci.low, row.wilson_ci.high] if row.wilson_ci else None),
            }
    return {}


def _delta(before: Any, after: Any) -> Any:
    if isinstance(before, (int, float)) and isinstance(after, (int, float)):
        return after - before
    return None


def _paired_overall_scores(
    before: Sequence[Any], after: Sequence[Any]
) -> tuple[list[float], list[float]]:
    """Align two arms of the ablation on ``case_id``.

    Zipping by position would silently mis-pair whenever one arm drops a case to
    a parse error or a call failure -- and the arms are run separately, so they
    drop different cases. The pairing is the whole basis of the test.
    """

    def scored(results: Sequence[Any]) -> dict[str, float]:
        out: dict[str, float] = {}
        for r in results:
            verdict = getattr(r, "verdict", None)
            score = getattr(verdict, "overall_score", None)
            if score is not None:
                out[r.case_id] = float(score)
        return out

    b, a = scored(before), scored(after)
    common = sorted(set(b) & set(a))
    return [b[cid] for cid in common], [a[cid] for cid in common]


def _discordant_resolution_counts(
    without_swap: Sequence[Any], with_swap: Sequence[Any]
) -> tuple[int, int]:
    """(resolved only without swap, resolved only with swap), paired by case id.

    McNemar's two cells are the pairs where the arms *disagree*. Passing a
    concordant count instead silently tests "is the withdrawal rate 50%?",
    which is not the hypothesis and inverts the verdict: zero withdrawals then
    looks highly significant and near-total withdrawal looks like noise.
    """

    def resolved(results: Sequence[Any]) -> dict[str, bool]:
        out: dict[str, bool] = {}
        for r in results:
            winner = getattr(r, "final_winner", None)
            out[r.case_id] = bool(winner) and winner in {"Model_A", "Model_B"}
        return out

    off, on = resolved(without_swap), resolved(with_swap)
    only_off = sum(1 for cid in off.keys() & on.keys() if off[cid] and not on[cid])
    only_on = sum(1 for cid in off.keys() & on.keys() if on[cid] and not off[cid])
    return only_off, only_on


def _attach_significance(
    report: BiasAblationReport,
    paired_pointwise: tuple[list[float], list[float]],
    paired_position: tuple[int, int] | None = None,
) -> None:
    """Attach paired tests, then Holm-adjust across the family.

    One subtlety that is easy to get wrong: ``MitigationSettings.all_off()``
    flips the anchors *and* the anti-verbosity clause together, so a single
    ablation arm cannot attribute an overall-score shift to either one alone.
    The Wilcoxon on overall score is therefore computed **once** and reported as
    a joint test of the whole pointwise bundle. Running it per row would publish
    two byte-identical results as if they were independent findings, and would
    enter the same evidence into the Holm family twice -- which inflates the
    correction and can flip a real effect to non-significant.
    """
    before, after = paired_pointwise
    tests: list[PairedTestResult] = []

    joint: PairedTestResult | None = None
    if before and after:
        joint = wilcoxon_paired(
            before,
            after,
            label="overall score, all pointwise mitigations on/off (joint -- not attributable "
            "to anchors or the anti-verbosity clause separately)",
        )

    for row in report.rows:
        if row.bias == "score_clustering":
            # Carries the joint test; the verbosity row points at it rather than
            # re-running the identical comparison.
            row.significance = joint
        elif row.bias == "verbosity" and joint is not None:
            row.note += (
                " Significance for the overall-score shift is the joint pointwise test on the "
                "score_clustering row: this ablation switches anchors and the anti-verbosity "
                "clause together, so neither is separately identifiable from it. The padded-probe "
                "pass rate is the verbosity-specific measurement."
            )
        elif row.bias == "position" and paired_position is not None:
            only_off, only_on = paired_position
            row.significance = mcnemar_exact(
                only_on, only_off, label="pairs resolved with vs without order swap"
            )

    tests = [r.significance for r in report.rows if r.significance is not None]
    # De-duplicate by identity: the joint test is attached to one row only, but
    # this keeps the family honest if a future row reuses a result object.
    unique: list[PairedTestResult] = []
    for test in tests:
        if not any(test is seen for seen in unique):
            unique.append(test)
    holm_bonferroni(unique)


def run_bias_ablation(
    *,
    config: PipelineConfig,
    rubric: Rubric,
    client: LLMClient,
    logger: AuditLogger,
    run_id: str,
    pointwise_cases: Sequence[TestCase] = (),
    pointwise_suite_id: str = "",
    pointwise_suite_path: str = "",
    comparison_cases: Sequence[ComparisonCase] = (),
    comparison_suite_id: str = "",
    comparison_suite_path: str = "",
    config_a_label: str | None = None,
    config_b_label: str | None = None,
    probe_suite: ProbeSuite | None = None,
    self_enhancement: SelfEnhancementResult | None = None,
) -> BiasAblationReport:
    on = config.mitigations
    off = MitigationSettings.all_off()
    ctx_on = _ctx(config, rubric, client, logger, run_id, on)
    ctx_off = _ctx(config, rubric, client, logger, run_id, off)

    report = BiasAblationReport(
        run_id=run_id,
        generated_at=utc_now_iso(),
        judge_model=config.judge.model,
        judge_temperature=config.judge.temperature,
        self_enhancement=self_enhancement,
    )

    paired_pointwise: tuple[list[float], list[float]] = ([], [])
    paired_position: tuple[int, int] | None = None
    if pointwise_cases:
        with_v = run_pointwise_cases(pointwise_cases, ctx_on, stage="ablation-pointwise-on")
        without_v = run_pointwise_cases(pointwise_cases, ctx_off, stage="ablation-pointwise-off")
        paired_pointwise = _paired_overall_scores(without_v, with_v)
        report.pointwise_with_mitigations = _aggregate(
            with_v,
            config,
            run_id=run_id,
            mitigations=on,
            suite_id=pointwise_suite_id,
            suite_path=pointwise_suite_path,
            criteria=rubric.criterion_names,
        )
        report.pointwise_without_mitigations = _aggregate(
            without_v,
            config,
            run_id=run_id,
            mitigations=off,
            suite_id=pointwise_suite_id,
            suite_path=pointwise_suite_path,
            criteria=rubric.criterion_names,
        )

    if comparison_cases:
        swap_v = run_pairwise_cases(comparison_cases, ctx_on, stage="ablation-pairwise-on")
        noswap_v = run_pairwise_cases(comparison_cases, ctx_off, stage="ablation-pairwise-off")
        paired_position = _discordant_resolution_counts(noswap_v, swap_v)
        report.pairwise_with_order_swap = _aggregate(
            swap_v,
            config,
            run_id=run_id,
            mitigations=on,
            suite_id=comparison_suite_id,
            suite_path=comparison_suite_path,
            config_a_label=config_a_label,
            config_b_label=config_b_label,
        )
        report.pairwise_without_order_swap = _aggregate(
            noswap_v,
            config,
            run_id=run_id,
            mitigations=off,
            suite_id=comparison_suite_id,
            suite_path=comparison_suite_path,
            config_a_label=config_a_label,
            config_b_label=config_b_label,
        )

    if probe_suite is not None:
        subset = probe_suite.model_copy(
            update={
                "probes": [
                    p
                    for p in probe_suite.probes
                    if p.category in ABLATABLE_PROBE_CATEGORIES and p.kind == "relational_pair"
                ]
            }
        )
        if subset.probes:
            report.probes_with_clause = run_adversarial_probes(
                subset, ctx_on, stage="ablation-probe-on"
            )
            report.probes_without_clause = run_adversarial_probes(
                subset, ctx_off, stage="ablation-probe-off"
            )

    report.rows = _build_rows(report)
    _attach_significance(report, paired_pointwise, paired_position)
    report.caveats = [
        "Every number here comes from one run of a small suite with a stochastic judge. Treat "
        "the direction of each delta as the finding and the magnitude as indicative.",
        "rationale-first and cross-family judging are not flag-ablatable (see module docstring); "
        "their rows point at the artifacts that do measure them.",
        "Paired tests are reported per ablatable row, with Holm-Bonferroni adjustment across the "
        "family. At this sample size most effects will not survive adjustment; that is the "
        "honest reading, not a defect in the mitigation.",
    ]
    return report


def _build_rows(report: BiasAblationReport) -> list[BiasRow]:
    rows: list[BiasRow] = []

    # -- position ----------------------------------------------------------
    before_t = (
        report.pairwise_without_order_swap.tally if report.pairwise_without_order_swap else None
    )
    after_t = report.pairwise_with_order_swap.tally if report.pairwise_with_order_swap else None
    rows.append(
        BiasRow(
            bias="position",
            mitigation="run every pair in both orders; resolve disagreement to Tie (Position Inconsistency)",
            ablatable=True,
            measurement="flip rate = position-inconsistent pairs / completed pairs",
            before={
                "order_swap": False,
                "declared_winners": (before_t.wins_a + before_t.wins_b) if before_t else None,
                "completed_pairs": before_t.completed_pairs if before_t else None,
                "flip_rate": None,
                "note": "with a single order the judge always looks decisive; that is the point "
                "of the comparison, not a missing measurement",
            },
            after={
                "order_swap": True,
                "declared_winners": (after_t.wins_a + after_t.wins_b) if after_t else None,
                "completed_pairs": after_t.completed_pairs if after_t else None,
                "flip_rate": after_t.flip_rate if after_t else None,
                "inconsistent_pairs": after_t.inconsistent_pairs if after_t else None,
            },
            delta={
                "winners_withdrawn_as_unstable": (
                    (before_t.wins_a + before_t.wins_b) - (after_t.wins_a + after_t.wins_b)
                    if before_t and after_t
                    else None
                )
            },
            note="The raw flip rate is an UPPER BOUND on position bias; subtract the same-order "
            "repeat flip rate from judge_validation_report.json for the noise-corrected estimate.",
        )
    )

    # -- verbosity ---------------------------------------------------------
    before_rho = _rho(report.pointwise_without_mitigations)
    after_rho = _rho(report.pointwise_with_mitigations)
    before_probe = _probe_rate(report.probes_without_clause, "verbosity")
    after_probe = _probe_rate(report.probes_with_clause, "verbosity")
    rows.append(
        BiasRow(
            bias="verbosity",
            mitigation="anti-verbosity rubric clause + padded-answer probes + suite-scale length control",
            ablatable=True,
            measurement="Spearman rho between len(model_output) and overall_score across the "
            "suite (zero extra API calls), plus padded-vs-concise probe pass rate",
            before={**before_rho, "padded_probe": before_probe},
            after={**after_rho, "padded_probe": after_probe},
            delta={
                # _rho() emits "spearman_rho"; reading "rho" silently made the
                # only before/after number for verbosity null on every run.
                "spearman_rho": _delta(
                    before_rho.get("spearman_rho"), after_rho.get("spearman_rho")
                ),
                "padded_probe_pass_rate": _delta(
                    before_probe.get("pass_rate"), after_probe.get("pass_rate")
                ),
            },
            note="A positive rho means longer answers score higher. The clause should move rho "
            "toward zero; it should NOT drive it negative, which would mean the judge has been "
            "pushed into penalising legitimate thoroughness -- the over_correction probes exist "
            "to catch exactly that.",
        )
    )

    # -- self-enhancement --------------------------------------------------
    se = report.self_enhancement
    rows.append(
        BiasRow(
            bias="self_enhancement",
            mitigation="judge from a different model family than every candidate, enforced as a "
            "hard startup assertion on the model portion of the id",
            ablatable=False,
            measurement="win-rate delta between the cross-family judge and a same-family judge on "
            "identical content",
            before=(
                {
                    "judge": se.same_family_judge,
                    "judge_family": se.same_family_judge_family,
                    "win_rate_b": se.same_family_tally.win_rate_b,
                    "wins_a": se.same_family_tally.wins_a,
                    "wins_b": se.same_family_tally.wins_b,
                }
                if se
                else {"note": "run `measure-self-enhancement --allow-same-family` to populate"}
            ),
            after=(
                {
                    "judge": se.cross_family_judge,
                    "judge_family": se.cross_family_judge_family,
                    "win_rate_b": se.cross_family_tally.win_rate_b,
                    "wins_a": se.cross_family_tally.wins_a,
                    "wins_b": se.cross_family_tally.wins_b,
                }
                if se
                else {}
            ),
            delta={"win_rate_b_delta": se.win_rate_b_delta if se else None},
            note="Not a flag. The 'before' condition is a different judge, which is why this row "
            "comes from its own command. The two judges differ in capability as well as in "
            "family, so the delta is an upper bound on self-enhancement.",
        )
    )

    # -- sycophancy --------------------------------------------------------
    rows.append(
        BiasRow(
            bias="sycophancy_style",
            mitigation="rationale-before-score enforced by schema field order + confidently-wrong "
            "and terse-but-correct probes",
            ablatable=False,
            measurement="per-category adversarial probe pass rate with Wilson intervals "
            "(judge_validation_report.json -> probes.by_category)",
            before={
                "note": "Not flag-ablatable: rationale-first is the schema's field order under "
                "constrained decoding, so turning it off means shipping a second schema. The "
                "probe pass rate is the measurement."
            },
            after={},
            delta={},
            note="See judge_validation_report.json for the sycophancy and prompt_injection "
            "category pass rates.",
        )
    )

    # -- score clustering --------------------------------------------------
    before_d = _dist(report.pointwise_without_mitigations)
    after_d = _dist(report.pointwise_with_mitigations)
    rows.append(
        BiasRow(
            bias="score_clustering",
            mitigation="few-shot calibration anchors at 1/3/5 per criterion, loaded from rubric.yaml",
            ablatable=True,
            measurement="overall-score std dev, modal-value share, Shannon entropy over the five "
            "levels, and the share of cases scoring 4 or 5",
            before=before_d,
            after=after_d,
            delta={
                "std_dev": _delta(before_d.get("std_dev"), after_d.get("std_dev")),
                "modal_share": _delta(before_d.get("modal_share"), after_d.get("modal_share")),
                "entropy_bits": _delta(before_d.get("entropy_bits"), after_d.get("entropy_bits")),
                "share_scoring_4_or_5": _delta(
                    before_d.get("share_scoring_4_or_5"), after_d.get("share_scoring_4_or_5")
                ),
            },
            note="Anchors should raise std dev and entropy and lower modal share. 'We added "
            "anchors' is not a measurement; 'scores collapsed to 4-or-5 on X% of cases without "
            "anchors and Y% with' is.",
        )
    )
    return rows
