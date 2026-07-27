"""Typer CLI -- the only user-facing entrypoint.

python main.py run-suite data/test_suites/general_qa.json
python main.py compare-configs
python main.py validate-judge
python main.py run-ablation
python main.py measure-self-enhancement --allow-same-family
python main.py replay <run_id> --stage run
python main.py smoke-test
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import typer
from dotenv import load_dotenv

from src.ablation import run_bias_ablation
from src.aggregator import aggregate
from src.audit_log import AuditLogger, new_run_id, utc_now_iso
from src.datasets import (
    load_comparison_suite,
    load_gold_labels,
    load_probes,
    load_test_suite,
)
from src.errors import PipelineError
from src.gating import evaluate_gate, format_gate
from src.llm_client import LLMClient
from src.reports import COST_NOTE, read_report, write_json
from src.reports import replay as replay_run
from src.runner import RunContext, run_pairwise_cases, run_pointwise_cases
from src.schema import PairwiseCaseResult, SuiteReport
from src.suite_config import (
    MitigationSettings,
    PipelineConfig,
    Rubric,
    check_api_key,
    load_config,
    load_rubric,
    warn_if_deterministic_test_retest,
)
from src.validator import (
    JudgeValidationReport,
    length_quality_baseline,
    run_adversarial_probes,
    same_order_repeat_flip_rate,
    test_retest_consistency,
    validate_against_gold,
)
from src.validator import (
    measure_self_enhancement as measure_self_enhancement_fn,
)

app = typer.Typer(
    add_completion=False,
    help="LLM-as-judge evaluation pipeline with measured bias mitigation and judge validation.",
    no_args_is_help=True,
)

DEFAULT_CONFIG = "config/suite_config.yaml"
DEFAULT_RUBRIC = "config/rubric.yaml"
DEFAULT_SUITE = "data/test_suites/general_qa.json"
DEFAULT_COMPARISON = "data/test_suites/comparison_qa.json"
NOISE_FLOOR_FILE = "position_bias_noise_floor.json"
SELF_ENHANCEMENT_FILE = "self_enhancement_report.json"

#: Both configured models reason before answering, spending 250-350 completion
#: tokens before the first visible one. The smoke test's budget has to clear
#: that floor or it proves the providers are broken when they are fine.
SMOKE_TEST_MAX_TOKENS = 512


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def _echo(msg: str = "") -> None:
    typer.echo(msg)


def _fmt(value: float | None, spec: str = ".3f", missing: str = "n/a") -> str:
    """Format a number that may legitimately be None (undefined statistic)."""
    return missing if value is None else format(value, spec)


def _result_path(config: PipelineConfig, filename: str, override: str | None = None) -> Path:
    """Every artifact lands under run.results_dir unless --out says otherwise, so
    a test (or a second workspace) can redirect the whole set with one setting."""
    return Path(override) if override else Path(config.run.results_dir) / filename


def _fail(exc: Exception) -> None:
    typer.secho(f"\n{type(exc).__name__}: {exc}\n", fg=typer.colors.RED, err=True)
    raise typer.Exit(code=1)


def _setup(
    config_path: str,
    rubric_path: str,
    *,
    allow_same_family: bool = False,
    max_calls: int | None = None,
    max_cost: float | None = None,
    mitigations_off: bool = False,
    no_cache: bool = False,
    run_prefix: str = "run",
    run_id: str | None = None,
) -> tuple[PipelineConfig, Rubric, LLMClient, AuditLogger, str]:
    load_dotenv()
    config = load_config(config_path, allow_same_family=allow_same_family)
    rubric = load_rubric(rubric_path)
    if max_calls is not None:
        config.run.max_calls_per_run = max_calls
    if max_cost is not None:
        config.run.max_cost_usd = max_cost
    if mitigations_off:
        config.mitigations = MitigationSettings.all_off()
    rid = run_id or new_run_id(run_prefix)
    client = LLMClient(
        cache_path=None if no_cache else config.run.cache_path,
        enable_cache=not no_cache,
        max_calls=config.run.max_calls_per_run,
        max_cost_usd=config.run.max_cost_usd,
        price_lookup=lambda m: (config.price_for(m).input, config.price_for(m).output),
        tokens_per_minute=config.run.tokens_per_minute,
    )
    logger = AuditLogger(config.run.logs_dir, run_id=rid)
    return config, rubric, client, logger, rid


def _families_banner(config: PipelineConfig) -> None:
    fams = ", ".join(f"{k}={v}" for k, v in config.resolved_families.items())
    state = "ENFORCED" if config.cross_family_enforced else "WAIVED (--allow-same-family)"
    _echo(f"  cross-family invariant: {state}  [{fams}]")


def _print_pointwise(report: SuiteReport) -> None:
    _echo("")
    _echo(f"  suite            : {report.suite_id or report.suite_path}")
    _echo(f"  judge            : {report.judge_model} @ T={report.judge_temperature}")
    _echo(f"  mitigations      : {report.mitigations}")
    _echo(f"  cases            : {report.n_cases} ({report.n_scored} scored)")
    if report.n_parse_errors or report.n_call_failures:
        _echo(
            f"  excluded         : {report.n_parse_errors} parse errors, "
            f"{report.n_call_failures} call failures -> {report.failed_case_ids}"
        )
    if report.pass_rate is not None:
        _echo(
            f"  pass rate        : {report.pass_rate:.1%} ({report.n_passed}/{report.n_scored}) "
            f"[overall >= {report.pass_threshold} AND min criterion >= {report.min_criterion_score}]"
        )
    if report.mean_overall_score is not None:
        _echo(f"  mean overall     : {report.mean_overall_score:.2f}")
    for name, val in sorted(report.mean_scores_by_criterion.items()):
        _echo(f"    {name:<22}: {val:.2f}")
    d = report.overall_score_distribution
    if d:
        _echo(
            f"  score spread     : sd={d.std_dev:.2f} modal={d.modal_value} "
            f"({d.modal_share:.0%}) entropy={d.entropy_bits:.2f} bits, "
            f"4-or-5 share={d.top_two_share:.0%}"
        )
    c = report.length_vs_score_spearman
    if c:
        if c.rho is None:
            _echo(f"  length vs score  : not computed ({c.reason})")
        else:
            _echo(
                f"  length vs score  : Spearman rho={c.rho:+.3f} (p={c.p_value:.3f}, n={c.n}) "
                "-- positive means longer answers scored higher"
            )
    _print_cost(report)


def _print_pairwise(report: SuiteReport) -> None:
    t = report.tally
    _echo("")
    _echo(f"  judge            : {report.judge_model} @ T={report.judge_temperature}")
    _echo(f"  order swap       : {report.mitigations.get('order_swap')}")
    if t:
        _echo(f"  pairs            : {t.n_cases} ({t.completed_pairs} completed in both orders)")
        _echo(
            f"  wins             : A[{report.config_a_label}]={t.wins_a}  "
            f"B[{report.config_b_label}]={t.wins_b}  ties={t.ties}"
        )
        _echo(f"  position-inconsistent: {t.inconsistent_pairs}   incomplete: {t.incomplete_pairs}")
        if t.flip_rate is not None:
            _echo(f"  flip rate        : {t.flip_rate:.1%}")
        if t.win_rate_b is not None:
            ci = t.win_rate_b_ci
            span = f" [95% CI {ci.low:.1%}-{ci.high:.1%}]" if ci else ""
            _echo(f"  win rate (B)     : {t.win_rate_b:.1%} of {t.resolved_pairs} resolved{span}")
    _echo("")
    _echo(f"  WINNER: {report.winner}")
    for line in report.decision_trace:
        _echo(f"    - {line}")
    _print_cost(report)


def _print_cost(report: SuiteReport) -> None:
    _echo("")
    _echo(
        f"  calls            : {report.total_judge_calls} api + {report.total_cached_calls} cached "
        f"({report.total_attempts} attempts incl. repairs)"
    )
    _echo(
        f"  tokens           : {report.total_prompt_tokens} in / "
        f"{report.total_completion_tokens} out"
    )
    _echo(f"  latency          : {report.total_latency_ms / 1000:.1f}s total judge wall clock")
    _echo(f"  paid-tier equiv  : ${report.estimated_cost_usd:.4f} (actual spend $0.00, free tier)")


# --------------------------------------------------------------------------
# commands
# --------------------------------------------------------------------------


@app.command()
def run_suite(
    suite: str = typer.Argument(DEFAULT_SUITE, help="Path to a JSON test suite."),
    config_path: str = typer.Option(DEFAULT_CONFIG, "--config"),
    rubric_path: str = typer.Option(DEFAULT_RUBRIC, "--rubric"),
    sample_size: int | None = typer.Option(
        None, "--sample-size", help="Judge only the first N cases (fast development iteration)."
    ),
    mitigations: str = typer.Option(
        "on", "--mitigations", help="'on' or 'off'. 'off' runs the ablation baseline."
    ),
    resume: str | None = typer.Option(
        None, "--resume", help="Resume a run id, reusing verdicts already in logs/verdicts.jsonl."
    ),
    max_calls: int | None = typer.Option(None, "--max-calls-per-run"),
    max_cost: float | None = typer.Option(None, "--max-cost-usd"),
    no_cache: bool = typer.Option(False, "--no-cache"),
    out: str | None = typer.Option(None, "--out"),
) -> None:
    """Judge a pointwise suite and write results/suite_report.json."""
    try:
        config, rubric, client, logger, rid = _setup(
            config_path,
            rubric_path,
            mitigations_off=(mitigations.lower() == "off"),
            max_calls=max_calls,
            max_cost=max_cost,
            no_cache=no_cache,
            run_id=resume,
        )
        check_api_key(config.judge.model)
        loaded = load_test_suite(suite)
        cases = loaded.cases[:sample_size] if sample_size else loaded.cases

        _echo(f"run {rid}: {len(cases)} case(s) from {suite}")
        _families_banner(config)

        ctx = RunContext.build(config, rubric, client=client, logger=logger, run_id=rid)
        resumed = AuditLogger.read_verdicts(config.run.logs_dir, rid, stage="run") if resume else []
        if resumed:
            _echo(f"  resuming: {len(resumed)} verdict(s) already on disk")
        verdicts = run_pointwise_cases(cases, ctx, stage="run", resume_from=resumed)

        price = config.price_for(config.judge.model)
        report = aggregate(
            verdicts,
            run_id=rid,
            generated_at=utc_now_iso(),
            criteria=rubric.criterion_names,
            pass_threshold=config.scoring.pass_threshold,
            min_criterion_score=config.scoring.min_criterion_score,
            suite_id=loaded.suite_id,
            suite_path=suite,
            judge_model=config.judge.model,
            judge_temperature=config.judge.temperature,
            mitigations=config.mitigations.as_dict(),
            resolved_families=config.resolved_families,
            cross_family_enforced=config.cross_family_enforced,
            price_per_million=(price.input, price.output),
            cost_note=COST_NOTE,
        )
        _print_pointwise(report)
        path = write_json(_result_path(config, "suite_report.json", out), report)
        _echo("")
        _echo(f"  wrote {path}")
        _echo(f"  replay with: python main.py replay {rid} --stage run --report {path}")
    except PipelineError as exc:
        _fail(exc)


@app.command()
def compare_configs(
    suite: str = typer.Argument(DEFAULT_COMPARISON, help="Path to a comparison (A/B) suite."),
    config_path: str = typer.Option(DEFAULT_CONFIG, "--config"),
    rubric_path: str = typer.Option(DEFAULT_RUBRIC, "--rubric"),
    sample_size: int | None = typer.Option(None, "--sample-size"),
    mitigations: str = typer.Option("on", "--mitigations"),
    noise_floor: bool = typer.Option(
        True,
        "--noise-floor/--no-noise-floor",
        help="Re-judge each pair in the SAME order to separate sampling noise from position bias.",
    ),
    resume: str | None = typer.Option(None, "--resume"),
    max_calls: int | None = typer.Option(None, "--max-calls-per-run"),
    max_cost: float | None = typer.Option(None, "--max-cost-usd"),
    no_cache: bool = typer.Option(False, "--no-cache"),
    out: str | None = typer.Option(None, "--out"),
) -> None:
    """Run the pairwise A/B with order swap and declare a winner by the stated rule."""
    try:
        config, rubric, client, logger, rid = _setup(
            config_path,
            rubric_path,
            mitigations_off=(mitigations.lower() == "off"),
            max_calls=max_calls,
            max_cost=max_cost,
            no_cache=no_cache,
            run_id=resume,
        )
        check_api_key(config.judge.model)
        loaded = load_comparison_suite(suite)
        cases = loaded.cases[:sample_size] if sample_size else loaded.cases

        _echo(f"run {rid}: {len(cases)} pair(s) from {suite}")
        _echo(
            f"  A = {loaded.config_a_label}   B = {loaded.config_b_label}   "
            f"({2 * len(cases)} judge calls with order swap)"
        )
        _families_banner(config)

        ctx = RunContext.build(config, rubric, client=client, logger=logger, run_id=rid)
        resumed = (
            AuditLogger.read_verdicts(config.run.logs_dir, rid, stage="compare") if resume else []
        )
        verdicts = run_pairwise_cases(cases, ctx, stage="compare", resume_from=resumed)

        price = config.price_for(config.judge.model)
        report = aggregate(
            verdicts,
            run_id=rid,
            generated_at=utc_now_iso(),
            flip_rate_threshold=config.comparison.flip_rate_threshold,
            win_rate_threshold=config.comparison.win_rate_threshold,
            suite_id=loaded.suite_id,
            suite_path=suite,
            judge_model=config.judge.model,
            judge_temperature=config.judge.temperature,
            mitigations=config.mitigations.as_dict(),
            resolved_families=config.resolved_families,
            cross_family_enforced=config.cross_family_enforced,
            config_a_label=loaded.config_a_label,
            config_b_label=loaded.config_b_label,
            price_per_million=(price.input, price.output),
            cost_note=COST_NOTE,
        )
        _print_pairwise(report)
        path = write_json(_result_path(config, "ab_comparison_report.json", out), report)
        _echo("")
        _echo(f"  wrote {path}")

        if noise_floor:
            _echo("")
            _echo("  measuring the noise floor (same-order repeat, cache bypassed)...")
            nf = same_order_repeat_flip_rate(
                cases,
                ctx,
                [v for v in verdicts if isinstance(v, PairwiseCaseResult)],
                swapped_flip_rate=report.tally.flip_rate if report.tally else None,
            )
            nf_path = write_json(_result_path(config, NOISE_FLOOR_FILE), nf)
            _echo(f"  same-order repeat flip rate : {_fmt(nf.same_order_flip_rate, '.1%')}")
            _echo(f"  order-swapped flip rate     : {_fmt(nf.swapped_flip_rate, '.1%')}")
            _echo(
                f"  noise-corrected position bias: {_fmt(nf.noise_corrected_position_bias, '+.1%')}"
            )
            _echo(f"  wrote {nf_path}")
        _echo(f"  replay with: python main.py replay {rid} --stage compare --report {path}")
    except PipelineError as exc:
        _fail(exc)


@app.command()
def validate_judge(
    suite: str = typer.Option(DEFAULT_SUITE, "--suite"),
    config_path: str = typer.Option(DEFAULT_CONFIG, "--config"),
    rubric_path: str = typer.Option(DEFAULT_RUBRIC, "--rubric"),
    gold: str | None = typer.Option(None, "--gold"),
    probes: str | None = typer.Option(None, "--probes"),
    retest_runs: int | None = typer.Option(None, "--retest-runs"),
    retest_temperature: float | None = typer.Option(None, "--retest-temperature"),
    retest_sample: int = typer.Option(
        8, "--retest-sample", help="Cases used for test-retest (k runs each; keeps quota sane)."
    ),
    acknowledge_t0: bool = typer.Option(
        False, "--acknowledge-t0", help="Proceed with test-retest at T=0 and record the caveat."
    ),
    skip_retest: bool = typer.Option(False, "--skip-retest"),
    skip_probes: bool = typer.Option(False, "--skip-probes"),
    max_calls: int | None = typer.Option(None, "--max-calls-per-run"),
    max_cost: float | None = typer.Option(None, "--max-cost-usd"),
    out: str | None = typer.Option(None, "--out"),
) -> None:
    """Kappa vs. gold labels, test-retest at T>0, and adversarial probe pass rates."""
    try:
        config, rubric, client, logger, rid = _setup(
            config_path, rubric_path, max_calls=max_calls, max_cost=max_cost, run_prefix="validate"
        )
        check_api_key(config.judge.model)
        gold_path = gold or config.validation.gold_labels_path
        probes_path = probes or config.validation.probes_path
        k = retest_runs or config.validation.test_retest_runs
        t_retest = (
            retest_temperature
            if retest_temperature is not None
            else config.validation.test_retest_temperature
        )

        caveats: list[str] = []
        warn = warn_if_deterministic_test_retest(t_retest, acknowledged=acknowledge_t0)
        if warn:
            typer.secho(warn, fg=typer.colors.YELLOW, err=True)
            caveats.append(warn)

        loaded = load_test_suite(suite)
        gold_set = load_gold_labels(gold_path)
        _echo(f"run {rid}: judge validation")
        _families_banner(config)

        report = JudgeValidationReport(
            run_id=rid,
            generated_at=utc_now_iso(),
            judge_model=config.judge.model,
            judge_temperature=config.judge.temperature,
            resolved_families=config.resolved_families,
            cross_family_enforced=config.cross_family_enforced,
            mitigations=config.mitigations.as_dict(),
        )

        ctx = RunContext.build(config, rubric, client=client, logger=logger, run_id=rid)

        _echo("")
        _echo(f"[1/3] gold-set agreement ({len(gold_set.labels)} hand-labelled cases)")
        gold_result, _ = validate_against_gold(
            loaded.cases,
            gold_set,
            ctx,
            bootstrap_resamples=config.validation.bootstrap_resamples,
            seed=config.validation.bootstrap_seed,
            labels=config.validation.kappa_labels,
        )
        report.gold = gold_result
        kap = gold_result.overall_kappa
        if kap.kappa_quadratic is None:
            _echo(f"  kappa_w: null -- {kap.reason}")
        else:
            ci = kap.kappa_ci
            span = f" [95% CI {ci.low:+.2f}, {ci.high:+.2f}]" if ci else ""
            _echo(f"  kappa_w (quadratic): {kap.kappa_quadratic:+.3f}{span}  n={kap.n}")
        _echo(f"  exact agreement    : {(kap.percent_agreement or 0):.1%}")
        _echo(f"  within-1 agreement : {(kap.within_one_agreement or 0):.1%}")
        if kap.spearman and kap.spearman.rho is not None:
            _echo(f"  Spearman rho       : {kap.spearman.rho:+.3f}")
        _echo(f"  reading            : {kap.interpretation}")

        # Zero API cost: correlates output length against the HUMAN labels, so
        # the suite's length_vs_score_spearman can be read as bias rather than
        # as bias plus whatever real length/quality relationship the data has.
        report.length_quality_baseline = length_quality_baseline(loaded.cases, gold_set)
        lqb = report.length_quality_baseline
        if lqb.rho_gold is not None:
            _echo(
                f"  length vs gold     : rho={lqb.rho_gold:+.3f} (p={lqb.p_value:.3f}, n={lqb.n}) "
                "-- the baseline a fair judge should reproduce, not 0"
            )

        if not skip_retest:
            _echo("")
            _echo(f"[2/3] test-retest, k={k} at T={t_retest}")
            retest_ctx = RunContext.build(
                config,
                rubric,
                judge=config.judge.with_temperature(t_retest),
                client=client,
                logger=logger,
                run_id=rid,
            )
            report.test_retest = test_retest_consistency(
                loaded.cases[:retest_sample],
                retest_ctx,
                runs=k,
                pass_threshold=config.scoring.pass_threshold,
                min_criterion_score=config.scoring.min_criterion_score,
            )
            tr = report.test_retest
            _echo(
                f"  ICC(1,1)               : {_fmt(tr.icc.icc_1_1)}"
                f"   mean within-case SD: {_fmt(tr.mean_within_case_sd)}"
            )
            if tr.icc.reason:
                _echo(f"    ({tr.icc.reason})")
            _echo(f"  overall-score flip rate: {_fmt(tr.overall_score_flip_rate, '.1%')}")
            _echo(
                f"  pass/fail flip rate    : {_fmt(tr.passed_flip_rate, '.1%')} "
                "(threshold artifact - secondary)"
            )
            if tr.sampling_warning:
                typer.secho(f"  WARNING: {tr.note}", fg=typer.colors.YELLOW)
                caveats.append(tr.note)

            if (
                config.validation.test_retest_also_at_judge_temperature
                and t_retest != config.judge.temperature
            ):
                _echo(f"  also at the production judge temperature T={config.judge.temperature}")
                report.test_retest_at_judge_temperature = test_retest_consistency(
                    loaded.cases[:retest_sample],
                    RunContext.build(config, rubric, client=client, logger=logger, run_id=rid),
                    runs=min(k, 3),
                    pass_threshold=config.scoring.pass_threshold,
                    min_criterion_score=config.scoring.min_criterion_score,
                    stage_prefix="retest-prod",
                )

        if not skip_probes:
            _echo("")
            probe_suite = load_probes(probes_path)
            _echo(f"[3/3] adversarial probes ({len(probe_suite.probes)} probes)")
            report.probes = run_adversarial_probes(probe_suite, ctx)
            for row in report.probes.by_category:
                ci = row.wilson_ci
                span = f" [95% CI {ci.low:.0%}-{ci.high:.0%}]" if ci else ""
                rate = "n/a" if row.pass_rate is None else f"{row.pass_rate:.0%}"
                _echo(f"  {row.category:<22}: {rate} ({row.n_passed}/{row.n_evaluated}){span}")
            if report.probes.noise_floor_non_tie_rate is not None:
                _echo(
                    f"  identical-pair non-Tie rate: "
                    f"{report.probes.noise_floor_non_tie_rate:.1%} (pure position/sampling noise)"
                )

        nf_path = _result_path(config, NOISE_FLOOR_FILE)
        if nf_path.is_file():
            from src.validator import NoiseFloorResult

            report.noise_floor = NoiseFloorResult.model_validate_json(
                nf_path.read_text(encoding="utf-8")
            )
        else:
            caveats.append(
                "Noise-corrected position bias not available: run `compare-configs --noise-floor` "
                "first, which writes results/position_bias_noise_floor.json."
            )

        se_path = _result_path(config, SELF_ENHANCEMENT_FILE)
        if se_path.is_file():
            from src.validator import SelfEnhancementResult

            report.self_enhancement = SelfEnhancementResult.model_validate_json(
                se_path.read_text(encoding="utf-8")
            )
        else:
            caveats.append(
                "Self-enhancement delta not available: run `measure-self-enhancement "
                "--allow-same-family`."
            )

        report.total_judge_calls = client.api_calls
        report.total_prompt_tokens = client.prompt_tokens
        report.total_completion_tokens = client.completion_tokens
        report.estimated_cost_usd = client.cost_usd
        report.caveats = caveats
        path = write_json(_result_path(config, "judge_validation_report.json", out), report)
        _echo("")
        _echo(f"  wrote {path}")
    except (PipelineError, ValueError) as exc:
        _fail(exc)


@app.command()
def measure_self_enhancement(
    suite: str = typer.Argument(DEFAULT_COMPARISON),
    config_path: str = typer.Option(DEFAULT_CONFIG, "--config"),
    rubric_path: str = typer.Option(DEFAULT_RUBRIC, "--rubric"),
    allow_same_family: bool = typer.Option(
        False,
        "--allow-same-family",
        help="REQUIRED. This run deliberately violates the cross-family invariant.",
    ),
    sample_size: int | None = typer.Option(None, "--sample-size"),
    max_calls: int | None = typer.Option(None, "--max-calls-per-run"),
    out: str | None = typer.Option(None, "--out"),
) -> None:
    """Judge the same pairs cross-family and same-family; the win-rate delta is the estimate."""
    try:
        if not allow_same_family:
            raise PipelineError(
                "This command judges candidates with a model from their own family, which "
                "violates the self-enhancement invariant on purpose. Pass --allow-same-family to "
                "acknowledge that, so the invariant stays a hard assertion everywhere else."
            )
        config, rubric, client, logger, rid = _setup(
            config_path,
            rubric_path,
            allow_same_family=True,
            max_calls=max_calls,
            run_prefix="self-enh",
        )
        probe_judge = config.self_enhancement_probe_judge
        if probe_judge is None:
            raise PipelineError("suite_config.yaml has no `self_enhancement_probe_judge` block.")
        check_api_key(config.judge.model)
        check_api_key(probe_judge.model)

        loaded = load_comparison_suite(suite)
        cases = loaded.cases[:sample_size] if sample_size else loaded.cases
        _echo(f"run {rid}: self-enhancement measurement over {len(cases)} pair(s)")
        _echo(f"  cross-family judge : {config.judge.model} ({config.judge.family})")
        _echo(f"  same-family judge  : {probe_judge.model} ({probe_judge.family})")

        cross_ctx = RunContext.build(config, rubric, client=client, logger=logger, run_id=rid)
        same_ctx = RunContext.build(
            config, rubric, judge=probe_judge, client=client, logger=logger, run_id=rid
        )
        result = measure_self_enhancement_fn(
            cases,
            cross_ctx,
            same_ctx,
            candidate_family=config.generator_family or "unknown",
        )
        _echo("")
        _echo(f"  cross-family win rate (B): {_fmt(result.cross_family_tally.win_rate_b, '.1%')}")
        _echo(f"  same-family  win rate (B): {_fmt(result.same_family_tally.win_rate_b, '.1%')}")
        _echo(f"  delta (self-enhancement estimate): {_fmt(result.win_rate_b_delta, '+.1%')}")
        _echo(f"  {result.interpretation}")
        path = write_json(_result_path(config, SELF_ENHANCEMENT_FILE, out), result)
        _echo(f"  wrote {path}")
    except PipelineError as exc:
        _fail(exc)


@app.command()
def run_ablation(
    suite: str = typer.Option(DEFAULT_SUITE, "--suite"),
    comparison: str | None = typer.Option(DEFAULT_COMPARISON, "--comparison"),
    config_path: str = typer.Option(DEFAULT_CONFIG, "--config"),
    rubric_path: str = typer.Option(DEFAULT_RUBRIC, "--rubric"),
    probes: str | None = typer.Option(None, "--probes"),
    sample_size: int | None = typer.Option(None, "--sample-size"),
    skip_pairwise: bool = typer.Option(False, "--skip-pairwise"),
    skip_probes: bool = typer.Option(False, "--skip-probes"),
    max_calls: int | None = typer.Option(None, "--max-calls-per-run"),
    out: str | None = typer.Option(None, "--out"),
) -> None:
    """Run every ablatable mitigation on and off; emit the before/after bias table."""
    try:
        config, rubric, client, logger, rid = _setup(
            config_path, rubric_path, max_calls=max_calls, run_prefix="ablation"
        )
        check_api_key(config.judge.model)
        loaded = load_test_suite(suite)
        cases = loaded.cases[:sample_size] if sample_size else loaded.cases

        comp_cases: list = []
        comp_id = comp_path = ""
        a_label = b_label = None
        if comparison and not skip_pairwise:
            comp = load_comparison_suite(comparison)
            comp_cases = comp.cases[:sample_size] if sample_size else comp.cases
            comp_id, comp_path = comp.suite_id, comparison
            a_label, b_label = comp.config_a_label, comp.config_b_label

        probe_suite = None
        if not skip_probes:
            probe_suite = load_probes(probes or config.validation.probes_path)

        se = None
        se_path = _result_path(config, SELF_ENHANCEMENT_FILE)
        if se_path.is_file():
            from src.validator import SelfEnhancementResult

            se = SelfEnhancementResult.model_validate_json(se_path.read_text(encoding="utf-8"))

        _echo(f"run {rid}: bias ablation (each condition runs the suite twice)")
        _families_banner(config)
        report = run_bias_ablation(
            config=config,
            rubric=rubric,
            client=client,
            logger=logger,
            run_id=rid,
            pointwise_cases=cases,
            pointwise_suite_id=loaded.suite_id,
            pointwise_suite_path=suite,
            comparison_cases=comp_cases,
            comparison_suite_id=comp_id,
            comparison_suite_path=comp_path,
            config_a_label=a_label,
            config_b_label=b_label,
            probe_suite=probe_suite,
            self_enhancement=se,
        )
        _echo("")
        for row in report.rows:
            _echo(f"  {row.bias} ({'ablatable' if row.ablatable else 'not flag-ablatable'})")
            _echo(f"    before: {json.dumps(row.before, default=str)[:220]}")
            _echo(f"    after : {json.dumps(row.after, default=str)[:220]}")
            _echo(f"    delta : {json.dumps(row.delta, default=str)[:220]}")
        path = write_json(_result_path(config, "bias_ablation_report.json", out), report)
        _echo("")
        _echo(f"  wrote {path}")
    except PipelineError as exc:
        _fail(exc)


@app.command()
def replay(
    run_id: str = typer.Argument(..., help="A run id from logs/verdicts.jsonl."),
    stage: str = typer.Option("run", "--stage", help="run | compare | ablation-* | gold | probe"),
    report: str | None = typer.Option(
        "results/suite_report.json", "--report", help="Committed report to diff against."
    ),
    config_path: str = typer.Option(DEFAULT_CONFIG, "--config"),
    rubric_path: str = typer.Option(DEFAULT_RUBRIC, "--rubric"),
    out: str | None = typer.Option(None, "--out", help="Write the rebuilt report here."),
) -> None:
    """Rebuild a report from logs/verdicts.jsonl alone and diff it against the committed one."""
    try:
        load_dotenv()
        config = load_config(config_path)
        rubric = load_rubric(rubric_path)
        expected = report if report and Path(report).is_file() else None
        if report and not expected:
            typer.secho(
                f"  note: {report} does not exist; rebuilding without a comparison.",
                fg=typer.colors.YELLOW,
            )
        result = replay_run(
            run_id,
            config,
            stage=stage,
            criteria=rubric.criterion_names,
            expected_report_path=expected,
        )
        _echo(f"replayed {result.n_verdicts} verdict(s) for run {run_id} (stage={stage})")
        if result.rebuilt.mode == "pointwise":
            _print_pointwise(result.rebuilt)
        else:
            _print_pairwise(result.rebuilt)
        _echo("")
        if expected is None:
            _echo("  no committed report to compare against.")
        elif result.matches:
            typer.secho(
                f"  MATCH: the report rebuilt from the log is identical to {expected} "
                "(ignoring generated_at and wall-clock latency).",
                fg=typer.colors.GREEN,
            )
        else:
            typer.secho(f"  MISMATCH vs {expected}:", fg=typer.colors.RED)
            for d in result.differences[:40]:
                _echo(f"    {d}")
            raise typer.Exit(code=2)
        if out:
            write_json(out, result.rebuilt)
            _echo(f"  wrote {out}")
    except (PipelineError, ValueError) as exc:
        _fail(exc)


@app.command()
def gate(
    report: str = typer.Option(
        None, "--report", help="judge_validation_report.json for the run being gated."
    ),
    baseline: str = typer.Option(
        None, "--baseline", help="A previous judge_validation_report.json to regress against."
    ),
    comparison: str = typer.Option(
        None, "--comparison", help="ab_comparison_report.json, for the flip-rate check."
    ),
    config_path: str = typer.Option(DEFAULT_CONFIG, "--config"),
    out: str | None = typer.Option(None, "--out"),
) -> None:
    """Deployment-readiness gate. Exits 1 when it blocks, so CI can use it.

    Gates on instrument *regression*, not on absolute quality: a hard stop on
    the flip rate, and a hard stop on any probe category that falls below its
    previous run's lower confidence bound. Gold agreement is reported and never
    blocks -- at n=15 with no established labelling ceiling, an absolute bar is
    a number this design cannot defend, and a gate nobody trusts gets disabled.
    """
    try:
        config = load_config(config_path, enforce=False)
        current_path = report or str(_result_path(config, "judge_validation_report.json"))
        current = JudgeValidationReport.model_validate_json(
            Path(current_path).read_text(encoding="utf-8")
        )
        base = (
            JudgeValidationReport.model_validate_json(Path(baseline).read_text(encoding="utf-8"))
            if baseline
            else None
        )
        comparison_path = comparison or str(_result_path(config, "ab_comparison_report.json"))
        comp = read_report(comparison_path) if Path(comparison_path).is_file() else None

        result = evaluate_gate(
            current,
            baseline=base,
            comparison=comp,
            flip_rate_threshold=config.comparison.flip_rate_threshold,
        )
        _echo("")
        for line in format_gate(result):
            _echo(line)
        _echo("")
        _echo(f"  rule: {result.rule}")

        destination = _result_path(config, "gate_report.json", out)
        write_json(destination, result)
        _echo(f"\n  wrote {destination}")

        if result.blocked:
            raise typer.Exit(code=1)
    except (PipelineError, ValueError, OSError) as exc:
        _fail(exc)


@app.command()
def smoke_test(
    config_path: str = typer.Option(DEFAULT_CONFIG, "--config"),
) -> None:
    """One tiny live call to each provider. Run this before anything else."""
    load_dotenv()
    config = load_config(config_path, enforce=False)
    client = LLMClient(cache_path=None, enable_cache=False, max_calls=4)
    models = [config.judge.model]
    if config.self_enhancement_probe_judge:
        models.append(config.self_enhancement_probe_judge.model)
    failures = 0
    for model in models:
        _echo(f"\n{model}")
        try:
            check_api_key(model)
            resp = client.complete(
                model=model,
                messages=[{"role": "user", "content": "Reply with the single word: ready"}],
                temperature=0.0,
                # Both configured models are reasoning models: they spend 250-350
                # completion tokens thinking before emitting a single visible one.
                # A budget under that floor returns finish_reason="length" with
                # empty content, so a 16-token probe reported success while
                # proving nothing.
                max_tokens=SMOKE_TEST_MAX_TOKENS,
            )
            _echo(f"  response : {resp.text.strip()[:80]!r}")
            _echo(
                f"  usage    : {resp.prompt_tokens} in / {resp.completion_tokens} out, "
                f"{resp.latency_ms:.0f} ms, finish={resp.finish_reason}"
            )
            if not resp.text.strip():
                failures += 1
                typer.secho(
                    f"  FAILED   : empty response (finish_reason={resp.finish_reason}). "
                    "The provider billed tokens but returned no text, so this model "
                    "cannot drive a judge run.",
                    fg=typer.colors.RED,
                )
        except Exception as exc:  # noqa: BLE001 - a smoke test reports, it does not raise
            failures += 1
            typer.secho(f"  FAILED   : {type(exc).__name__}: {exc}", fg=typer.colors.RED)
    _echo("")
    _echo(
        "Check your actual per-day request ceiling before planning a run. Google no longer "
        "publishes free-tier limits, and a measured run here recorded "
        "GenerateRequestsPerDayPerProjectPerModel-FreeTier limit: 20 for gemini-3.5-flash -- "
        "20 requests per DAY, against a validation suite that needs ~370. That is why the judge "
        "is a Groq-hosted model (1,000 RPD / 30 RPM on the same free tier). Gemini limits: "
        "https://aistudio.google.com/rate-limit  Groq limits: https://console.groq.com/settings/limits"
    )
    if failures:
        raise typer.Exit(code=1)


if __name__ == "__main__":
    try:
        app()
    except PipelineError as exc:  # pragma: no cover - defensive top-level guard
        typer.secho(f"{type(exc).__name__}: {exc}", fg=typer.colors.RED, err=True)
        sys.exit(1)
