"""The deployment-readiness gate.

The design commitment being tested is as much about what does *not* block as
what does: absolute quality is never gated, because at n=15 with no established
labelling ceiling that would be a number the design cannot defend.
"""

from __future__ import annotations

from src.gating import evaluate_gate
from src.schema import Interval, PairwiseTally, SuiteReport
from src.stats import wilson_interval
from src.validator import (
    JudgeValidationReport,
    ProbeCategoryResult,
    ProbeValidation,
)


def _validation(run_id: str, rates: dict[str, tuple[int, int]]) -> JudgeValidationReport:
    """Build a report whose probe categories have the given (passed, evaluated)."""
    rows = [
        ProbeCategoryResult(
            category=category,
            n_probes=n,
            n_evaluated=n,
            n_passed=passed,
            pass_rate=(passed / n if n else None),
            wilson_ci=wilson_interval(passed, n),
        )
        for category, (passed, n) in rates.items()
    ]
    return JudgeValidationReport(
        run_id=run_id,
        generated_at="2026-07-27T00:00:00Z",
        judge_model="groq/llama-3.3-70b-versatile",
        judge_temperature=0.0,
        probes=ProbeValidation(
            n_probes=sum(n for _, n in rates.values()),
            n_evaluated=sum(n for _, n in rates.values()),
            n_passed=sum(p for p, _ in rates.values()),
            by_category=rows,
        ),
    )


def _comparison(flip_rate: float | None, completed: int = 10) -> SuiteReport:
    return SuiteReport(
        run_id="cmp",
        generated_at="2026-07-27T00:00:00Z",
        mode="pairwise",
        tally=PairwiseTally(
            n_cases=completed,
            completed_pairs=completed,
            flip_rate=flip_rate,
        ),
    )


def test_a_clean_run_passes() -> None:
    current = _validation("r2", {"verbosity": (6, 6)})
    baseline = _validation("r1", {"verbosity": (6, 6)})
    report = evaluate_gate(current, baseline=baseline, comparison=_comparison(0.05))
    assert report.decision == "PASS"
    assert report.baseline_run_id == "r1"


def test_a_high_flip_rate_blocks() -> None:
    """The judge saying it cannot resolve the suite is the clearest stop signal."""
    current = _validation("r2", {"verbosity": (6, 6)})
    report = evaluate_gate(current, comparison=_comparison(0.40), flip_rate_threshold=0.20)
    assert report.decision == "BLOCK"
    flip = next(c for c in report.checks if c.name == "flip_rate")
    assert flip.blocking and not flip.passed
    assert "cannot resolve" in flip.detail


def test_a_probe_category_below_its_baseline_lower_bound_blocks() -> None:
    baseline = _validation("r1", {"sycophancy": (6, 6)})  # 100%, lower bound ~0.61
    current = _validation("r2", {"sycophancy": (2, 6)})  # 33%
    report = evaluate_gate(current, baseline=baseline, comparison=_comparison(0.0))
    assert report.decision == "BLOCK"
    check = next(c for c in report.checks if c.name.endswith("sycophancy"))
    assert not check.passed
    assert "sampling noise" in check.detail


def test_a_small_drop_within_noise_does_not_block() -> None:
    """Comparing point estimates would fire constantly at 6 probes per category;
    comparing against the interval only fires on a real regression."""
    baseline = _validation("r1", {"sycophancy": (5, 6)})  # 83%, lower bound well below
    current = _validation("r2", {"sycophancy": (4, 6)})  # 67%
    report = evaluate_gate(current, baseline=baseline, comparison=_comparison(0.0))
    assert report.decision == "PASS"


def test_the_first_run_is_not_judged_against_a_baseline_it_defines() -> None:
    current = _validation("r1", {"verbosity": (3, 6)})
    report = evaluate_gate(current, baseline=None, comparison=_comparison(0.0))
    assert report.decision == "PASS"
    check = next(c for c in report.checks if c.name == "probe_regression")
    assert not check.blocking
    assert "cannot also be judged by it" in check.detail


def test_a_missing_flip_rate_is_skipped_not_assumed_clean() -> None:
    """No resolved pair means the gate has no evidence, which is not the same as
    evidence of consistency. It must be visible as un-evaluated."""
    current = _validation("r2", {"verbosity": (6, 6)})
    report = evaluate_gate(current, comparison=_comparison(None))
    flip = next(c for c in report.checks if c.name == "flip_rate")
    assert not flip.blocking
    assert "cannot be evaluated" in flip.detail


def test_gold_agreement_is_reported_but_never_blocks() -> None:
    """The core design commitment: this gate regresses the instrument, it does
    not set an absolute quality bar the sample size cannot support."""
    current = _validation("r2", {"verbosity": (6, 6)})
    report = evaluate_gate(current, comparison=_comparison(0.0))
    gold_checks = [c for c in report.checks if c.name == "gold_agreement"]
    assert all(not c.blocking for c in gold_checks)


def test_wilson_lower_bound_is_what_regression_is_measured_against() -> None:
    """Guards the choice itself: 6/6 has a lower bound near 0.61, not 1.0, so a
    single failed probe next run is tolerated and four are not."""
    ci: Interval | None = wilson_interval(6, 6)
    assert ci is not None
    assert 0.5 < ci.low < 0.8
