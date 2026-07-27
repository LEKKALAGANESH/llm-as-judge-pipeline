"""Deployment-readiness gate: the decision the README commits to, in code.

The README argues the judge should be a **pre-merge signal, not a gate** on
absolute quality -- at n=15 with an unestablished labelling ceiling, an absolute
bar is not something this design can defend. What it *can* defend is a
regression signal about the instrument itself, and that is what this module
enforces:

1. **Flip rate ceiling.** If the judge cannot resolve a suite consistently under
   order swap, it is telling you it cannot rank these candidates. Block.
2. **Probe-category regression.** Block if any adversarial category's pass rate
   falls below the *lower confidence bound* of its previous run. Comparing point
   estimates would fire on ordinary sampling noise at 5-6 probes per category;
   comparing against the interval only fires when the drop exceeds what noise
   explains.

Everything else is reported and blocks nothing, deliberately. A gate that
blocks on a number it cannot defend gets switched off, and then nothing is
gated at all.
"""

from __future__ import annotations

from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict, Field

from .schema import SuiteReport
from .validator import JudgeValidationReport


class GateCheck(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    passed: bool
    blocking: bool = Field(description="False for checks reported for context but never blocking.")
    detail: str


class GateReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    decision: str = Field(description="PASS or BLOCK")
    run_id: str
    baseline_run_id: str | None = None
    checks: list[GateCheck] = Field(default_factory=list)
    rule: str = ""

    @property
    def blocked(self) -> bool:
        return self.decision == "BLOCK"


GATE_RULE_TEXT = (
    "BLOCK if (a) the pairwise flip rate exceeds flip_rate_threshold, or (b) any adversarial "
    "probe category's pass rate falls below the lower bound of that category's Wilson interval "
    "in the baseline run. Everything else is advisory. Absolute quality is deliberately not "
    "gated: this gold set is n=15 with no established intra-rater ceiling, so an absolute bar "
    "would be a number the design cannot defend."
)


def evaluate_gate(
    current: JudgeValidationReport,
    *,
    baseline: JudgeValidationReport | None = None,
    comparison: SuiteReport | None = None,
    flip_rate_threshold: float = 0.20,
) -> GateReport:
    checks: list[GateCheck] = []

    # -- 1. flip rate ------------------------------------------------------
    tally = comparison.tally if comparison else None
    flip = tally.flip_rate if tally else None
    if flip is None:
        checks.append(
            GateCheck(
                name="flip_rate",
                passed=True,
                blocking=False,
                detail="No pairwise comparison supplied, or no pair resolved in both orders; "
                "the flip-rate gate cannot be evaluated and is skipped rather than assumed clean.",
            )
        )
    else:
        ok = flip <= flip_rate_threshold
        checks.append(
            GateCheck(
                name="flip_rate",
                passed=ok,
                blocking=True,
                detail=(
                    f"flip rate {flip:.1%} vs threshold {flip_rate_threshold:.0%} "
                    f"over {tally.completed_pairs if tally else 0} completed pairs"
                    + ("" if ok else " -- the judge cannot resolve this suite consistently")
                ),
            )
        )

    # -- 2. probe-category regression --------------------------------------
    current_rates = _probe_rates(current)
    baseline_bounds = _probe_lower_bounds(baseline) if baseline else {}
    if not baseline_bounds:
        checks.append(
            GateCheck(
                name="probe_regression",
                passed=True,
                blocking=False,
                detail="No baseline run supplied, so there is nothing to regress against. The "
                "first run establishes the baseline; it cannot also be judged by it.",
            )
        )
    else:
        for category in sorted(current_rates):
            rate = current_rates[category]
            bound = baseline_bounds.get(category)
            if rate is None or bound is None:
                checks.append(
                    GateCheck(
                        name=f"probe_regression:{category}",
                        passed=True,
                        blocking=False,
                        detail="Not comparable: missing in the current run or the baseline.",
                    )
                )
                continue
            ok = rate >= bound
            checks.append(
                GateCheck(
                    name=f"probe_regression:{category}",
                    passed=ok,
                    blocking=True,
                    detail=(
                        f"pass rate {rate:.0%} vs baseline lower bound {bound:.0%}"
                        + ("" if ok else " -- a drop larger than sampling noise explains")
                    ),
                )
            )

    # -- 3. advisory context ----------------------------------------------
    kappa = current.gold.overall_kappa if current.gold else None
    if kappa is not None:
        value = kappa.kappa_quadratic
        checks.append(
            GateCheck(
                name="gold_agreement",
                passed=True,
                blocking=False,
                detail=(
                    f"kappa_w = {value:+.3f}"
                    if value is not None
                    else f"kappa undefined: {kappa.reason}"
                )
                + ". Reported, never blocking: see the rule text.",
            )
        )

    blocking_failures = [c for c in checks if c.blocking and not c.passed]
    return GateReport(
        decision="BLOCK" if blocking_failures else "PASS",
        run_id=current.run_id,
        baseline_run_id=baseline.run_id if baseline else None,
        checks=checks,
        rule=GATE_RULE_TEXT,
    )


def _probe_rates(report: JudgeValidationReport | None) -> dict[str, float | None]:
    if report is None or report.probes is None:
        return {}
    return {row.category: row.pass_rate for row in report.probes.by_category}


def _probe_lower_bounds(report: JudgeValidationReport | None) -> dict[str, float | None]:
    """Wilson lower bound per category, which is what a regression is measured against."""
    if report is None or report.probes is None:
        return {}
    out: dict[str, float | None] = {}
    for row in report.probes.by_category:
        out[row.category] = row.wilson_ci.low if row.wilson_ci else None
    return out


def format_gate(report: GateReport) -> Sequence[str]:
    lines = [f"  decision : {report.decision}"]
    if report.baseline_run_id:
        lines.append(f"  baseline : {report.baseline_run_id}")
    for check in report.checks:
        mark = "PASS" if check.passed else "BLOCK"
        tag = "" if check.blocking else " (advisory)"
        lines.append(f"  [{mark}]{tag} {check.name}: {check.detail}")
    return lines
