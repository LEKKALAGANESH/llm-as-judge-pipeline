"""Report serialisation and the replay check.

``replay`` is the reason the audit-log claim is a demonstrated property rather
than an assertion: it rebuilds a ``SuiteReport`` from ``logs/verdicts.jsonl``
alone and diffs it against the committed report, field by field. If the log were
incomplete, mutated, or written after aggregation rather than alongside it, the
diff would show it. The same reconstruction doubles as ``--resume``: a run that
dies at case 24 of 25 has 24 verdicts on disk and can pick up from there.

Two fields are excluded from the comparison, deliberately and explicitly:
``generated_at`` (wall-clock at report time) and ``total_latency_ms`` (wall-clock
per call). Everything else -- every rate, count, score and decision -- must match
exactly.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from .aggregator import aggregate
from .audit_log import AuditLogger, utc_now_iso
from .schema import SuiteReport, Verdict
from .suite_config import PipelineConfig

VOLATILE_FIELDS = ("generated_at", "total_latency_ms")


def write_json(path: str | Path, model: BaseModel) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        json.dumps(json.loads(model.model_dump_json()), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return p


def read_report(path: str | Path) -> SuiteReport:
    return SuiteReport.model_validate_json(Path(path).read_text(encoding="utf-8"))


def _flatten(obj: Any, prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.update(_flatten(v, f"{prefix}.{k}" if prefix else str(k)))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            out.update(_flatten(v, f"{prefix}[{i}]"))
    else:
        out[prefix] = obj
    return out


def diff_reports(
    left: SuiteReport, right: SuiteReport, *, ignore: Sequence[str] = VOLATILE_FIELDS
) -> list[str]:
    lf = _flatten(json.loads(left.model_dump_json()))
    rf = _flatten(json.loads(right.model_dump_json()))
    ignored = set(ignore)
    diffs: list[str] = []
    for key in sorted(set(lf) | set(rf)):
        if key.split(".")[0] in ignored:
            continue
        a, b = lf.get(key, "<missing>"), rf.get(key, "<missing>")
        # Floats are compared with a tolerance because JSON round-tripping a
        # rate can shift the last bit; everything else must match exactly.
        if isinstance(a, float) and isinstance(b, float) and abs(a - b) <= 1e-9:
            continue
        if a != b:
            diffs.append(f"{key}: rebuilt={a!r} committed={b!r}")
    return diffs


class ReplayResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str
    stage: str
    n_verdicts: int
    matches: bool
    compared_against: str | None = None
    differences: list[str] = Field(default_factory=list)
    rebuilt: SuiteReport


def rebuild_report(
    verdicts: Sequence[Verdict],
    config: PipelineConfig,
    *,
    run_id: str,
    criteria: Sequence[str] = (),
    suite_id: str = "",
    suite_path: str = "",
    config_a_label: str | None = None,
    config_b_label: str | None = None,
    generated_at: str | None = None,
    mitigations: dict[str, bool] | None = None,
) -> SuiteReport:
    # Sorted, not set-iteration order: a report rebuilt from the same log must be
    # byte-identical every time, or the replay check becomes flaky.
    judge_models = sorted({v.judge_model for v in verdicts if v.judge_model})
    judge_temps = sorted({v.judge_temperature for v in verdicts})
    price = config.price_for(judge_models[0] if judge_models else config.judge.model)
    return aggregate(
        verdicts,
        run_id=run_id,
        generated_at=generated_at or utc_now_iso(),
        criteria=criteria,
        pass_threshold=config.scoring.pass_threshold,
        min_criterion_score=config.scoring.min_criterion_score,
        flip_rate_threshold=config.comparison.flip_rate_threshold,
        win_rate_threshold=config.comparison.win_rate_threshold,
        suite_id=suite_id,
        suite_path=suite_path,
        judge_model=judge_models[0] if judge_models else config.judge.model,
        judge_temperature=judge_temps[0] if judge_temps else config.judge.temperature,
        # From the log when it recorded them, not from whatever config is
        # loaded now: replaying a `--mitigations off` run against an on-config
        # otherwise reports MISMATCH on an intact log.
        mitigations=mitigations if mitigations is not None else config.mitigations.as_dict(),
        resolved_families=config.resolved_families,
        cross_family_enforced=config.cross_family_enforced,
        config_a_label=config_a_label,
        config_b_label=config_b_label,
        price_per_million=(price.input, price.output),
        cost_note=COST_NOTE,
    )


COST_NOTE = (
    "Both providers are free tier, so the actual spend on this run was $0.00. The figure above is "
    "the counterfactual paid-tier cost at the list prices in suite_config.yaml, reported because "
    "'we did not track cost because it was free' misses the point of the requirement."
)


def replay(
    run_id: str,
    config: PipelineConfig,
    *,
    stage: str,
    criteria: Sequence[str] = (),
    expected_report_path: str | Path | None = None,
    suite_id: str = "",
    suite_path: str = "",
    config_a_label: str | None = None,
    config_b_label: str | None = None,
) -> ReplayResult:
    verdicts = AuditLogger.read_verdicts(config.run.logs_dir, run_id, stage=stage)
    if not verdicts:
        raise ValueError(
            f"no verdicts logged for run_id={run_id!r} stage={stage!r} in "
            f"{config.run.logs_dir}/verdicts.jsonl. Known run ids: "
            f"{AuditLogger.known_run_ids(config.run.logs_dir)}"
        )
    expected = read_report(expected_report_path) if expected_report_path else None
    rebuilt = rebuild_report(
        verdicts,
        config,
        run_id=run_id,
        mitigations=AuditLogger.read_mitigations(config.run.logs_dir, run_id, stage=stage),
        criteria=criteria,
        suite_id=suite_id or (expected.suite_id if expected else ""),
        suite_path=suite_path or (expected.suite_path if expected else ""),
        config_a_label=config_a_label or (expected.config_a_label if expected else None),
        config_b_label=config_b_label or (expected.config_b_label if expected else None),
        generated_at=expected.generated_at if expected else None,
    )
    differences = diff_reports(rebuilt, expected) if expected else []
    return ReplayResult(
        run_id=run_id,
        stage=stage,
        n_verdicts=len(verdicts),
        matches=bool(expected) and not differences,
        compared_against=str(expected_report_path) if expected_report_path else None,
        differences=differences,
        rebuilt=rebuilt,
    )
