"""End-to-end through the real CLI with a fake provider.

Every command is exercised: run-suite, compare-configs, validate-judge,
run-ablation, measure-self-enhancement, replay and smoke-test. Nothing here
needs an API key or a network -- the seam is ``_default_completion_fn``, so the
Typer layer, config loading, prompt building, the repair loop, the order-swap
protocol, aggregation, report writing and replay are all the production code.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

import main as cli
from src.audit_log import AuditLogger
from src.llm_client import RateLimitError
from tests.conftest import FakeJudge
from tests.fixtures import malformed_responses as mr

runner = CliRunner()


@pytest.fixture
def workspace(tmp_path: Path, repo_root: Path, monkeypatch: pytest.MonkeyPatch):
    """A temp config pointing every output path inside tmp_path."""
    raw = yaml.safe_load((repo_root / "config" / "suite_config.yaml").read_text(encoding="utf-8"))
    raw["run"]["results_dir"] = str(tmp_path / "results")
    raw["run"]["logs_dir"] = str(tmp_path / "logs")
    raw["run"]["cache_path"] = str(tmp_path / "cache")
    raw["run"]["max_concurrent_judge_calls"] = 1
    # No client-side TPM pacing against a fake provider: there is no quota to
    # protect here, and pacing would make the offline suite sleep for minutes.
    raw["run"]["tokens_per_minute"] = None
    cfg_path = tmp_path / "suite_config.yaml"
    cfg_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    monkeypatch.setenv("GEMINI_API_KEY", "test-key-not-real")
    monkeypatch.setenv("GROQ_API_KEY", "test-key-not-real")
    return {
        "config": str(cfg_path),
        "rubric": str(repo_root / "config" / "rubric.yaml"),
        "suite": str(repo_root / "data" / "test_suites" / "general_qa.json"),
        "comparison": str(repo_root / "data" / "test_suites" / "comparison_qa.json"),
        "probes": str(repo_root / "data" / "test_suites" / "adversarial_probes.json"),
        "gold": str(repo_root / "data" / "gold_labels" / "human_annotated.json"),
        "logs": tmp_path / "logs",
        "results": tmp_path / "results",
        "tmp": tmp_path,
    }


def install_judge(monkeypatch: pytest.MonkeyPatch, judge: FakeJudge) -> FakeJudge:
    monkeypatch.setattr("src.llm_client._default_completion_fn", lambda: judge)
    return judge


def run(args: list[str]):
    return runner.invoke(cli.app, args, catch_exceptions=False)


# --------------------------------------------------------------------------
# run-suite
# --------------------------------------------------------------------------


def test_run_suite_end_to_end_with_a_malformed_reply_and_a_transient_error(workspace, monkeypatch):
    judge = install_judge(
        monkeypatch,
        FakeJudge(
            # gq-002's output triggers one malformed reply (repaired on retry);
            # gq-010's triggers one 429 (retried transparently by the client).
            malformed_markers=("was signed in 1945",),
            transient_markers=("I'm sorry, but I can't help with requests",),
        ),
    )
    out = workspace["tmp"] / "suite_report.json"
    result = run(
        [
            "run-suite",
            workspace["suite"],
            "--config",
            workspace["config"],
            "--rubric",
            workspace["rubric"],
            "--out",
            str(out),
        ]
    )
    assert result.exit_code == 0, result.output
    report = json.loads(out.read_text(encoding="utf-8"))

    assert report["mode"] == "pointwise"
    assert report["n_cases"] == 18
    assert report["n_scored"] == 18, "both injected failures should have recovered"
    assert report["pass_rate"] is not None
    assert set(report["mean_scores_by_criterion"]) == {
        "correctness",
        "faithfulness",
        "completeness",
        "instruction_following",
        "tone",
        "safety",
    }
    assert report["overall_score_distribution"]["n"] == 18
    assert report["length_vs_score_spearman"]["n"] == 18
    assert report["cross_family_enforced"] is True
    assert report["resolved_families"]["judge"] != report["resolved_families"].get("candidate:c0")
    assert report["total_attempts"] > report["n_cases"], "the repair attempt should be counted"

    # 18 cases + 1 repair + 1 transient retry
    assert len(judge.calls) == 20

    # Audit trail: one record per attempt, and the failed attempt is in there.
    records = AuditLogger.read_calls(workspace["logs"], report["run_id"])
    assert len(records) == 19, "transient retries happen below the logging layer"
    assert any(r["parse_ok"] is False for r in records)
    assert all(r["messages"] for r in records)


def test_run_suite_with_sample_size_and_mitigations_off(workspace, monkeypatch):
    install_judge(monkeypatch, FakeJudge())
    out = workspace["tmp"] / "ablated.json"
    result = run(
        [
            "run-suite",
            workspace["suite"],
            "--config",
            workspace["config"],
            "--rubric",
            workspace["rubric"],
            "--sample-size",
            "4",
            "--mitigations",
            "off",
            "--out",
            str(out),
        ]
    )
    assert result.exit_code == 0, result.output
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["n_cases"] == 4
    assert report["mitigations"]["few_shot_anchors"] is False
    assert report["mitigations"]["anti_verbosity_clause"] is False
    assert report["mitigations"]["rationale_first"] is True


def test_run_suite_reports_unrecoverable_failures_instead_of_crashing(workspace, monkeypatch):
    class AlwaysBad(FakeJudge):
        def __call__(self, **kwargs):
            self.calls.append(kwargs)
            from tests.conftest import FakeResponse

            return FakeResponse(mr.TRUNCATED)

    install_judge(monkeypatch, AlwaysBad())
    out = workspace["tmp"] / "broken.json"
    result = run(
        [
            "run-suite",
            workspace["suite"],
            "--config",
            workspace["config"],
            "--rubric",
            workspace["rubric"],
            "--sample-size",
            "3",
            "--out",
            str(out),
        ]
    )
    assert result.exit_code == 0, "one bad response must not abort the suite"
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["n_scored"] == 0
    assert report["n_parse_errors"] == 3
    assert len(report["failed_case_ids"]) == 3
    assert report["pass_rate"] is None


def test_quota_guard_stops_the_run(workspace, monkeypatch):
    install_judge(monkeypatch, FakeJudge())
    result = run(
        [
            "run-suite",
            workspace["suite"],
            "--config",
            workspace["config"],
            "--rubric",
            workspace["rubric"],
            "--max-calls-per-run",
            "2",
            "--out",
            str(workspace["tmp"] / "q.json"),
        ]
    )
    assert result.exit_code == 1
    assert "max_calls_per_run" in result.output or "max_calls_per_run" in str(result.exception)


# --------------------------------------------------------------------------
# compare-configs
# --------------------------------------------------------------------------


def test_compare_configs_runs_both_orders_and_declares_a_winner(workspace, monkeypatch):
    # A judge that always prefers whichever output it sees FIRST: the textbook
    # position-biased judge. Every pair must therefore come out inconsistent,
    # the flip rate must be 1.0, and the validity gate must refuse to name a
    # winner -- which is the behaviour the whole protocol exists to produce.
    judge = install_judge(monkeypatch, FakeJudge(winner_for=lambda user: "Model_A"))
    out = workspace["tmp"] / "ab.json"
    result = run(
        [
            "compare-configs",
            workspace["comparison"],
            "--config",
            workspace["config"],
            "--rubric",
            workspace["rubric"],
            "--sample-size",
            "5",
            "--no-noise-floor",
            "--out",
            str(out),
        ]
    )
    assert result.exit_code == 0, result.output
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["mode"] == "pairwise"
    assert report["tally"]["n_cases"] == 5
    assert report["tally"]["inconsistent_pairs"] == 5
    assert report["tally"]["flip_rate"] == 1.0
    assert report["winner"] == "Judge unreliable on this suite - no winner declared"
    assert "short-circuit" in " ".join(report["decision_trace"])
    assert len(judge.calls) == 10, "exactly two calls per pairwise case, never more"


def test_compare_configs_with_a_consistent_judge_names_a_winner_and_a_noise_floor(
    workspace, monkeypatch
):
    # Content-based, order-independent preference: config B's text always wins on
    # the merits, wherever it is shown. A judge like this must produce a 0% flip
    # rate, which is what makes the position protocol's output meaningful.
    b_snippets = [
        case["output_b"][:40]
        for case in json.loads(Path(workspace["comparison"]).read_text(encoding="utf-8"))["cases"]
    ]

    def winner_for(user: str) -> str:
        a_block = user[user.index("BEGIN OUTPUT_MODEL_A") : user.index("BEGIN OUTPUT_MODEL_B")]
        return "Model_A" if any(s in a_block for s in b_snippets) else "Model_B"

    install_judge(monkeypatch, FakeJudge(winner_for=winner_for))
    out = workspace["tmp"] / "ab2.json"
    result = run(
        [
            "compare-configs",
            workspace["comparison"],
            "--config",
            workspace["config"],
            "--rubric",
            workspace["rubric"],
            "--sample-size",
            "4",
            "--out",
            str(out),
        ]
    )
    assert result.exit_code == 0, result.output
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["tally"]["flip_rate"] == 0.0
    assert report["tally"]["wins_b"] == 4
    assert report["winner"] == "prompt_v2_direct"
    assert report["tally"]["win_rate_b_ci"]["method"] == "clopper-pearson"

    # The noise floor lands under run.results_dir, which the workspace redirects
    # into tmp_path -- no command writes outside the configured results dir.
    nf_path = workspace["results"] / cli.NOISE_FLOOR_FILE
    nf = json.loads(nf_path.read_text(encoding="utf-8"))
    assert nf["n_pairs"] == 4
    assert nf["same_order_flip_rate"] == 0.0
    assert nf["noise_corrected_position_bias"] == 0.0


# --------------------------------------------------------------------------
# validate-judge
# --------------------------------------------------------------------------


def test_validate_judge_produces_all_three_validation_numbers(workspace, monkeypatch):
    # Score the judge's opinion off the gold labels by a fixed offset so kappa is
    # neither degenerate nor perfect.
    lookup = {
        label["case_id"]: label["overall_score"]
        for label in json.loads(Path(workspace["gold"]).read_text(encoding="utf-8"))["labels"]
    }
    cases = json.loads(Path(workspace["suite"]).read_text(encoding="utf-8"))["cases"]
    by_output = {c["model_output"][:60]: c["case_id"] for c in cases}

    def score_for(user: str) -> int:
        for snippet, case_id in by_output.items():
            if snippet in user and case_id in lookup:
                return min(5, max(1, lookup[case_id] + 1))
        return 3

    install_judge(monkeypatch, FakeJudge(score_for=score_for))
    out = workspace["tmp"] / "validation.json"
    result = run(
        [
            "validate-judge",
            "--suite",
            workspace["suite"],
            "--config",
            workspace["config"],
            "--rubric",
            workspace["rubric"],
            "--gold",
            workspace["gold"],
            "--probes",
            workspace["probes"],
            "--retest-runs",
            "3",
            "--retest-sample",
            "4",
            "--out",
            str(out),
        ]
    )
    assert result.exit_code == 0, result.output
    report = json.loads(out.read_text(encoding="utf-8"))

    gold = report["gold"]
    assert gold["n_judged"] == 15
    assert gold["overall_kappa"]["n"] == 15
    assert gold["overall_kappa"]["kappa_quadratic"] is not None
    assert gold["overall_kappa"]["kappa_ci"] is not None
    assert len(gold["overall_kappa"]["confusion_matrix"]) == 5
    assert gold["overall_kappa"]["spearman"]["rho"] is not None
    assert gold["overall_kappa"]["interpretation"]
    assert gold["pooled_criterion_kappa"]["n"] == 90, "15 cases x 6 criteria"
    assert gold["intra_rater_kappa"] is None
    assert "ceiling" in gold["intra_rater_note"].lower()
    assert gold["labeling_method"]["labeled_before_judge_run"] is True

    retest = report["test_retest"]
    assert retest["runs"] == 3 and retest["temperature"] == 0.7
    assert retest["sampling_warning"] is True, "a deterministic fake must trip the control"
    assert "byte-identical" in retest["note"]

    probes = report["probes"]
    assert probes["n_probes"] == 27
    categories = {row["category"] for row in probes["by_category"]}
    assert categories == {
        "verbosity",
        "sycophancy",
        "position_noise_floor",
        "over_correction",
        "prompt_injection",
    }
    for row in probes["by_category"]:
        assert row["n_evaluated"] >= 5
        assert row["wilson_ci"] is not None
    assert probes["noise_floor_non_tie_rate"] is not None
    assert any("Noise-corrected" in c or "Self-enhancement" in c for c in report["caveats"])


def test_validate_judge_refuses_test_retest_at_zero_temperature(workspace, monkeypatch):
    install_judge(monkeypatch, FakeJudge())
    result = run(
        [
            "validate-judge",
            "--suite",
            workspace["suite"],
            "--config",
            workspace["config"],
            "--rubric",
            workspace["rubric"],
            "--gold",
            workspace["gold"],
            "--retest-temperature",
            "0",
            "--skip-probes",
            "--out",
            str(workspace["tmp"] / "v.json"),
        ]
    )
    assert result.exit_code == 1
    assert "not a validation result" in result.output


def test_validate_judge_fails_loudly_on_a_missing_gold_file(workspace, monkeypatch):
    install_judge(monkeypatch, FakeJudge())
    result = run(
        [
            "validate-judge",
            "--suite",
            workspace["suite"],
            "--config",
            workspace["config"],
            "--rubric",
            workspace["rubric"],
            "--gold",
            str(workspace["tmp"] / "absent.json"),
            "--out",
            str(workspace["tmp"] / "v.json"),
        ]
    )
    assert result.exit_code == 1
    assert "not found" in result.output


# --------------------------------------------------------------------------
# self-enhancement, ablation
# --------------------------------------------------------------------------


def test_self_enhancement_requires_the_explicit_waiver(workspace, monkeypatch):
    install_judge(monkeypatch, FakeJudge())
    result = run(
        [
            "measure-self-enhancement",
            workspace["comparison"],
            "--config",
            workspace["config"],
            "--rubric",
            workspace["rubric"],
        ]
    )
    assert result.exit_code == 1
    assert "--allow-same-family" in result.output


def test_self_enhancement_measures_a_win_rate_delta(workspace, monkeypatch):
    def winner_for_model(model: str):
        # The same-family judge (Groq) prefers B; the cross-family judge is split.
        def _w(user: str) -> str:
            return "Model_B"

        return _w

    class DualJudge(FakeJudge):
        def __call__(self, *, model, messages, **params):
            self.winner_for = (
                (lambda user: "Model_B") if "groq" in model else (lambda user: "Model_A")
            )
            return super().__call__(model=model, messages=messages, **params)

    install_judge(monkeypatch, DualJudge())
    out = workspace["tmp"] / "se.json"
    result = run(
        [
            "measure-self-enhancement",
            workspace["comparison"],
            "--config",
            workspace["config"],
            "--rubric",
            workspace["rubric"],
            "--allow-same-family",
            "--sample-size",
            "3",
            "--out",
            str(out),
        ]
    )
    assert result.exit_code == 0, result.output
    report = json.loads(out.read_text(encoding="utf-8"))
    # The cross-family judge must differ from the candidates; the same-family
    # probe judge must match them. The specific vendor is a quota decision.
    assert report["cross_family_judge_family"] != report["candidate_family"]
    assert report["same_family_judge_family"] == "openai"
    assert report["candidate_family"] == "openai"
    # Both judges are perfectly order-consistent in their own frame, so both
    # produce inconsistent pairs; what matters is that the delta is computed and
    # the confound is stated.
    assert "upper bound" in report["interpretation"]


def test_run_ablation_emits_a_before_after_row_for_every_bias(workspace, monkeypatch):
    install_judge(monkeypatch, FakeJudge())
    out = workspace["tmp"] / "ablation.json"
    result = run(
        [
            "run-ablation",
            "--suite",
            workspace["suite"],
            "--comparison",
            workspace["comparison"],
            "--probes",
            workspace["probes"],
            "--config",
            workspace["config"],
            "--rubric",
            workspace["rubric"],
            "--sample-size",
            "3",
            "--out",
            str(out),
        ]
    )
    assert result.exit_code == 0, result.output
    report = json.loads(out.read_text(encoding="utf-8"))
    biases = {row["bias"] for row in report["rows"]}
    assert biases == {
        "position",
        "verbosity",
        "self_enhancement",
        "sycophancy_style",
        "score_clustering",
    }
    assert report["pointwise_with_mitigations"]["mitigations"]["few_shot_anchors"] is True
    assert report["pointwise_without_mitigations"]["mitigations"]["few_shot_anchors"] is False
    assert report["pairwise_with_order_swap"]["mitigations"]["order_swap"] is True
    assert report["pairwise_without_order_swap"]["mitigations"]["order_swap"] is False
    clustering = next(r for r in report["rows"] if r["bias"] == "score_clustering")
    assert "std_dev" in clustering["before"] and "std_dev" in clustering["after"]
    non_ablatable = {r["bias"] for r in report["rows"] if not r["ablatable"]}
    assert non_ablatable == {"self_enhancement", "sycophancy_style"}

    # A delta that is silently None is worse than a missing row: the report
    # still looks complete. `_build_rows` read "rho" while `_rho` emits
    # "spearman_rho", so the only before/after number for verbosity was null on
    # every run and no assertion here caught it.
    verbosity = next(r for r in report["rows"] if r["bias"] == "verbosity")
    for side in ("before", "after"):
        assert "spearman_rho" in verbosity[side], (
            f"delta is read from this key; {side} must publish it under the same name"
        )
    # Whenever both operands are present the delta must be computed. Asserting
    # only that *some* delta is non-null is too weak: the padded-probe delta
    # masked a permanently-null rho.
    if verbosity["before"]["spearman_rho"] is not None and (
        verbosity["after"]["spearman_rho"] is not None
    ):
        assert verbosity["delta"]["spearman_rho"] is not None, (
            "both rho values exist, so a null delta means the key names disagree"
        )


# --------------------------------------------------------------------------
# replay: the audit claim, demonstrated
# --------------------------------------------------------------------------


def test_replay_rebuilds_the_report_from_the_log_alone(workspace, monkeypatch):
    install_judge(monkeypatch, FakeJudge())
    out = workspace["tmp"] / "suite_report.json"
    first = run(
        [
            "run-suite",
            workspace["suite"],
            "--config",
            workspace["config"],
            "--rubric",
            workspace["rubric"],
            "--sample-size",
            "6",
            "--out",
            str(out),
        ]
    )
    assert first.exit_code == 0, first.output
    run_id = json.loads(out.read_text(encoding="utf-8"))["run_id"]

    replayed = run(
        [
            "replay",
            run_id,
            "--stage",
            "run",
            "--report",
            str(out),
            "--config",
            workspace["config"],
            "--rubric",
            workspace["rubric"],
        ]
    )
    assert replayed.exit_code == 0, replayed.output
    assert "MATCH" in replayed.output
    assert "replayed 6 verdict" in replayed.output


def test_replay_of_a_mitigations_off_run_matches(workspace, monkeypatch):
    """Replay must read the mitigation set back from the log, not from config.

    `run-ablation` produces `--mitigations off` runs, and replaying one against
    the shipped (mitigations-on) config used to report MISMATCH on a perfectly
    intact log -- on exactly the runs that carry the bias evidence.
    """
    install_judge(monkeypatch, FakeJudge())
    out = workspace["tmp"] / "ablated_report.json"
    first = run(
        [
            "run-suite",
            workspace["suite"],
            "--config",
            workspace["config"],
            "--rubric",
            workspace["rubric"],
            "--sample-size",
            "4",
            "--mitigations",
            "off",
            "--out",
            str(out),
        ]
    )
    assert first.exit_code == 0, first.output
    committed = json.loads(out.read_text(encoding="utf-8"))
    assert committed["mitigations"]["few_shot_anchors"] is False

    # Replayed with the default config, whose mitigations are ON.
    replayed = run(
        [
            "replay",
            committed["run_id"],
            "--stage",
            "run",
            "--report",
            str(out),
            "--config",
            workspace["config"],
            "--rubric",
            workspace["rubric"],
        ]
    )
    assert replayed.exit_code == 0, replayed.output
    assert "MATCH" in replayed.output, "the log records how the run was configured"


def test_replay_detects_a_tampered_report(workspace, monkeypatch):
    install_judge(monkeypatch, FakeJudge())
    out = workspace["tmp"] / "suite_report.json"
    run(
        [
            "run-suite",
            workspace["suite"],
            "--config",
            workspace["config"],
            "--rubric",
            workspace["rubric"],
            "--sample-size",
            "4",
            "--out",
            str(out),
        ]
    )
    report = json.loads(out.read_text(encoding="utf-8"))
    run_id = report["run_id"]
    report["pass_rate"] = 1.0
    report["n_passed"] = 4
    out.write_text(json.dumps(report), encoding="utf-8")

    replayed = run(
        [
            "replay",
            run_id,
            "--stage",
            "run",
            "--report",
            str(out),
            "--config",
            workspace["config"],
            "--rubric",
            workspace["rubric"],
        ]
    )
    assert replayed.exit_code == 2
    assert "MISMATCH" in replayed.output
    assert "pass_rate" in replayed.output


def test_resume_reuses_verdicts_already_on_disk(workspace, monkeypatch):
    judge = install_judge(monkeypatch, FakeJudge())
    out = workspace["tmp"] / "r.json"
    run(
        [
            "run-suite",
            workspace["suite"],
            "--config",
            workspace["config"],
            "--rubric",
            workspace["rubric"],
            "--sample-size",
            "3",
            "--out",
            str(out),
        ]
    )
    run_id = json.loads(out.read_text(encoding="utf-8"))["run_id"]
    calls_after_first = len(judge.calls)

    resumed = run(
        [
            "run-suite",
            workspace["suite"],
            "--config",
            workspace["config"],
            "--rubric",
            workspace["rubric"],
            "--sample-size",
            "5",
            "--resume",
            run_id,
            "--out",
            str(out),
        ]
    )
    assert resumed.exit_code == 0, resumed.output
    assert "resuming: 3 verdict" in resumed.output
    assert len(judge.calls) == calls_after_first + 2, "only the two new cases were judged"
    assert json.loads(out.read_text(encoding="utf-8"))["n_cases"] == 5


def test_replay_on_an_unknown_run_id_is_a_clear_error(workspace, monkeypatch):
    result = run(
        [
            "replay",
            "run-does-not-exist",
            "--config",
            workspace["config"],
            "--rubric",
            workspace["rubric"],
            "--report",
            str(workspace["tmp"] / "missing.json"),
        ]
    )
    assert result.exit_code == 1
    assert "no verdicts logged" in result.output


# --------------------------------------------------------------------------
# cache, concurrency, smoke test
# --------------------------------------------------------------------------


def test_second_identical_run_is_served_from_the_cache(workspace, monkeypatch):
    judge = install_judge(monkeypatch, FakeJudge())
    args = [
        "run-suite",
        workspace["suite"],
        "--config",
        workspace["config"],
        "--rubric",
        workspace["rubric"],
        "--sample-size",
        "5",
        "--out",
        str(workspace["tmp"] / "c1.json"),
    ]
    run(args)
    assert len(judge.calls) == 5
    run(args[:-1] + [str(workspace["tmp"] / "c2.json")])
    assert len(judge.calls) == 5, "the second run must cost zero provider calls"
    second = json.loads((workspace["tmp"] / "c2.json").read_text(encoding="utf-8"))
    assert second["total_judge_calls"] == 0
    assert second["total_cached_calls"] == 5


def test_bounded_concurrency_produces_the_same_report(workspace, monkeypatch, repo_root):
    raw = yaml.safe_load(Path(workspace["config"]).read_text(encoding="utf-8"))
    raw["run"]["max_concurrent_judge_calls"] = 4
    raw["run"]["cache_path"] = str(workspace["tmp"] / "cache_concurrent")
    concurrent_cfg = workspace["tmp"] / "concurrent.yaml"
    concurrent_cfg.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    install_judge(monkeypatch, FakeJudge())
    out = workspace["tmp"] / "conc.json"
    result = run(
        [
            "run-suite",
            workspace["suite"],
            "--config",
            str(concurrent_cfg),
            "--rubric",
            workspace["rubric"],
            "--sample-size",
            "8",
            "--out",
            str(out),
        ]
    )
    assert result.exit_code == 0, result.output
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["n_scored"] == 8
    records = AuditLogger.read_calls(workspace["logs"], report["run_id"])
    assert len(records) == 8, "concurrent writers must not lose or interleave log lines"


def test_smoke_test_reports_failures_without_raising(workspace, monkeypatch):
    def boom():
        def _fn(**kwargs):
            raise RateLimitError("429 from the fake provider")

        return _fn

    monkeypatch.setattr("src.llm_client._default_completion_fn", boom)
    result = run(["smoke-test", "--config", workspace["config"]])
    assert result.exit_code == 1
    assert "FAILED" in result.output
    assert "rate-limit" in result.output, "a failed smoke test must point at the quota pages"
