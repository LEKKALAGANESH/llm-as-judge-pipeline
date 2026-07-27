"""Append-only JSONL audit trail.

Two logs, written by the same object, for two different purposes:

``logs/judge_calls.jsonl``
    One record per *attempt* -- including failed and repaired attempts, and
    including cache hits -- carrying the full message list sent, the raw
    response text, token counts and ``latency_ms``. This is the audit trail:
    a parse failure still leaves a record, because the record is written before
    parsing is attempted.

``logs/verdicts.jsonl``
    One record per case per run, carrying the parsed verdict. This is what
    ``replay`` reconstructs the suite report from, and what ``--resume`` reads
    to skip work already on disk.

Append-only is a property, not an aspiration: nothing in this module opens a log
for anything but ``"a"``, and no code path rewrites or truncates one. That is
what makes the trail credible, so it is stated here rather than left implicit.
"""

from __future__ import annotations

import json
import os
import threading
import uuid
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .schema import Verdict, VerdictAdapter

CALLS_LOG = "judge_calls.jsonl"
VERDICTS_LOG = "verdicts.jsonl"


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def new_run_id(prefix: str = "run") -> str:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{prefix}-{stamp}-{uuid.uuid4().hex[:6]}"


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (set, frozenset)):
        return sorted(obj)
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, datetime):
        return obj.isoformat()
    return str(obj)


class AuditLogger:
    """Writes both logs for one run. Cheap to construct; opens files per write
    so that a crashed process never leaves a half-flushed buffer."""

    def __init__(self, logs_dir: str | Path = "logs", run_id: str | None = None) -> None:
        self.logs_dir = Path(logs_dir)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id or new_run_id()
        # The runner may drive several cases concurrently; one lock keeps whole
        # lines atomic so a JSONL record can never be interleaved with another.
        self._lock = threading.Lock()

    # -- writing -----------------------------------------------------------

    def _append(self, filename: str, record: dict[str, Any]) -> None:
        path = self.logs_dir / filename
        line = json.dumps(record, ensure_ascii=False, default=_json_default)
        with self._lock, path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")

    def log_call(
        self,
        *,
        stage: str,
        case_id: str,
        attempt: int,
        model: str,
        temperature: float,
        messages: list[dict[str, str]],
        raw_response: str | None,
        parse_ok: bool,
        error: str | None,
        prompt_tokens: int,
        completion_tokens: int,
        latency_ms: float,
        cached: bool,
        extra: dict[str, Any] | None = None,
    ) -> None:
        """Called once per attempt, BEFORE the caller returns -- success or not."""
        self._append(
            CALLS_LOG,
            {
                "run_id": self.run_id,
                "logged_at": utc_now_iso(),
                "stage": stage,
                "case_id": case_id,
                "attempt": attempt,
                "model": model,
                "temperature": temperature,
                "messages": messages,
                "raw_response": raw_response,
                "parse_ok": parse_ok,
                "error": error,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "latency_ms": round(latency_ms, 2),
                "cached": cached,
                **(extra or {}),
            },
        )

    def log_verdict(
        self,
        verdict: Verdict,
        *,
        stage: str = "run",
        mitigations: dict[str, bool] | None = None,
    ) -> None:
        """Record one verdict.

        ``mitigations`` is stored because it is an *input* to the run that
        cannot be recovered from the verdict itself. Without it, replay had to
        take the mitigation set from whatever config happened to be loaded at
        replay time, so replaying any ``--mitigations off`` run reported a
        MISMATCH against an intact log -- on precisely the ablation runs that
        produce the bias evidence.
        """
        record: dict[str, Any] = {
            "run_id": self.run_id,
            "logged_at": utc_now_iso(),
            "stage": stage,
            "verdict": json.loads(VerdictAdapter.dump_json(verdict).decode("utf-8")),
        }
        if mitigations is not None:
            record["mitigations"] = mitigations
        self._append(VERDICTS_LOG, record)

    # -- reading -----------------------------------------------------------

    @staticmethod
    def iter_records(path: str | Path) -> Iterator[dict[str, Any]]:
        p = Path(path)
        if not p.is_file():
            return
        with p.open("r", encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:  # pragma: no cover - defensive
                    raise ValueError(f"{p}:{lineno} is not valid JSON: {exc}") from exc

    @classmethod
    def read_verdicts(
        cls,
        logs_dir: str | Path,
        run_id: str,
        *,
        stage: str | None = None,
    ) -> list[Verdict]:
        """Rebuild the verdict stream for one run, de-duplicated by case id.

        Later records win, so a resumed run that re-judges a case supersedes the
        earlier attempt without the log ever being rewritten.
        """
        path = Path(logs_dir) / VERDICTS_LOG
        by_case: dict[str, Verdict] = {}
        for rec in cls.iter_records(path):
            if rec.get("run_id") != run_id:
                continue
            if stage is not None and rec.get("stage") != stage:
                continue
            verdict = VerdictAdapter.validate_python(rec["verdict"])
            by_case[f"{rec.get('stage')}::{verdict.case_id}"] = verdict
        return list(by_case.values())

    @classmethod
    def read_mitigations(
        cls, logs_dir: str | Path, run_id: str, *, stage: str | None = None
    ) -> dict[str, bool] | None:
        """The mitigation set recorded for a run, or ``None`` for older logs.

        Returning ``None`` rather than a default keeps replay honest about the
        difference between "the log says mitigations were off" and "the log
        predates this field", so the caller can fall back explicitly.
        """
        path = Path(logs_dir) / VERDICTS_LOG
        found: dict[str, bool] | None = None
        for rec in cls.iter_records(path):
            if rec.get("run_id") != run_id:
                continue
            if stage is not None and rec.get("stage") != stage:
                continue
            recorded = rec.get("mitigations")
            if isinstance(recorded, dict):
                found = {str(k): bool(v) for k, v in recorded.items()}
        return found

    @classmethod
    def read_calls(cls, logs_dir: str | Path, run_id: str) -> list[dict[str, Any]]:
        path = Path(logs_dir) / CALLS_LOG
        return [r for r in cls.iter_records(path) if r.get("run_id") == run_id]

    @classmethod
    def known_run_ids(cls, logs_dir: str | Path) -> list[str]:
        path = Path(logs_dir) / VERDICTS_LOG
        seen: list[str] = []
        for rec in cls.iter_records(path):
            rid = rec.get("run_id")
            if rid and rid not in seen:
                seen.append(rid)
        return seen


def redact_env_values(text: str) -> str:
    """Belt-and-braces: never let a key that leaked into a prompt reach a log."""
    out = text
    for env in ("GEMINI_API_KEY", "GROQ_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
        value = os.environ.get(env)
        if value and len(value) > 8 and value in out:
            out = out.replace(value, f"<{env}:redacted>")
    return out
