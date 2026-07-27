"""Shared fixtures. Nothing here touches the network or reads an API key.

``FakeCompletion`` is injected through ``LLMClient.completion_fn`` -- the single
seam the whole design leaves for this -- so every test below drives the real
prompt builder, the real repair loop, the real mitigation protocol and the real
aggregator, with only the HTTP call replaced.
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.llm_client import (  # noqa: E402
    AuthenticationError,
    LLMClient,
    RateLimitError,
)
from src.suite_config import PipelineConfig, Rubric, load_config, load_rubric  # noqa: E402
from tests.fixtures import malformed_responses as mr  # noqa: E402

FAKE_MODEL = "gemini/gemini-3.5-flash"


def rate_limit_error(message: str = "429 rate limited (fake)") -> RateLimitError:
    """litellm's exception constructors require provider/model, so tests build
    them through helpers rather than repeating the signature everywhere."""
    return RateLimitError(message=message, llm_provider="gemini", model=FAKE_MODEL)


def auth_error(message: str = "invalid api key (fake)") -> AuthenticationError:
    return AuthenticationError(message=message, llm_provider="gemini", model=FAKE_MODEL)


# --------------------------------------------------------------------------
# Fake provider
# --------------------------------------------------------------------------


class _Msg:
    def __init__(self, content: str) -> None:
        self.content = content


class _Choice:
    def __init__(self, content: str, finish_reason: str = "stop") -> None:
        self.message = _Msg(content)
        self.finish_reason = finish_reason


class _Usage:
    def __init__(self, prompt_tokens: int, completion_tokens: int) -> None:
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens


class FakeResponse:
    def __init__(
        self,
        content: str,
        prompt_tokens: int = 800,
        completion_tokens: int = 200,
        finish_reason: str = "stop",
    ) -> None:
        self.choices = [_Choice(content, finish_reason)]
        self.usage = _Usage(prompt_tokens, completion_tokens)


@dataclass
class FakeCompletion:
    """Scripted provider. ``script`` entries are either strings (returned as the
    reply) or exception instances (raised)."""

    script: list[Any] = field(default_factory=list)
    default: Any = None
    calls: list[dict[str, Any]] = field(default_factory=list)

    def __call__(self, *, model: str, messages: list[dict[str, str]], **params: Any) -> Any:
        # Snapshot the message list: the caller keeps mutating it as it appends
        # repair turns, so storing the reference would make every recorded
        # request look like the final one.
        self.calls.append(
            {"model": model, "messages": [dict(m) for m in messages], "params": params}
        )
        idx = len(self.calls) - 1
        item = self.script[idx] if idx < len(self.script) else self.default
        if item is None:
            item = mr.valid_pointwise()
        if isinstance(item, BaseException):
            raise item
        if callable(item):
            item = item(model=model, messages=messages, **params)
        # A pre-built response passes through, so a script can set finish_reason
        # (e.g. "length") rather than only the reply text.
        if isinstance(item, FakeResponse):
            return item
        return FakeResponse(item)

    @property
    def last_messages(self) -> list[dict[str, str]]:
        return self.calls[-1]["messages"]


@dataclass
class FakeJudge:
    """A content-aware fake: decides pointwise vs pairwise from the system prompt,
    returns deterministic scores keyed on the judged content, and can be told to
    fail specific cases in specific ways."""

    criteria: list[str] = field(default_factory=lambda: list(mr.CRITERIA))
    malformed_markers: tuple[str, ...] = ()
    transient_markers: tuple[str, ...] = ()
    score_for: Callable[[str], int] | None = None
    winner_for: Callable[[str], str] | None = None
    calls: list[dict[str, Any]] = field(default_factory=list)
    _seen: dict[str, int] = field(default_factory=dict)

    def __call__(self, *, model: str, messages: list[dict[str, str]], **params: Any) -> Any:
        self.calls.append(
            {"model": model, "messages": [dict(m) for m in messages], "params": params}
        )
        system = messages[0]["content"]
        user = messages[-1]["content"]
        blob = "\n".join(m["content"] for m in messages)
        pairwise = "Two model outputs answer the same request" in system

        for marker in self.transient_markers:
            if marker in blob:
                key = f"transient::{marker}"
                self._seen[key] = self._seen.get(key, 0) + 1
                if self._seen[key] == 1:
                    raise rate_limit_error()

        for marker in self.malformed_markers:
            if marker in blob:
                key = f"malformed::{marker}"
                self._seen[key] = self._seen.get(key, 0) + 1
                if self._seen[key] == 1:
                    return FakeResponse(mr.TRUNCATED)

        if pairwise:
            winner = self.winner_for(user) if self.winner_for else self._default_winner(user)
            return FakeResponse(mr.valid_pairwise(winner))

        score = self.score_for(user) if self.score_for else self._default_score(user)
        return FakeResponse(
            mr.valid_pointwise(scores=dict.fromkeys(self.criteria, score), overall=score)
        )

    @staticmethod
    def _default_score(user: str) -> int:
        # Deterministic but content-dependent, so score distributions are not
        # degenerate and the aggregator's statistics have something to chew on.
        return 1 + (sum(ord(c) for c in user[-160:]) % 5)

    @staticmethod
    def _default_winner(user: str) -> str:
        return ["Model_A", "Model_B", "Tie"][sum(ord(c) for c in user[-160:]) % 3]


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------


@pytest.fixture
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture
def rubric() -> Rubric:
    return load_rubric(REPO_ROOT / "config" / "rubric.yaml")


@pytest.fixture
def config(tmp_path: Path) -> PipelineConfig:
    cfg = load_config(REPO_ROOT / "config" / "suite_config.yaml")
    cfg.run.results_dir = str(tmp_path / "results")
    cfg.run.logs_dir = str(tmp_path / "logs")
    cfg.run.cache_path = str(tmp_path / "cache")
    cfg.run.max_concurrent_judge_calls = 1
    return cfg


@pytest.fixture
def make_client(tmp_path: Path):
    def _make(fn: Callable[..., Any], **kwargs: Any) -> LLMClient:
        kwargs.setdefault("cache_path", tmp_path / "cache")
        kwargs.setdefault("retry_wait_initial", 0.0)
        return LLMClient(completion_fn=fn, **kwargs)

    return _make
