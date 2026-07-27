"""Transport layer: one provider-agnostic call, with a cache, a quota guard and
transient-only retries.

This module owns exactly one of the two error layers described in the design.

**Outer layer (here): transient failures.** ``tenacity`` retries rate limits,
connection errors and 5xx. It cannot and must not be used to repair malformed
JSON, because tenacity re-invokes the wrapped callable with *identical*
arguments -- a "repair" implemented that way is just a re-issue of the same
prompt that failed, and a test of it passes whenever the mock's second canned
response happens to be well formed. The repair layer lives in ``judge.py`` and
mutates the message list.

Authentication, permission and bad-request failures are explicitly *not*
retried. Inside a per-case loop those retries multiply one misconfiguration
across the whole suite while delaying the only useful signal; instead the call
fails immediately with a message naming which provider's key to check.

The disk cache is keyed on ``sha256(model, messages, params)``. On a free tier
this is the single highest-leverage cost control: development iteration burns
far more quota than the final run, and a cached re-run costs nothing.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from .errors import FatalProviderError, QuotaExceeded
from .suite_config import api_key_env_for

# --------------------------------------------------------------------------
# Provider exception classes.
#
# Imported from litellm when available, with local stand-ins otherwise so the
# entire unit/regression test suite runs without litellm (and therefore without
# a network stack) installed.
# --------------------------------------------------------------------------

try:  # pragma: no cover - trivial import branch
    from litellm import (
        APIConnectionError,
        AuthenticationError,
        BadRequestError,
        ContentPolicyViolationError,
        ContextWindowExceededError,
        InternalServerError,
        NotFoundError,
        RateLimitError,
        ServiceUnavailableError,
        Timeout,
        UnsupportedParamsError,
    )

    LITELLM_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only where litellm is absent

    class _Shim(Exception):
        def __init__(self, message: str = "", *args: Any, **kwargs: Any) -> None:
            super().__init__(message)

    class APIConnectionError(_Shim): ...

    class AuthenticationError(_Shim): ...

    class BadRequestError(_Shim): ...

    class ContentPolicyViolationError(_Shim): ...

    class ContextWindowExceededError(_Shim): ...

    class InternalServerError(_Shim): ...

    class NotFoundError(_Shim): ...

    class RateLimitError(_Shim): ...

    class ServiceUnavailableError(_Shim): ...

    class Timeout(_Shim): ...

    class UnsupportedParamsError(_Shim): ...

    LITELLM_AVAILABLE = False


TRANSIENT_ERRORS: tuple[type[BaseException], ...] = (
    RateLimitError,
    APIConnectionError,
    InternalServerError,
    ServiceUnavailableError,
    Timeout,
)

FATAL_ERRORS: tuple[type[BaseException], ...] = (
    AuthenticationError,
    BadRequestError,
    NotFoundError,
    ContextWindowExceededError,
    ContentPolicyViolationError,
    UnsupportedParamsError,
)


@dataclass
class LLMResponse:
    text: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0
    cached: bool = False
    structured_output_used: bool = False
    finish_reason: str | None = None


@dataclass
class ResponseCache:
    """Disk-backed cache. One JSON file per key; misses are silent."""

    path: Path
    enabled: bool = True
    hits: int = 0
    misses: int = 0

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        if self.enabled:
            self.path.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def make_key(model: str, messages: list[dict[str, str]], params: dict[str, Any]) -> str:
        payload = json.dumps(
            {"model": model, "messages": messages, "params": params},
            sort_keys=True,
            ensure_ascii=False,
            default=str,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def get(self, key: str) -> dict[str, Any] | None:
        if not self.enabled:
            return None
        f = self.path / f"{key}.json"
        if not f.is_file():
            self.misses += 1
            return None
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            self.misses += 1
            return None
        self.hits += 1
        return data

    def put(self, key: str, value: dict[str, Any]) -> None:
        if not self.enabled:
            return
        tmp = self.path / f"{key}.json.tmp"
        tmp.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
        tmp.replace(self.path / f"{key}.json")


def _default_completion_fn() -> Callable[..., Any]:
    """Imported lazily so importing this package does not pull in litellm."""
    import litellm

    # Providers disagree about which optional params they accept (Groq takes
    # reasoning_effort, Gemini does not). Dropping unsupported params is the
    # documented litellm behaviour for exactly this case and is preferable to
    # a per-provider branch that goes stale.
    litellm.drop_params = True
    return litellm.completion


#: Model-id fragments that identify a reasoning model. Only these accept
#: ``reasoning_effort`` / ``include_reasoning``; Groq rejects them with a 400
#: on its non-reasoning models, and ``litellm.drop_params`` does not save us
#: because the provider -- not litellm -- is the one refusing.
_REASONING_MODEL_MARKERS = ("gpt-oss", "o1", "o3", "o4")

#: Fraction of the stated TPM ceiling the client will actually spend. Headroom
#: for token-estimation error and for a provider window that this process did
#: not start empty.
TOKEN_BUDGET_SAFETY_FACTOR = 0.75


def _is_reasoning_model(model: str) -> bool:
    lowered = model.lower()
    return any(marker in lowered for marker in _REASONING_MODEL_MARKERS)


def _extract_text(response: Any) -> tuple[str, str | None]:
    try:
        choice = response.choices[0]
    except (AttributeError, IndexError, KeyError, TypeError):
        return "", None
    message = getattr(choice, "message", None)
    if message is None and isinstance(choice, dict):
        message = choice.get("message")
    content = getattr(message, "content", None)
    if content is None and isinstance(message, dict):
        content = message.get("content")
    finish = getattr(choice, "finish_reason", None)
    if finish is None and isinstance(choice, dict):
        finish = choice.get("finish_reason")
    return (content or ""), finish


def _extract_usage(response: Any) -> tuple[int, int]:
    usage = getattr(response, "usage", None)
    if usage is None and isinstance(response, dict):
        usage = response.get("usage")
    if usage is None:
        return 0, 0

    def _get(name: str) -> int:
        val = getattr(usage, name, None)
        if val is None and isinstance(usage, dict):
            val = usage.get(name)
        try:
            return int(val or 0)
        except (TypeError, ValueError):
            return 0

    return _get("prompt_tokens"), _get("completion_tokens")


@dataclass
class LLMClient:
    """Provider-agnostic completion with cache, quota guard and retries.

    ``completion_fn`` is the single test seam: every test in this repository
    injects a fake here rather than monkeypatching a module global, which is why
    the whole CI path runs with no API keys and no network.
    """

    cache_path: str | Path | None = ".cache/llm_responses"
    enable_cache: bool = True
    max_calls: int | None = None
    max_cost_usd: float | None = None
    completion_fn: Callable[..., Any] | None = None
    price_lookup: Callable[[str], tuple[float, float]] | None = None
    retry_wait_initial: float = 1.0
    """Backoff seed for transient retries. Tests set it to 0 so the retry path is
    exercised without sleeping; production leaves it at 1s with jitter."""
    retry_wait_max: float = 45.0
    """Ceiling for a single backoff sleep.

    Must exceed the longest ``retryDelay`` a provider actually asks for, or the
    retry gives up while still inside the window it was told to wait out. The
    Gemini free tier meters this model at 5 requests/minute and returns
    ``retryDelay: 34s``; the previous 20s ceiling could not survive it, so a
    free-tier run lost ~40% of its cases to rate limiting."""
    tokens_per_minute: int | None = None
    """Client-side TPM budget, paced *before* sending rather than after a 429.

    This is the limit that actually binds this workload: one judge call costs
    ~4-6K tokens against a 12,000 TPM free tier, so roughly two calls per
    minute. Relying on reactive backoff alone meant every rejection burned a
    round trip and its retry budget -- a measured pairwise run resolved only 3
    of 15 pairs, and a report built on 3 pairs is a much weaker claim than the
    same code makes on 15.

    Defaults to ``None`` (no pacing) because the ceiling is a property of the
    provider and tier, not of this class, and because pacing a fake provider
    would make the offline test suite sleep for no reason. ``main.py`` supplies
    the configured value for real runs."""

    api_calls: int = field(default=0, init=False)
    cached_calls: int = field(default=0, init=False)
    prompt_tokens: int = field(default=0, init=False)
    completion_tokens: int = field(default=0, init=False)
    cost_usd: float = field(default=0.0, init=False)
    _cache: ResponseCache = field(init=False)
    _fn: Callable[..., Any] | None = field(default=None, init=False)
    _lock: threading.Lock = field(init=False)

    def __post_init__(self) -> None:
        self._cache = ResponseCache(
            path=Path(self.cache_path or ".cache/llm_responses"),
            enabled=bool(self.enable_cache and self.cache_path),
        )
        self._fn = self.completion_fn
        # Counters are read by the quota guard and mutated by every call; the
        # runner may drive up to max_concurrent_judge_calls of them at once.
        self._lock = threading.Lock()
        self._token_window = deque()

    # -- rate limiting -----------------------------------------------------

    def _throttle(self, estimated_tokens: int) -> float:
        """Sleep until ``estimated_tokens`` fit inside the trailing-minute budget.

        A sliding window over (timestamp, tokens) for the last 60s. If the new
        request would push the total past the ceiling, wait exactly long enough
        for the oldest entries to age out -- which is strictly cheaper than
        sending it, being refused, and paying a backoff anyway.

        Returns the seconds actually slept, so tests can assert pacing without
        depending on wall-clock timing.
        """
        if not self.tokens_per_minute:
            return 0.0
        # Pace against a fraction of the stated ceiling. Two things make the
        # naive budget too optimistic: the ~4-chars-per-token estimate can
        # undershoot, and the provider's window is already partly consumed by
        # whatever ran before this process started, while ours begins empty.
        # Leaving headroom costs a little throughput and avoids the refusals
        # that cost far more.
        budget = max(1, int(self.tokens_per_minute * TOKEN_BUDGET_SAFETY_FACTOR))
        # A single call larger than the whole budget can never fit; send it and
        # let the provider decide rather than sleeping forever.
        estimated = min(estimated_tokens, budget)
        slept = 0.0
        while True:
            with self._lock:
                now = time.monotonic()
                while self._token_window and now - self._token_window[0][0] >= 60.0:
                    self._token_window.popleft()
                used = sum(tokens for _, tokens in self._token_window)
                if used + estimated <= budget or not self._token_window:
                    self._token_window.append((now, estimated))
                    return slept
                wait = 60.0 - (now - self._token_window[0][0])
            wait = max(0.05, min(wait, 60.0))
            time.sleep(wait)
            slept += wait

    @staticmethod
    def _estimate_tokens(messages: list[dict[str, str]], max_tokens: int) -> int:
        """Rough prompt+completion size. ~4 characters per token is close enough
        for pacing; the window self-corrects because it is re-measured every
        call and a slight overestimate only costs a little throughput."""
        chars = sum(len(m.get("content") or "") for m in messages)
        return chars // 4 + max_tokens

    # -- accounting --------------------------------------------------------

    @property
    def calls_remaining(self) -> int | None:
        return None if self.max_calls is None else max(0, self.max_calls - self.api_calls)

    def _account(self, model: str, prompt_tokens: int, completion_tokens: int) -> None:
        with self._lock:
            self.prompt_tokens += prompt_tokens
            self.completion_tokens += completion_tokens
            if self.price_lookup is not None:
                in_price, out_price = self.price_lookup(model)
                self.cost_usd += (
                    prompt_tokens * in_price + completion_tokens * out_price
                ) / 1_000_000

    def _check_quota(self, model: str) -> None:
        """Checked BEFORE the request is issued, so the breaker bounds real
        provider usage instead of reporting the overrun afterwards."""
        with self._lock:
            calls, cost = self.api_calls, self.cost_usd
        if self.max_calls is not None and calls >= self.max_calls:
            raise QuotaExceeded(
                f"max_calls_per_run={self.max_calls} reached before calling {model!r}. "
                "Raise run.max_calls_per_run, or re-run with --resume to continue from the log."
            )
        if self.max_cost_usd is not None and cost >= self.max_cost_usd:
            raise QuotaExceeded(
                f"max_cost_usd={self.max_cost_usd} reached (estimated ${cost:.4f}) "
                f"before calling {model!r}."
            )

    # -- the call ----------------------------------------------------------

    def complete(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        temperature: float = 0.0,
        max_tokens: int = 2048,
        response_format: dict[str, Any] | None = None,
        timeout: int | None = 120,
        max_transient_retries: int = 3,
        extra_params: dict[str, Any] | None = None,
        bypass_cache: bool = False,
    ) -> LLMResponse:
        params: dict[str, Any] = {
            "temperature": temperature,
            "max_tokens": max_tokens,
            "timeout": timeout,
        }
        if response_format is not None:
            params["response_format"] = response_format
        if extra_params:
            params.update(extra_params)

        # gpt-oss-* are reasoning models and reasoning tokens bill as output.
        # At default effort they can silently halve a day's free-tier budget.
        # Gated on the model, not the provider: Groq also serves non-reasoning
        # models (llama-3.3-70b) that reject these parameters outright with a
        # 400, which is fatal rather than retryable.
        if _is_reasoning_model(model):
            params.setdefault("reasoning_effort", "low")
            params.setdefault("include_reasoning", False)

        cache_key = ResponseCache.make_key(model, messages, params)
        cached = None if bypass_cache else self._cache.get(cache_key)
        if cached is not None:
            with self._lock:
                self.cached_calls += 1
            return LLMResponse(
                text=cached["text"],
                model=model,
                prompt_tokens=cached.get("prompt_tokens", 0),
                completion_tokens=cached.get("completion_tokens", 0),
                latency_ms=0.0,
                cached=True,
                structured_output_used=cached.get("structured_output_used", False),
                finish_reason=cached.get("finish_reason"),
            )

        self._check_quota(model)
        response = self._invoke_with_retry(
            model=model,
            messages=messages,
            params=params,
            max_transient_retries=max_transient_retries,
            estimated_tokens=self._estimate_tokens(messages, max_tokens),
        )
        if bypass_cache:
            # Deliberately not written back either: a test-retest run must not
            # poison the cache with one sample of a distribution we are measuring.
            return response
        self._cache.put(
            cache_key,
            {
                "text": response.text,
                "prompt_tokens": response.prompt_tokens,
                "completion_tokens": response.completion_tokens,
                "structured_output_used": response.structured_output_used,
                "finish_reason": response.finish_reason,
            },
        )
        return response

    def _invoke_with_retry(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        params: dict[str, Any],
        max_transient_retries: int,
        estimated_tokens: int = 0,
    ) -> LLMResponse:
        """`tenacity.retry` applied at call time so the attempt count stays
        config-driven. Transient classes only -- see FATAL_ERRORS."""

        def _once(call_params: dict[str, Any]) -> LLMResponse:
            fn = self._fn
            if fn is None:
                fn = _default_completion_fn()
                self._fn = fn
            # Pace every *attempt*, not just the first. Throttling once outside
            # this function let a single paced call emit a burst of unpaced
            # retries, which is precisely what the pacing exists to prevent --
            # measured, it left the refusal rate at 71%.
            self._throttle(estimated_tokens)
            with self._lock:
                self.api_calls += 1
            started = time.perf_counter()
            try:
                raw = fn(model=model, messages=messages, **call_params)
            except TRANSIENT_ERRORS:
                raise
            except FATAL_ERRORS as exc:
                raise FatalProviderError(
                    f"non-retryable provider error calling {model!r}: {type(exc).__name__}: {exc}. "
                    f"Check that {api_key_env_for(model)} is set and valid for this model, and "
                    "that the model id still exists (both providers deprecated models recently)."
                ) from exc
            latency_ms = (time.perf_counter() - started) * 1000.0
            text, finish = _extract_text(raw)
            ptok, ctok = _extract_usage(raw)
            self._account(model, ptok, ctok)
            return LLMResponse(
                text=text,
                model=model,
                prompt_tokens=ptok,
                completion_tokens=ctok,
                latency_ms=latency_ms,
                cached=False,
                structured_output_used="response_format" in call_params,
                finish_reason=finish,
            )

        runner = retry(
            retry=retry_if_exception_type(TRANSIENT_ERRORS),
            stop=stop_after_attempt(max(1, max_transient_retries)),
            wait=wait_exponential_jitter(initial=self.retry_wait_initial, max=self.retry_wait_max),
            reraise=True,
        )(_once)

        try:
            return runner(params)
        except FatalProviderError as exc:
            # One narrow, deterministic fallback: a provider that rejects the
            # structured-output parameter is retried once without it, because
            # the repair loop in judge.py can still recover a plain-text JSON
            # response. Any other bad-request stays fatal.
            if "response_format" in params and _mentions_structured_output(str(exc)):
                degraded = {k: v for k, v in params.items() if k != "response_format"}
                return runner(degraded)
            raise


def _mentions_structured_output(message: str) -> bool:
    lowered = message.lower()
    return any(
        token in lowered
        for token in ("response_format", "response_schema", "json_schema", "responseschema")
    )
