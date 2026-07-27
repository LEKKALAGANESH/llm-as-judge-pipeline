"""Client-side TPM pacing.

The limit that actually binds this workload is tokens per minute, not requests
per minute: one judge call costs ~4-6K tokens against a 12,000 TPM free tier.
Discovering that by being refused costs a round trip and a backoff per rejection
-- a measured pairwise run resolved only 3 of 15 pairs that way -- so the client
paces *before* sending.

These tests drive `_throttle` directly with a fake clock rather than sleeping,
because a pacing test that actually waits a minute is a test nobody runs.
"""

from __future__ import annotations

import pytest

from src.llm_client import TOKEN_BUDGET_SAFETY_FACTOR, LLMClient


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch):
    """A controllable monotonic clock. `sleep` advances it instead of waiting."""
    state = {"now": 1000.0, "slept": []}

    def fake_monotonic() -> float:
        return state["now"]

    def fake_sleep(seconds: float) -> None:
        state["slept"].append(seconds)
        state["now"] += seconds

    monkeypatch.setattr("src.llm_client.time.monotonic", fake_monotonic)
    monkeypatch.setattr("src.llm_client.time.sleep", fake_sleep)
    return state


def _client(tpm: int | None) -> LLMClient:
    return LLMClient(cache_path=None, enable_cache=False, tokens_per_minute=tpm)


def _budget(tpm: int) -> int:
    """The tokens the client will actually spend, after safety headroom."""
    return int(tpm * TOKEN_BUDGET_SAFETY_FACTOR)


def test_pacing_is_off_by_default() -> None:
    """The ceiling belongs to the provider and tier, not to this class. A default
    would silently pace the offline test suite against a fake provider."""
    assert _client(None).tokens_per_minute is None
    assert LLMClient(cache_path=None, enable_cache=False).tokens_per_minute is None


def test_a_safety_factor_is_applied_to_the_stated_ceiling() -> None:
    """Spending the full stated TPM is what produced the refusals: the token
    estimate can undershoot, and the provider's window is not empty when this
    process starts."""
    assert 0.0 < TOKEN_BUDGET_SAFETY_FACTOR < 1.0


def test_requests_inside_the_budget_never_sleep(clock) -> None:
    client = _client(12_000)
    half = _budget(12_000) // 2
    assert client._throttle(half) == 0.0
    assert client._throttle(half) == 0.0
    assert clock["slept"] == [], "two half-budget calls fit; neither should wait"


def test_exceeding_the_budget_waits_for_the_window_to_age_out(clock) -> None:
    client = _client(12_000)
    two_thirds = int(_budget(12_000) * 0.67)
    client._throttle(two_thirds)

    slept = client._throttle(two_thirds)  # 134% of budget -> must wait

    assert slept > 0, "the second call must be paced, not refused by the provider"
    # The oldest entry sits at the head of the window, so we wait out its 60s.
    assert slept == pytest.approx(60.0, abs=0.1)


def test_the_window_slides_rather_than_resetting(clock) -> None:
    """Entries older than 60s stop counting, so throughput recovers gradually
    instead of the budget refilling in one lump."""
    client = _client(12_000)
    nearly_all = _budget(12_000) - 1
    client._throttle(nearly_all)
    clock["now"] += 61.0  # the entry ages out
    assert client._throttle(nearly_all) == 0.0, "an expired entry must free its tokens"


def test_a_call_larger_than_the_whole_budget_is_sent_not_deadlocked(clock) -> None:
    """Sleeping forever for a request that can never fit would hang the run.
    Send it and let the provider decide."""
    client = _client(12_000)
    assert client._throttle(50_000) == 0.0


def test_every_retry_attempt_is_paced_not_just_the_first(clock, monkeypatch) -> None:
    """Throttling once outside the retry loop let one paced call emit a burst of
    unpaced retries -- which is exactly what pacing exists to prevent."""
    from src.llm_client import RateLimitError

    calls = {"n": 0}

    def always_429(**kwargs):
        calls["n"] += 1
        raise RateLimitError(message="slow down", llm_provider="groq", model="m")

    client = LLMClient(
        cache_path=None,
        enable_cache=False,
        tokens_per_minute=12_000,
        completion_fn=always_429,
        retry_wait_initial=0.0,
    )
    big = _budget(12_000)  # each attempt alone fills the window
    with pytest.raises(RateLimitError):
        client._invoke_with_retry(
            model="m",
            messages=[{"role": "user", "content": "x"}],
            params={},
            max_transient_retries=3,
            estimated_tokens=big,
        )
    assert calls["n"] == 3, "all three attempts should have been made"
    # Attempts 2 and 3 each had to wait for the window to clear.
    assert len(clock["slept"]) >= 2, "retries after the first must also be paced"


def test_estimate_counts_prompt_and_completion() -> None:
    messages = [{"role": "user", "content": "x" * 4_000}]
    # ~4 chars per token, plus the reserved completion budget.
    assert LLMClient._estimate_tokens(messages, max_tokens=512) == 1_000 + 512


def test_estimate_tolerates_missing_content() -> None:
    assert LLMClient._estimate_tokens([{"role": "user"}], max_tokens=100) == 100
