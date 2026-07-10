"""Tests for src/llm_router.py (Tier 2 — LLM routing, quota, cache, JSON).

Everything here runs offline: ``_call_provider`` (the one function that
actually talks to Groq/Gemini) is monkeypatched with a scripted fake, so
routing, fallback, quota prediction, caching, and the invoke_json
repair-retry are all exercised without real API keys or network calls.
A live smoke test lives in the module's own ``__main__`` block instead
(it needs real keys and isn't part of the automated suite).
"""

from __future__ import annotations

import pytest

from src import llm_router


@pytest.fixture(autouse=True)
def _isolated_state(tmp_path, monkeypatch):
    """Give every test a clean cache, quota tracker, and client registry."""
    from diskcache import Cache

    monkeypatch.setattr(llm_router, "_disk_cache", Cache(str(tmp_path)))
    monkeypatch.setattr(llm_router, "_memory_cache", {})
    monkeypatch.setattr(llm_router, "_quota", llm_router.QuotaTracker(llm_router._QUOTA_LIMITS))
    monkeypatch.setattr(llm_router, "_clients", {})
    yield


def _scripted_provider(responses):
    """Build a fake _call_provider(provider, prompt) from a responses dict.

    ``responses[provider]`` may be a string (returned) or an Exception
    instance/class (raised). Calls are recorded in ``calls``.
    """
    calls = []

    def fake(provider, prompt):
        calls.append((provider, prompt))
        outcome = responses[provider]
        if isinstance(outcome, Exception):
            raise outcome
        if isinstance(outcome, type) and issubclass(outcome, Exception):
            raise outcome("simulated failure")
        return outcome

    fake.calls = calls
    return fake


# ── cache key ───────────────────────────────────────────────────────
def test_cache_key_deterministic_and_task_scoped():
    a = llm_router._cache_key("hello", "reasoning")
    b = llm_router._cache_key("hello", "reasoning")
    c = llm_router._cache_key("hello", "long_context")
    assert a == b
    assert a != c  # same prompt, different task_type -> different key


# ── routing + fallback ─────────────────────────────────────────────
def test_reasoning_routes_to_groq_primary(monkeypatch):
    fake = _scripted_provider({"groq": "groq says hi"})
    monkeypatch.setattr(llm_router, "_call_provider", fake)

    assert llm_router.invoke("hi", "reasoning") == "groq says hi"
    assert fake.calls == [("groq", "hi")]


def test_long_context_routes_to_gemini_primary(monkeypatch):
    fake = _scripted_provider({"gemini": "gemini says hi"})
    monkeypatch.setattr(llm_router, "_call_provider", fake)

    assert llm_router.invoke("hi", "long_context") == "gemini says hi"
    assert fake.calls == [("gemini", "hi")]


def test_falls_back_to_secondary_on_primary_error(monkeypatch):
    fake = _scripted_provider({"groq": RuntimeError("429 rate limited"), "gemini": "backup ok"})
    monkeypatch.setattr(llm_router, "_call_provider", fake)

    result = llm_router.invoke("hi", "reasoning")
    assert result == "backup ok"
    assert [p for p, _ in fake.calls] == ["groq", "gemini"]


def test_both_providers_failing_raises_unavailable(monkeypatch):
    fake = _scripted_provider({"groq": RuntimeError("down"), "gemini": RuntimeError("also down")})
    monkeypatch.setattr(llm_router, "_call_provider", fake)

    with pytest.raises(llm_router.LLMUnavailableError):
        llm_router.invoke("hi", "reasoning")


def test_unknown_task_type_raises_value_error():
    with pytest.raises(ValueError):
        llm_router.invoke("hi", "not_a_real_task_type")  # type: ignore[arg-type]


# ── caching ─────────────────────────────────────────────────────────
def test_identical_call_is_served_from_cache_without_hitting_provider(monkeypatch):
    fake = _scripted_provider({"groq": "first response"})
    monkeypatch.setattr(llm_router, "_call_provider", fake)

    assert llm_router.invoke("same prompt", "reasoning") == "first response"
    assert llm_router.invoke("same prompt", "reasoning") == "first response"
    assert len(fake.calls) == 1  # second call served from cache


def test_different_prompts_are_not_conflated(monkeypatch):
    fake = _scripted_provider({"groq": "response"})
    monkeypatch.setattr(llm_router, "_call_provider", fake)

    llm_router.invoke("prompt A", "reasoning")
    llm_router.invoke("prompt B", "reasoning")
    assert len(fake.calls) == 2


def test_clear_cache_forces_recompute(monkeypatch):
    fake = _scripted_provider({"groq": "response"})
    monkeypatch.setattr(llm_router, "_call_provider", fake)

    llm_router.invoke("prompt", "reasoning")
    llm_router.clear_cache()
    llm_router.invoke("prompt", "reasoning")
    assert len(fake.calls) == 2


# ── quota tracker ───────────────────────────────────────────────────
def test_quota_tracker_counts_within_minute_and_day():
    tracker = llm_router.QuotaTracker({"p": {"rpm": 5, "rpd": 100}})
    now = 1_000_000.0
    for i in range(3):
        tracker.record("p", now=now + i)
    minute_count, day_count = tracker.counts("p", now=now + 3)
    assert minute_count == 3
    assert day_count == 3


def test_quota_tracker_would_exceed_rpm():
    tracker = llm_router.QuotaTracker({"p": {"rpm": 2, "rpd": 100}})
    now = 1_000_000.0
    tracker.record("p", now=now)
    tracker.record("p", now=now + 1)
    assert tracker.would_exceed("p", now=now + 2) is True


def test_quota_tracker_would_exceed_rpd():
    tracker = llm_router.QuotaTracker({"p": {"rpm": 1000, "rpd": 2}})
    now = 1_000_000.0
    tracker.record("p", now=now)
    tracker.record("p", now=now + 5)
    assert tracker.would_exceed("p", now=now + 10) is True


def test_quota_tracker_old_events_pruned_after_a_day():
    tracker = llm_router.QuotaTracker({"p": {"rpm": 5, "rpd": 1}})
    now = 1_000_000.0
    tracker.record("p", now=now)
    # A day and a bit later, that old event should no longer count.
    assert tracker.would_exceed("p", now=now + 86400 + 60) is False


def test_predicted_quota_breach_skips_straight_to_fallback(monkeypatch):
    fake = _scripted_provider({"gemini": "fallback response"})
    monkeypatch.setattr(llm_router, "_call_provider", fake)

    # Exhaust groq's RPM budget before ever calling invoke().
    for _ in range(llm_router._QUOTA_LIMITS["groq"]["rpm"]):
        llm_router._quota.record("groq")

    result = llm_router.invoke("hi", "reasoning")
    assert result == "fallback response"
    # groq should never have been called at all -- only gemini.
    assert fake.calls == [("gemini", "hi")]


# ── invoke_json ─────────────────────────────────────────────────────
def test_invoke_json_parses_clean_json(monkeypatch):
    fake = _scripted_provider({"groq": '{"answer": 4}'})
    monkeypatch.setattr(llm_router, "_call_provider", fake)

    assert llm_router.invoke_json("2+2?", '{"answer": <int>}') == {"answer": 4}


def test_invoke_json_strips_markdown_fences(monkeypatch):
    fake = _scripted_provider({"groq": '```json\n{"answer": 4}\n```'})
    monkeypatch.setattr(llm_router, "_call_provider", fake)

    assert llm_router.invoke_json("2+2?", '{"answer": <int>}') == {"answer": 4}


def test_invoke_json_repairs_broken_output_once(monkeypatch):
    # First call (original prompt) returns garbage; the repair-retry
    # call (a different, longer prompt) returns valid JSON. Cache keys
    # differ because the prompts differ, so both calls actually fire.
    responses = iter(["not json at all", '{"answer": 4}'])

    def fake(provider, prompt):
        return next(responses)

    monkeypatch.setattr(llm_router, "_call_provider", fake)

    result = llm_router.invoke_json("2+2?", '{"answer": <int>}')
    assert result == {"answer": 4}


def test_invoke_json_raises_after_failed_repair(monkeypatch):
    monkeypatch.setattr(llm_router, "_call_provider", lambda provider, prompt: "still not json")

    with pytest.raises(llm_router.LLMJSONParseError):
        llm_router.invoke_json("2+2?", '{"answer": <int>}')


# ── JSON helpers ────────────────────────────────────────────────────
def test_strip_markdown_fences_removes_json_fence():
    text = '```json\n{"a": 1}\n```'
    assert llm_router._strip_markdown_fences(text) == '{"a": 1}'


def test_strip_markdown_fences_removes_bare_fence():
    text = '```\n{"a": 1}\n```'
    assert llm_router._strip_markdown_fences(text) == '{"a": 1}'


def test_strip_markdown_fences_passes_through_unfenced_text():
    text = '{"a": 1}'
    assert llm_router._strip_markdown_fences(text) == '{"a": 1}'


def test_try_parse_json_returns_none_on_garbage():
    assert llm_router._try_parse_json("not json") is None


# ── content normalization (Gemini block-list responses) ─────────────
def test_extract_text_passes_through_plain_string():
    assert llm_router._extract_text('{"a": 1}') == '{"a": 1}'


def test_extract_text_joins_gemini_block_list():
    blocks = [
        {"type": "text", "text": '{"answer":', "extras": {"signature": "xyz"}},
        {"type": "text", "text": " 4}"},
    ]
    assert llm_router._extract_text(blocks) == '{"answer": 4}'


def test_extract_text_skips_non_text_blocks_and_handles_strings():
    blocks = ["plain, ", {"type": "thinking", "thinking": "hidden"}, {"type": "text", "text": "visible"}]
    assert llm_router._extract_text(blocks) == "plain, visible"


# ── client config errors (no real keys needed to prove the guard fires) ─
def test_missing_groq_key_raises_config_error(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with pytest.raises(llm_router.LLMConfigError):
        llm_router._build_groq_client()


def test_missing_gemini_key_raises_config_error(monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    with pytest.raises(llm_router.LLMConfigError):
        llm_router._build_gemini_client()
