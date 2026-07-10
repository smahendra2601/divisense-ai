"""Tests for src/news.py (Tier 2 — recent-news context via Tavily).

The graceful-degradation contract is the important part: fetch_recent_news()
must return [] for bad input, a missing API key, a network error, a
timeout, or a malformed response — never raise. Split into two groups:
outer-logic tests (mock the cached fetch directly) and wire-level tests
(mock urllib.request.urlopen to exercise request construction + response
parsing for real).
"""

from __future__ import annotations

import io
import json
import urllib.error

import pytest

from src import cache as cache_module
from src import news


@pytest.fixture(autouse=True)
def _no_real_key(monkeypatch):
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)


# ── outer logic: fetch_recent_news() ─────────────────────────────────
def test_empty_ticker_returns_empty_list():
    assert news.fetch_recent_news("") == []
    assert news.fetch_recent_news("   ") == []


def test_no_api_key_returns_empty_list_without_calling_search(monkeypatch):
    calls = []
    monkeypatch.setattr(
        news, "_fetch_recent_news_cached", lambda query, k: calls.append(1) or []
    )
    result = news.fetch_recent_news("ITC", "ITC Limited")
    assert result == []
    assert calls == []  # never even attempted — no key means no network call


def test_delegates_to_cached_fetch_when_key_present(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-fake-key")
    seen = {}

    def fake_cached(query, k):
        seen["query"] = query
        seen["k"] = k
        return [{"title": "T", "url": "u", "snippet": "s", "score": 0.9}]

    monkeypatch.setattr(news, "_fetch_recent_news_cached", fake_cached)
    result = news.fetch_recent_news("ITC", "ITC Limited", k=3)

    assert result == [{"title": "T", "url": "u", "snippet": "s", "score": 0.9}]
    assert seen["query"] == "ITC Limited dividend announcement"
    assert seen["k"] == 3


def test_falls_back_to_ticker_when_no_company_name(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-fake-key")
    seen = {}
    monkeypatch.setattr(
        news, "_fetch_recent_news_cached",
        lambda query, k: seen.setdefault("query", query) or [],
    )
    news.fetch_recent_news("ITC")
    assert seen["query"] == "ITC dividend announcement"


@pytest.mark.parametrize(
    "exc",
    [
        urllib.error.URLError("network down"),
        TimeoutError("timed out"),
        ValueError("bad response"),
        KeyError("TAVILY_API_KEY"),
        json.JSONDecodeError("bad json", "", 0),
    ],
)
def test_search_failures_degrade_to_empty_list(monkeypatch, exc):
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-fake-key")

    def boom(query, k):
        raise exc

    monkeypatch.setattr(news, "_fetch_recent_news_cached", boom)
    assert news.fetch_recent_news("ITC", "ITC Limited") == []


def test_unexpected_exception_is_not_swallowed(monkeypatch):
    # Only the documented failure modes degrade silently; anything else
    # (a programming bug) should surface, not be hidden as "no news".
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-fake-key")
    monkeypatch.setattr(
        news, "_fetch_recent_news_cached",
        lambda query, k: (_ for _ in ()).throw(RuntimeError("bug")),
    )
    with pytest.raises(RuntimeError):
        news.fetch_recent_news("ITC", "ITC Limited")


# ── wire-level: request construction + response parsing ─────────────
class _FakeResponse:
    def __init__(self, payload: dict):
        self._body = json.dumps(payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self):
        return self._body


@pytest.fixture
def isolated_news_cache(tmp_path, monkeypatch):
    from diskcache import Cache

    monkeypatch.setattr(cache_module, "_cache", Cache(str(tmp_path)))
    # Rebuild the decorated function against the isolated cache.
    monkeypatch.setattr(
        news, "_fetch_recent_news_cached",
        cache_module.disk_cached(ttl=999)(news._fetch_recent_news_cached.__wrapped__),
    )
    yield


def test_search_tavily_builds_expected_request(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-secret")
    captured = {}

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["method"] = request.get_method()
        captured["headers"] = {k.lower(): v for k, v in request.headers.items()}
        captured["body"] = json.loads(request.data.decode("utf-8"))
        captured["timeout"] = timeout
        return _FakeResponse({"results": []})

    monkeypatch.setattr(news.urllib.request, "urlopen", fake_urlopen)
    news._search_tavily("ITC Limited dividend announcement", 5)

    assert captured["method"] == "POST"
    assert captured["url"] == news.config.TAVILY_SEARCH_URL
    assert captured["headers"]["authorization"] == "Bearer tvly-secret"
    assert captured["headers"]["content-type"] == "application/json"
    assert captured["body"]["query"] == "ITC Limited dividend announcement"
    assert captured["body"]["max_results"] == 5
    assert captured["body"]["time_range"] == news.config.NEWS_TIME_RANGE
    assert captured["timeout"] == news.config.NEWS_TIMEOUT_SECONDS


def test_fetch_recent_news_cached_parses_and_shapes_results(monkeypatch, isolated_news_cache):
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-secret")
    api_response = {
        "results": [
            {"title": "ITC declares interim dividend", "url": "https://x/1", "content": "A" * 600, "score": 0.91},
            {"title": "Unrelated", "url": "https://x/2", "content": "short", "score": 0.4},
            {"title": "", "url": "https://x/3", "content": "", "score": 0.1},  # both empty -> dropped
        ]
    }
    monkeypatch.setattr(
        news.urllib.request, "urlopen", lambda request, timeout: _FakeResponse(api_response)
    )

    hits = news.fetch_recent_news("ITC", "ITC Limited")

    assert len(hits) == 2  # the empty title+content result is filtered out
    assert hits[0]["title"] == "ITC declares interim dividend"
    assert hits[0]["url"] == "https://x/1"
    assert len(hits[0]["snippet"]) == 500  # truncated
    assert hits[0]["score"] == 0.91


def test_transient_empty_response_is_retried_once(monkeypatch, isolated_news_cache):
    # Tavily intermittently returns 0 results; one immediate retry recovers.
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-secret")
    responses = [
        {"results": []},
        {"results": [{"title": "T", "url": "u", "content": "c", "score": 0.5}]},
    ]
    calls = []

    def fake_urlopen(request, timeout):
        calls.append(1)
        return _FakeResponse(responses[len(calls) - 1])

    monkeypatch.setattr(news.urllib.request, "urlopen", fake_urlopen)
    hits = news.fetch_recent_news("ITC", "ITC Limited")

    assert len(calls) == 2  # empty first response triggered exactly one retry
    assert [h["title"] for h in hits] == ["T"]


def test_persistently_empty_response_is_not_cached(monkeypatch, isolated_news_cache):
    # An empty result must not be memoized for the full TTL: the next query
    # should hit the network again (empty raises internally; degrades to []).
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-secret")
    calls = []

    def fake_urlopen(request, timeout):
        calls.append(1)
        return _FakeResponse({"results": []})

    monkeypatch.setattr(news.urllib.request, "urlopen", fake_urlopen)

    assert news.fetch_recent_news("ITC", "ITC Limited") == []
    assert news.fetch_recent_news("ITC", "ITC Limited") == []
    assert len(calls) == 4  # 2 calls per fetch (retry), nothing served from cache


def test_identical_query_is_cached_without_a_second_request(monkeypatch, isolated_news_cache):
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-secret")
    calls = []

    def fake_urlopen(request, timeout):
        calls.append(1)
        return _FakeResponse({"results": [{"title": "T", "url": "u", "content": "c", "score": 0.5}]})

    monkeypatch.setattr(news.urllib.request, "urlopen", fake_urlopen)

    news.fetch_recent_news("ITC", "ITC Limited")
    news.fetch_recent_news("ITC", "ITC Limited")

    assert len(calls) == 1  # second call served from cache
