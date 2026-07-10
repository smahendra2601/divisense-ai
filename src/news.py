"""Tier 2 — Knowledge: recent-news context for the Forecast Agent (Tavily).

``fetch_recent_news(ticker, company_name, k)`` searches for recent
dividend-relevant news via the Tavily search API and returns scored
snippets — qualitative context (special-dividend rumors, regulatory
risk, M&A) that a once-a-year annual report and a metrics-only prompt
can't see.

This source is entirely **optional**: no ``TAVILY_API_KEY`` in ``.env``,
a network error, or a timeout all degrade to ``[]`` rather than raising,
exactly like ``rag.retrieve()``. It spends zero LLM calls, so it never
touches the §6 per-query budget. News snippets are for the Forecast
Agent's *reasoning*, never a source of numbers — the prompt in
``graph.py`` says so explicitly, and the Critic's existing "every number
must trace to the metrics" rule enforces it.
"""

from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.request

from dotenv import load_dotenv

from . import config
from .cache import disk_cached

logger = logging.getLogger(__name__)

# Populate os.environ from .env without clobbering vars already set.
# Safe to call again if llm_router already did (override=False, idempotent).
load_dotenv(dotenv_path=config.PROJECT_ROOT / ".env", override=False)


def _search_tavily(query: str, k: int) -> list[dict]:
    """POST to the Tavily search API. Raises on any failure; never called
    without an API key (the caller checks first)."""
    api_key = os.environ["TAVILY_API_KEY"]
    payload = json.dumps(
        {
            "query": query,
            "max_results": k,
            "search_depth": "basic",
            "topic": "general",
            "time_range": config.NEWS_TIME_RANGE,
        }
    ).encode("utf-8")

    request = urllib.request.Request(
        config.TAVILY_SEARCH_URL,
        data=payload,
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )
    with urllib.request.urlopen(request, timeout=config.NEWS_TIMEOUT_SECONDS) as resp:
        body = json.loads(resp.read().decode("utf-8"))

    return body.get("results") or []


@disk_cached(ttl=config.NEWS_CACHE_TTL_SECONDS)
def _fetch_recent_news_cached(query: str, k: int) -> list[dict]:
    """Cached Tavily call, keyed on the exact query text. Raises on failure
    (diskcache.memoize only caches successful returns, so a failure is
    retried on the next call rather than cached)."""
    raw_results = _search_tavily(query, k)
    return [
        {
            "title": r.get("title"),
            "url": r.get("url"),
            "snippet": (r.get("content") or "")[:500],
            "score": r.get("score"),
        }
        for r in raw_results
        if r.get("title") or r.get("content")
    ]


def fetch_recent_news(ticker: str, company_name: str | None = None, k: int = config.NEWS_MAX_RESULTS) -> list[dict]:
    """Return up to ``k`` recent dividend-relevant news snippets for a company.

    Each item is ``{title, url, snippet, score}``, most-relevant first.
    Returns ``[]`` gracefully whenever there is nothing to return — no
    ``TAVILY_API_KEY``, a network error, a timeout, or a malformed
    response — so callers can treat news context as strictly optional.
    """
    if not ticker or not ticker.strip():
        return []
    if not os.environ.get("TAVILY_API_KEY"):
        logger.info("news: no TAVILY_API_KEY set; returning empty context for %s", ticker)
        return []

    name = (company_name or ticker).strip()
    query = f"{name} dividend announcement"

    try:
        return _fetch_recent_news_cached(query, k)
    except (urllib.error.URLError, TimeoutError, ValueError, KeyError, json.JSONDecodeError) as exc:
        logger.warning("news: search failed for %s (%s); returning empty context", ticker, exc)
        return []


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

    if len(sys.argv) < 2:
        print("Usage: python -m src.news <TICKER> [company name]")
        sys.exit(1)

    tk = sys.argv[1]
    name_arg = " ".join(sys.argv[2:]) if len(sys.argv) > 2 else tk
    hits = fetch_recent_news(tk, name_arg)
    print(f"\n{len(hits)} article(s) for {tk.upper()} ({name_arg!r})\n")
    print(json.dumps(hits, indent=2, ensure_ascii=False))
