"""Tier 1 — Data Acquisition: disk cache for fetched market data.

Wraps ``diskcache`` with a 1-hour TTL (``config.CACHE_TTL_SECONDS``)
keyed on ticker. Protects the demo from rate limits and repeated
fetches while keeping data effectively fresh.

Usage::

    from src.cache import disk_cached

    @disk_cached()
    def fetch_company_data(ticker: str) -> dict:
        ...

The decorator keys on the wrapped function's name plus its arguments,
so ``fetch_company_data("ITC")`` and ``fetch_company_data("SBIN")``
occupy distinct slots and each expires an hour after it was stored.
"""

from __future__ import annotations

from typing import Callable

from diskcache import Cache

from . import config

# Single shared on-disk cache for the whole application. diskcache is
# process- and thread-safe, so one instance is enough.
_cache = Cache(str(config.CACHE_DIR))


def disk_cached(ttl: int = config.CACHE_TTL_SECONDS) -> Callable:
    """Return a decorator that memoizes a function on disk for ``ttl`` seconds.

    Thin wrapper over ``diskcache.Cache.memoize`` so callers depend on
    our config rather than on diskcache directly. The default TTL is the
    project-wide 1-hour window from ``config.CACHE_TTL_SECONDS``.
    """

    def decorator(func: Callable) -> Callable:
        return _cache.memoize(expire=ttl)(func)

    return decorator


def clear() -> None:
    """Drop every cached entry. Handy for tests and manual cache busting."""
    _cache.clear()
