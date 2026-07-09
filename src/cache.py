"""Tier 1 — Data Acquisition: disk cache for fetched market data.

Wraps ``diskcache`` with a 1-hour TTL (``config.CACHE_TTL_SECONDS``)
keyed on ticker. Protects the demo from rate limits and repeated
fetches while keeping data effectively fresh.
"""
