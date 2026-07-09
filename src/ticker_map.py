"""Tier 1 — Data Acquisition: deterministic company-name → NSE-ticker resolution.

Loads ``data/ticker_aliases.csv`` (alias, ticker, company_name) covering
common Indian company names/variants and exposes
``resolve(name_or_ticker) -> ticker | None``. Matching is
case-insensitive and whitespace-tolerant; an exact ticker match wins
before alias matching.

The Intent Agent only *proposes* a company mention; this module (plus a
yfinance validity check downstream) confirms it. The LLM never has final
say on a ticker.
"""

from __future__ import annotations

import csv
from functools import lru_cache

from . import config


def _normalize(text: str) -> str:
    """Lowercase, strip, and collapse internal whitespace for matching."""
    return " ".join(text.strip().lower().split())


@lru_cache(maxsize=1)
def _load() -> tuple[dict[str, str], dict[str, str]]:
    """Read the alias CSV once into two normalized lookup maps.

    Returns ``(ticker_by_norm, alias_by_norm)`` where keys are normalized
    strings and values are the canonical NSE ticker as written in the CSV.
    """
    ticker_by_norm: dict[str, str] = {}
    alias_by_norm: dict[str, str] = {}

    with open(config.TICKER_ALIASES_CSV, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            ticker = (row.get("ticker") or "").strip()
            alias = (row.get("alias") or "").strip()
            if not ticker:
                continue
            ticker_by_norm[_normalize(ticker)] = ticker
            if alias:
                alias_by_norm[_normalize(alias)] = ticker

    return ticker_by_norm, alias_by_norm


def resolve(name_or_ticker: str) -> str | None:
    """Resolve a company name or ticker to its canonical NSE ticker.

    Case-insensitive and whitespace-tolerant. An exact ticker match is
    tried first (so ``"INFY"`` resolves even though it is not listed as an
    alias), then the alias table. Returns ``None`` when nothing matches —
    the caller (Intent Agent) then routes to ``clarify``.
    """
    if not name_or_ticker or not name_or_ticker.strip():
        return None

    key = _normalize(name_or_ticker)
    ticker_by_norm, alias_by_norm = _load()

    if key in ticker_by_norm:  # exact ticker match wins first
        return ticker_by_norm[key]
    return alias_by_norm.get(key)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print('Usage: python -m src.ticker_map "<company name or ticker>"')
        print('Example: python -m src.ticker_map "coal india"')
        sys.exit(1)

    query = " ".join(sys.argv[1:])
    result = resolve(query)
    if result:
        print(result)
    else:
        print(f"No NSE ticker found for '{query}'. Try a different name or spelling.")
        sys.exit(1)
