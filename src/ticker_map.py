"""Tier 1 — Data Acquisition: deterministic company-name → NSE-ticker resolution.

Loads ``data/ticker_aliases.csv`` (alias, ticker, company_name) covering
~100 common names/variants and exposes ``resolve(name_or_ticker) ->
ticker | None``. The Intent Agent proposes a company mention; this
module (plus a yfinance validity check) confirms it. The LLM never has
final say on a ticker.
"""
