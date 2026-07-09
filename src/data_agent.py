"""Tier 1 — Data Acquisition: live yfinance fetcher.

Given an NSE ticker (e.g. ``ITC``), appends ``.NS`` and fetches dividend
history, current price, annual income statement, balance sheet, and
cash flow via yfinance. Returns a normalized dict including a
``data_timestamp``. Raises a clear ``InvalidTickerError`` for bad
tickers. Fetch-on-demand — freshness by design.
"""
