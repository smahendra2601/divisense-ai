"""Tier 3 — Agentic Orchestration: Intent Agent prompt + parsing.

Parses the raw user query into structured JSON: ``{intent:
forecast_single | dividend_qa | clarify | out_of_scope,
company_mention, question, horizon}``. A bare ticker input (regex +
yfinance validity) skips the LLM entirely → forecast_single, saving
quota. Company mentions are resolved deterministically via
``ticker_map.resolve()``; unresolvable → clarify; multi-company /
screener questions → out_of_scope.
"""
