"""Tests for src/intent.py (Tier 3 — Intent Agent).

Covers intent parsing across query phrasings: bare tickers (regex
shortcut, no LLM), forecast_single, dividend_qa, clarify on ambiguous
input, and out_of_scope for screener-type questions.
"""
