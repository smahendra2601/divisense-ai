"""Tests for src/intent.py (Tier 3 — Intent Agent, graph node 0).

The LLM is always mocked (via ``intent.llm_router.invoke_json``) so these
are deterministic and offline. ``_ticker_exists`` is stubbed to False so
no test ever hits the network — every ticker used here resolves through
the shipped alias CSV instead. Ticker resolution itself is real
(``ticker_map`` against data/ticker_aliases.csv), which is the point:
the code, not the LLM, decides the ticker.
"""

from __future__ import annotations

import pytest

from src import intent


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    # Guarantee no accidental yfinance calls; CSV resolution covers the tests.
    monkeypatch.setattr(intent, "_ticker_exists", lambda candidate: False)


def _mock_llm(monkeypatch, response):
    """Point invoke_json at a fake and return the recorder to assert on it."""
    calls = []

    def fake(prompt, schema_hint, task_type="reasoning"):
        calls.append({"prompt": prompt, "schema_hint": schema_hint, "task_type": task_type})
        if isinstance(response, Exception):
            raise response
        return response

    monkeypatch.setattr(intent.llm_router, "invoke_json", fake)
    return calls


# ── bare ticker short-circuit (no LLM) ───────────────────────────────
def test_bare_ticker_short_circuits_without_llm(monkeypatch):
    calls = _mock_llm(monkeypatch, {"should": "not be used"})

    result = intent.parse_query("COALINDIA")

    assert result["intent"] == "forecast_single"
    assert result["ticker"] == "COALINDIA"
    assert result["company_mention"] == "COALINDIA"
    assert calls == []  # the LLM was never called


def test_bare_ticker_is_case_insensitive(monkeypatch):
    calls = _mock_llm(monkeypatch, {})
    result = intent.parse_query("itc")
    assert result["intent"] == "forecast_single"
    assert result["ticker"] == "ITC"
    assert calls == []


# ── LLM path: dividend_qa ────────────────────────────────────────────
def test_dividend_qa_infosys_next_quarter(monkeypatch):
    _mock_llm(
        monkeypatch,
        {
            "intent": "dividend_qa",
            "company_mention": "Infosys",
            "question": "Will Infosys increase its dividend next quarter?",
            "horizon": "next quarter",
        },
    )

    result = intent.parse_query("Will Infosys increase its dividend next quarter?")

    assert result["intent"] == "dividend_qa"
    assert result["ticker"] == "INFY"  # resolved by ticker_map, not the LLM
    assert result["horizon"] == "next quarter"
    assert result["company_mention"] == "Infosys"
    assert "Infosys" in result["question"]


# ── LLM path: forecast_single ────────────────────────────────────────
def test_forecast_single_itc_next_year(monkeypatch):
    _mock_llm(
        monkeypatch,
        {
            "intent": "forecast_single",
            "company_mention": "ITC",
            "question": "Forecast ITC's dividend for next year",
            "horizon": "next year",
        },
    )

    result = intent.parse_query("Forecast ITC's dividend for next year")

    assert result["intent"] == "forecast_single"
    assert result["ticker"] == "ITC"
    assert result["horizon"] == "next year"


def test_forecast_tcs_fy27(monkeypatch):
    _mock_llm(
        monkeypatch,
        {
            "intent": "forecast_single",
            "company_mention": "TCS",
            "question": "What dividend will TCS pay in FY27?",
            "horizon": "FY27",
        },
    )

    result = intent.parse_query("What dividend will TCS pay in FY27?")

    assert result["intent"] == "forecast_single"
    assert result["ticker"] == "TCS"
    assert result["horizon"] == "FY27"


# ── out_of_scope: screener / multi-company ───────────────────────────
def test_screener_query_is_out_of_scope(monkeypatch):
    _mock_llm(
        monkeypatch,
        {
            "intent": "out_of_scope",
            "company_mention": None,
            "question": "Top public sector dividend paying companies",
            "horizon": None,
        },
    )

    result = intent.parse_query("Top public sector dividend paying companies")

    assert result["intent"] == "out_of_scope"
    assert result["ticker"] is None
    assert "roadmap" in result["message"].lower() or "one" in result["message"].lower()


# ── clarify: nonsense ────────────────────────────────────────────────
def test_nonsense_query_is_clarify(monkeypatch):
    _mock_llm(
        monkeypatch,
        {"intent": "clarify", "company_mention": None, "question": None, "horizon": None},
    )

    result = intent.parse_query("asdf qwer zxcv")

    assert result["intent"] == "clarify"
    assert result["ticker"] is None
    assert result["message"]


# ── unresolvable company mention → clarify, name echoed back ──────────
def test_unresolvable_company_becomes_clarify_with_name_echoed(monkeypatch):
    _mock_llm(
        monkeypatch,
        {
            "intent": "forecast_single",
            "company_mention": "Totally Made Up Bank Ltd",
            "question": "Forecast its dividend",
            "horizon": None,
        },
    )

    result = intent.parse_query("Forecast Totally Made Up Bank Ltd's dividend")

    assert result["intent"] == "clarify"
    assert result["ticker"] is None
    assert "Totally Made Up Bank Ltd" in result["message"]


# ── graceful degradation ─────────────────────────────────────────────
def test_empty_query_is_clarify_without_llm(monkeypatch):
    calls = _mock_llm(monkeypatch, {})
    result = intent.parse_query("   ")
    assert result["intent"] == "clarify"
    assert calls == []


def test_llm_failure_degrades_to_clarify(monkeypatch):
    _mock_llm(monkeypatch, RuntimeError("LLM unavailable"))
    result = intent.parse_query("Will some company raise its dividend?")
    assert result["intent"] == "clarify"
    assert result["message"]


def test_unknown_intent_from_llm_degrades_to_clarify(monkeypatch):
    _mock_llm(
        monkeypatch,
        {"intent": "banana", "company_mention": "ITC", "question": None, "horizon": None},
    )
    result = intent.parse_query("something weird")
    assert result["intent"] == "clarify"


# ── result contract: every key always present ────────────────────────
def test_result_always_has_all_contract_keys(monkeypatch):
    _mock_llm(monkeypatch, {"intent": "clarify"})
    result = intent.parse_query("hmm")
    assert set(result.keys()) == {
        "intent",
        "ticker",
        "question",
        "horizon",
        "company_mention",
        "message",
        "llm_used",
    }


def test_llm_used_flag_tracks_actual_llm_usage(monkeypatch):
    # Bare ticker: no LLM spent.
    calls = _mock_llm(monkeypatch, {})
    assert intent.parse_query("ITC")["llm_used"] is False
    assert calls == []
    # LLM path: one call spent.
    _mock_llm(
        monkeypatch,
        {"intent": "dividend_qa", "company_mention": "Infosys", "question": "q", "horizon": None},
    )
    assert intent.parse_query("Will Infosys raise its dividend?")["llm_used"] is True
