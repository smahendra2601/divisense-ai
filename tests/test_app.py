"""Smoke tests for app.py (Tier 4 — Streamlit UI) via streamlit AppTest.

These run the app headless in-process. Rendering branches are tested by
seeding ``st.session_state`` with a canned pipeline result (so ``execute``
is skipped and no LLM/network is touched). One test exercises the
streamed ``execute`` path with ``src.graph.get_graph`` mocked. The bar
value is that the whole UI renders every branch without raising.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
from streamlit.testing.v1 import AppTest

_METRICS = {
    "current_yield_pct": 5.14,
    "total_dividends_last_fy": 14.5,
    "last_fiscal_year": 2026,
    "payout_ratio_5yr": {"average_pct": 79.3},
    "dividend_cagr_5yr": 5.97,
    "fcf_dividend_coverage": 0.89,
    "consecutive_increase_streak": 2,
    "dividend_consistency_score": 94,
    "recent_trajectory": {"classification": "mixed"},
    "debt_to_equity_trend": {"direction": "rising"},
    "years_of_history": 31,
    "warnings": [],
}

_RAW = {
    "dividends": [
        {"date": "2023-06-01", "amount": 12.0},
        {"date": "2024-06-01", "amount": 13.0},
        {"date": "2025-06-01", "amount": 14.0},
    ]
}


def _forecast_state(**over):
    base = {
        "user_query": "ITC",
        "intent": "forecast_single",
        "ticker": "ITC",
        "metrics": _METRICS,
        "forecast": {
            "direct_answer": None,
            "likelihood": None,
            "amount_range_inr": {"low": 14.0, "high": 16.0},
            "expected_window": "Interim ~Feb, final ~Jun",
            "confidence": "medium",
            "reasoning": ["Last-FY dividend was 14.5"],
            "risks": ["Cigarette-tax changes"],
        },
        "critique": {"approved": True, "issues": []},
        "rag_context": [],
        "retry_count": 0,
        "raw_data": _RAW,
        "data_timestamp": "2026-07-09T12:00:00+05:30",
    }
    base.update(over)
    return base


def _seed(state):
    """Build an AppTest whose stored result is `state` for user_query."""
    at = AppTest.from_file("app.py", default_timeout=30)
    at.session_state["pending_query"] = state["user_query"]
    at.session_state["result_query"] = state["user_query"]  # == pending → execute() skipped
    at.session_state["result_state"] = state
    return at


# ── initial page (no query) ──────────────────────────────────────────
def test_initial_page_renders_without_query():
    at = AppTest.from_file("app.py", default_timeout=30).run()
    assert not at.exception
    assert any("DiviSense AI" in t.value for t in at.title)
    # disclaimer banner present
    assert any("not investment advice" in w.value for w in at.warning)
    # three example chips (plus the form's submit button)
    labels = [b.label for b in at.button]
    assert "COALINDIA" in labels
    assert "Forecast ITC's dividend for next year" in labels
    assert "Will Infosys increase its dividend next quarter?" in labels


# ── forecast_single rendering ────────────────────────────────────────
def test_forecast_result_renders():
    at = _seed(_forecast_state()).run()
    assert not at.exception
    # forecast card metric
    assert any(m.label == "Expected dividend (next FY)" for m in at.metric)
    assert any("₹14.0 – ₹16.0" in m.value for m in at.metric)
    # metrics table + reasoning present
    joined = " ".join(md.value for md in at.markdown)
    assert "Reasoning" in joined
    assert "confidence" in joined.lower()


# ── dividend_qa direct-answer banner ─────────────────────────────────
def test_dividend_qa_shows_direct_answer_banner():
    state = _forecast_state(
        intent="dividend_qa",
        ticker="INFY",
        horizon="next quarter",
        question="Will Infosys increase its dividend next quarter?",
    )
    state["forecast"]["direct_answer"] = "Likely yes — a modest increase."
    state["forecast"]["likelihood"] = "likely"
    at = _seed(state).run()
    assert not at.exception
    joined = " ".join(md.value for md in at.markdown)
    assert "Likely yes" in joined


# ── clarify / out_of_scope / error branches ──────────────────────────
def test_clarify_renders_friendly_message():
    at = _seed(_forecast_state(intent="clarify", forecast=None, user_query="ZZZZZ")).run()
    assert not at.exception
    assert any("couldn't identify" in w.value.lower() for w in at.warning)


def test_out_of_scope_renders_roadmap_message():
    state = _forecast_state(intent="out_of_scope", forecast=None, user_query="top dividend payers")
    at = _seed(state).run()
    assert not at.exception
    assert any("screener" in i.value.lower() for i in at.info)


def test_error_state_renders_error():
    state = _forecast_state(
        intent="forecast_single", forecast=None, errors=["Couldn't find market data for 'ZZ'"]
    )
    at = _seed(state).run()
    assert not at.exception
    assert any("Couldn't find market data" in e.value for e in at.error)


# ── streamed execute() path with a mocked graph ──────────────────────
class _FakeGraph:
    def stream(self, initial, stream_mode="updates"):
        yield {"intent_node": {"intent": "forecast_single", "ticker": "ITC", "question": None, "horizon": None}}
        yield {"data_node": {"raw_data": _RAW, "data_timestamp": "2026-07-09T12:00:00+05:30"}}
        yield {"ratio_node": {"metrics": _METRICS}}
        yield {"rag_node": {"rag_context": []}}
        yield {"forecast_node": {"forecast": _forecast_state()["forecast"]}}
        yield {"critic_node": {"critique": {"approved": True, "issues": []}}}
        yield {"report_node": {"final_report": "ok"}}


def test_execute_streams_and_stores_result():
    at = AppTest.from_file("app.py", default_timeout=30)
    at.session_state["pending_query"] = "ITC"  # no result_query → execute() runs
    with patch("src.graph.get_graph", return_value=_FakeGraph()):
        at.run()
    assert not at.exception
    stored = at.session_state["result_state"]
    assert stored["ticker"] == "ITC"
    assert stored["forecast"]["amount_range_inr"] == {"low": 14.0, "high": 16.0}
    assert stored["critique"]["approved"] is True
