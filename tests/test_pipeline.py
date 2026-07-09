"""End-to-end tests for the LangGraph pipeline (src/graph.py), LLM mocked.

Every LLM call (intent, forecast, critic) is routed through one scripted
fake keyed on distinctive prompt text; data_agent and rag are faked too,
so the whole graph — routing, the critic retry loop, and error handling —
runs deterministically and offline. ratio_engine and report run for real
against the fake data, so this also exercises their integration.
"""

from __future__ import annotations

import pytest

from src import graph, intent


def _fake_raw(ticker="ITC"):
    return {
        "ticker": ticker,
        "company_name": f"{ticker} Ltd",
        "current_price": 400.0,
        "dividends": [
            {"date": "2023-06-01", "amount": 10.0},
            {"date": "2024-06-01", "amount": 11.0},
            {"date": "2025-06-01", "amount": 12.0},
        ],
        "income_statement": [{"period": "2025-03-31", "data": {"Net Income": 1000.0}}],
        "cash_flow": [
            {"period": "2025-03-31", "data": {"Free Cash Flow": 800.0, "Cash Dividends Paid": -400.0}}
        ],
        "balance_sheet": [
            {"period": "2025-03-31", "data": {"Total Debt": 200.0, "Stockholders Equity": 2000.0}}
        ],
        "data_timestamp": "2026-07-09T12:00:00+05:30",
    }


_FORECAST = {
    "direct_answer": None,
    "likelihood": None,
    "amount_range_inr": {"low": 12.0, "high": 13.0},
    "expected_window": "Interim ~Feb, final ~Jun",
    "confidence": "medium",
    "reasoning": ["Last-FY dividend was 12.0"],
    "risks": ["Cyclical earnings"],
}


class _ScriptedLLM:
    """Fake invoke_json: dispatches on prompt text; critic answers scripted."""

    def __init__(self, intent_response=None, forecast=None, critic_sequence=None):
        self.calls = {"intent": 0, "forecast": 0, "critic": 0}
        self.intent_response = intent_response
        self.forecast = forecast or _FORECAST
        self.critic_sequence = list(critic_sequence or [{"approved": True, "issues": []}])

    def __call__(self, prompt, schema_hint, task_type="reasoning"):
        if "intent parser" in prompt:
            self.calls["intent"] += 1
            return dict(self.intent_response)
        # Check "Critic Agent" before "Forecast Agent": the critic prompt
        # references "the Forecast Agent's output", so it contains both.
        if "Critic Agent" in prompt:
            self.calls["critic"] += 1
            nxt = self.critic_sequence.pop(0) if self.critic_sequence else {"approved": True, "issues": []}
            return dict(nxt)
        if "Forecast Agent" in prompt:
            self.calls["forecast"] += 1
            return dict(self.forecast)
        raise AssertionError(f"unexpected LLM prompt: {prompt[:60]}")


@pytest.fixture
def wire(monkeypatch):
    """Return a helper that installs a scripted LLM + fakes and runs a query."""

    def _install(llm, *, fetch=None, rag_hits=None):
        monkeypatch.setattr(graph.llm_router, "invoke_json", llm)
        monkeypatch.setattr(intent, "_ticker_exists", lambda c: False)
        monkeypatch.setattr(
            graph.data_agent, "fetch_company_data", fetch or (lambda t: _fake_raw(t))
        )
        monkeypatch.setattr(graph.rag, "retrieve", lambda ticker, *a, **k: rag_hits or [])
        # fresh compiled graph per test to avoid cross-test state
        monkeypatch.setattr(graph, "_graph", None)

    return _install


# ── happy path: bare ticker (no intent LLM) ──────────────────────────
def test_bare_ticker_full_pipeline(wire):
    llm = _ScriptedLLM()
    wire(llm)

    state = graph.run_pipeline("ITC")

    assert state["intent"] == "forecast_single"
    assert state["ticker"] == "ITC"
    assert llm.calls == {"intent": 0, "forecast": 1, "critic": 1}  # bare ticker skips intent LLM
    assert state["retry_count"] == 0
    assert state["critique"]["approved"] is True
    report = state["final_report"]
    assert "DiviSense AI — ITC" in report
    assert "Dividend forecast" in report
    assert "not investment advice" in report
    assert "Data as of" in report


# ── dividend_qa with direct-answer banner ────────────────────────────
def test_dividend_qa_pipeline_with_banner(wire):
    llm = _ScriptedLLM(
        intent_response={
            "intent": "dividend_qa",
            "company_mention": "Infosys",
            "question": "Will Infosys increase its dividend next quarter?",
            "horizon": "next quarter",
        },
        forecast={**_FORECAST, "direct_answer": "Likely yes — modest increase.", "likelihood": "likely"},
    )
    wire(llm)

    state = graph.run_pipeline("Will Infosys increase its dividend next quarter?")

    assert state["intent"] == "dividend_qa"
    assert state["ticker"] == "INFY"  # resolved from mention by ticker_map
    assert state["horizon"] == "next quarter"
    assert llm.calls == {"intent": 1, "forecast": 1, "critic": 1}
    assert "💬 Likely yes" in state["final_report"]


# ── critic retry loop: reject once, then approve ─────────────────────
def test_critic_retry_loop_runs_forecast_twice(wire):
    llm = _ScriptedLLM(
        critic_sequence=[
            {"approved": False, "issues": ["amount range ignores CAGR"]},
            {"approved": True, "issues": []},
        ]
    )
    wire(llm)

    state = graph.run_pipeline("ITC")

    assert llm.calls == {"intent": 0, "forecast": 2, "critic": 2}  # looped back exactly once
    assert state["retry_count"] == 1
    assert state["critique"]["approved"] is True


def test_critic_rejects_twice_stops_after_one_retry(wire):
    llm = _ScriptedLLM(
        critic_sequence=[
            {"approved": False, "issues": ["issue A"]},
            {"approved": False, "issues": ["issue B"]},
        ]
    )
    wire(llm)

    state = graph.run_pipeline("ITC")

    # Only ONE retry: forecast twice, critic twice, then report despite rejection.
    assert llm.calls == {"intent": 0, "forecast": 2, "critic": 2}
    assert state["retry_count"] == 1
    assert state["critique"]["approved"] is False
    assert state["final_report"]  # still produces a report


# ── routing: out_of_scope and clarify skip data/forecast ─────────────
def test_out_of_scope_routes_straight_to_report(wire):
    fetch_calls = []

    def _fetch(t):
        fetch_calls.append(t)
        return _fake_raw(t)

    llm = _ScriptedLLM(
        intent_response={"intent": "out_of_scope", "company_mention": None, "question": None, "horizon": None}
    )
    wire(llm, fetch=_fetch)

    state = graph.run_pipeline("Top public sector dividend paying companies")

    assert state["intent"] == "out_of_scope"
    assert fetch_calls == []  # never touched data/ratio/forecast
    assert llm.calls == {"intent": 1, "forecast": 0, "critic": 0}
    assert "Out of scope" in state["final_report"]


def test_clarify_routes_straight_to_report(wire):
    llm = _ScriptedLLM(
        intent_response={"intent": "clarify", "company_mention": None, "question": None, "horizon": None}
    )
    wire(llm)

    state = graph.run_pipeline("asdf qwer zxcv")

    assert state["intent"] == "clarify"
    assert llm.calls == {"intent": 1, "forecast": 0, "critic": 0}
    assert "clarify" in state["final_report"].lower()


# ── error handling: data-node failure routes to report ───────────────
def test_data_fetch_error_routes_to_report(wire):
    def _boom(ticker):
        raise graph.data_agent.InvalidTickerError(f"no data for {ticker}")

    llm = _ScriptedLLM()
    wire(llm, fetch=_boom)

    state = graph.run_pipeline("ITC")  # bare ticker -> data_node -> error

    assert state["errors"]
    assert llm.calls == {"intent": 0, "forecast": 0, "critic": 0}  # never reached forecast
    report = state["final_report"]
    assert "Couldn't" in report
    assert "not investment advice" in report  # disclaimer still present


# ── rag context flows into the report trace ──────────────────────────
def test_rag_snippets_appear_in_trace(wire):
    llm = _ScriptedLLM()
    wire(
        llm,
        rag_hits=[{"text": "Dividend policy: 40-50% payout.", "source_file": "itc.pdf", "page": 12, "score": 0.8}],
    )

    state = graph.run_pipeline("ITC")

    assert len(state["rag_context"]) == 1
    assert "1 annual-report snippet" in state["final_report"]
    assert "itc.pdf p12" in state["final_report"]
