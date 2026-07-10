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

    def _install(llm, *, fetch=None, rag_hits=None, news_hits=None):
        monkeypatch.setattr(graph.llm_router, "invoke_json", llm)
        monkeypatch.setattr(intent, "_ticker_exists", lambda c: False)
        monkeypatch.setattr(
            graph.data_agent, "fetch_company_data", fetch or (lambda t: _fake_raw(t))
        )
        monkeypatch.setattr(graph.rag, "retrieve", lambda ticker, *a, **k: rag_hits or [])
        monkeypatch.setattr(graph.news, "fetch_recent_news", lambda *a, **k: news_hits or [])
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


# ── critic retry loop under the ≤3-call budget (§6) ──────────────────
def test_bare_ticker_rejection_retries_within_budget(wire):
    """Bare ticker: forecast(1) + critic(2) reject → retry forecast(3).

    The retry lands exactly on the budget cap, so the second critic pass
    is skipped and the report downgrades confidence to low.
    """
    llm = _ScriptedLLM(
        critic_sequence=[{"approved": False, "issues": ["amount range ignores CAGR"]}]
    )
    wire(llm)

    state = graph.run_pipeline("ITC")

    assert llm.calls == {"intent": 0, "forecast": 2, "critic": 1}
    assert state["llm_calls"] == 3  # never exceeds the budget
    assert state["retry_count"] == 1
    # Unvalidated retry → rejection stands → confidence downgraded.
    assert state["critique"]["approved"] is False
    assert state["forecast"]["confidence"] == "low"
    assert any("validation" in r.lower() for r in state["forecast"]["risks"])
    assert state["final_report"]


def test_question_rejection_cannot_retry_and_flags_low_confidence(wire):
    """Question path: intent(1)+forecast(2)+critic(3) spends the whole budget.

    A rejection therefore does NOT loop back — the report flags low
    confidence instead, keeping the query at exactly 3 LLM calls.
    """
    llm = _ScriptedLLM(
        intent_response={
            "intent": "dividend_qa",
            "company_mention": "Infosys",
            "question": "Will Infosys increase its dividend?",
            "horizon": None,
        },
        critic_sequence=[{"approved": False, "issues": ["contradicts trajectory"]}],
    )
    wire(llm)

    state = graph.run_pipeline("Will Infosys increase its dividend?")

    assert llm.calls == {"intent": 1, "forecast": 1, "critic": 1}  # no retry
    assert state["llm_calls"] == 3
    assert state["retry_count"] == 0
    assert state["forecast"]["confidence"] == "low"
    assert state["final_report"]


@pytest.mark.parametrize(
    "query,intent_resp,critic_seq",
    [
        ("ITC", None, [{"approved": True, "issues": []}]),
        ("ITC", None, [{"approved": False, "issues": ["x"]}]),
        (
            "Will Infosys raise its dividend?",
            {"intent": "dividend_qa", "company_mention": "Infosys", "question": "q", "horizon": None},
            [{"approved": True, "issues": []}],
        ),
        (
            "Will Infosys raise its dividend?",
            {"intent": "dividend_qa", "company_mention": "Infosys", "question": "q", "horizon": None},
            [{"approved": False, "issues": ["x"]}, {"approved": False, "issues": ["y"]}],
        ),
    ],
)
def test_llm_budget_never_exceeded(wire, query, intent_resp, critic_seq):
    """§6 quota discipline: total LLM calls ≤ 3 in every scenario."""
    llm = _ScriptedLLM(intent_response=intent_resp, critic_sequence=critic_seq)
    wire(llm)

    state = graph.run_pipeline(query)

    total = sum(llm.calls.values())
    assert total <= 3, f"budget exceeded: {llm.calls}"
    assert state.get("llm_calls", 0) <= 3


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


# ── news context flows into the trace and the forecast prompt ────────
def test_news_snippets_appear_in_trace_and_prompt(wire):
    captured_prompts = []

    def llm(prompt, schema_hint, task_type="reasoning"):
        captured_prompts.append(prompt)
        if "Critic Agent" in prompt:
            return {"approved": True, "issues": []}
        if "Forecast Agent" in prompt:
            return dict(_FORECAST)
        raise AssertionError("unexpected prompt")

    wire(
        llm,
        news_hits=[{"title": "ITC declares interim dividend", "url": "https://x", "snippet": "...", "score": 0.9}],
    )

    state = graph.run_pipeline("ITC")

    assert len(state["news_context"]) == 1
    assert "1 article(s)" in state["final_report"]
    assert "ITC declares interim dividend" in state["final_report"]
    # and the forecast prompt itself carried the news + the numbers-only guard
    assert any("ITC declares interim dividend" in p for p in captured_prompts)
    assert any("NEVER use a rupee figure from a news snippet" in p for p in captured_prompts)


def test_no_news_shows_zero_articles_in_trace(wire):
    llm = _ScriptedLLM()
    wire(llm)  # news_hits defaults to None -> []

    state = graph.run_pipeline("ITC")

    assert state["news_context"] == []
    assert "0 article(s)" in state["final_report"]


# ── smoke tests: the five scenarios exercised live against real
# providers (see conversation record) — pinned here as fast, offline
# regressions. Numeric grounding itself can't be meaningfully re-tested
# with a scripted LLM (the mock controls the numbers by construction);
# that was verified separately with an automated grounding check against
# live Groq/Gemini output, which found zero fabricated numbers.
def test_smoke_bare_ticker_itc(wire):
    """'ITC' -> forecast_single, 2 LLM calls (intent skipped), no retry."""
    llm = _ScriptedLLM()
    wire(llm)

    state = graph.run_pipeline("ITC")

    assert state["intent"] == "forecast_single"
    assert state["ticker"] == "ITC"
    assert llm.calls == {"intent": 0, "forecast": 1, "critic": 1}
    assert state["retry_count"] == 0
    assert state["critique"]["approved"] is True
    assert "not investment advice" in state["final_report"]


def test_smoke_dividend_qa_infosys_next_quarter(wire):
    """Infosys next-quarter question -> dividend_qa/INFY, 3 LLM calls (no retry)."""
    llm = _ScriptedLLM(
        intent_response={
            "intent": "dividend_qa",
            "company_mention": "Infosys",
            "question": "Will Infosys increase its dividend next quarter?",
            "horizon": "next quarter",
        },
        forecast={**_FORECAST, "direct_answer": "Unclear — mixed signals.", "likelihood": "unclear"},
    )
    wire(llm)

    state = graph.run_pipeline("Will Infosys increase its dividend next quarter?")

    assert state["intent"] == "dividend_qa"
    assert state["ticker"] == "INFY"
    assert state["horizon"] == "next quarter"
    assert llm.calls == {"intent": 1, "forecast": 1, "critic": 1}
    assert state["retry_count"] == 0
    report = state["final_report"]
    assert "💬 Unclear" in report
    assert "not investment advice" in report


def test_smoke_forecast_coalindia_next_year(wire):
    """Coal India next-year forecast -> forecast_single/COALINDIA, 3 LLM calls."""
    llm = _ScriptedLLM(
        intent_response={
            "intent": "forecast_single",
            "company_mention": "Coal India",
            "question": "Forecast Coal India's dividend for next year",
            "horizon": "next year",
        }
    )
    wire(llm)

    state = graph.run_pipeline("Forecast Coal India's dividend for next year")

    assert state["intent"] == "forecast_single"
    assert state["ticker"] == "COALINDIA"
    assert state["horizon"] == "next year"
    assert llm.calls == {"intent": 1, "forecast": 1, "critic": 1}
    assert state["retry_count"] == 0
    assert "not investment advice" in state["final_report"]


def test_smoke_screener_query_is_out_of_scope(wire):
    """A multi-company/screener question -> polite out_of_scope, 1 LLM call."""
    fetch_calls = []
    llm = _ScriptedLLM(
        intent_response={"intent": "out_of_scope", "company_mention": None, "question": None, "horizon": None}
    )
    wire(llm, fetch=lambda t: fetch_calls.append(t) or _fake_raw(t))

    state = graph.run_pipeline("Top public sector dividend paying companies")

    assert state["intent"] == "out_of_scope"
    assert state["ticker"] is None
    assert llm.calls == {"intent": 1, "forecast": 0, "critic": 0}
    assert fetch_calls == []  # data/ratio/forecast never touched
    report = state["final_report"]
    assert "Out of scope" in report
    assert "not investment advice" in report


def test_smoke_nonsense_ticker_is_friendly_clarify(wire):
    """A ticker-shaped but unresolvable query ('ZZZZZ') -> friendly clarify."""
    llm = _ScriptedLLM(
        intent_response={"intent": "clarify", "company_mention": None, "question": None, "horizon": None}
    )
    wire(llm)

    state = graph.run_pipeline("ZZZZZ")

    assert state["intent"] == "clarify"
    assert state["ticker"] is None
    # Not a valid ticker, so it falls through to the LLM (one call), never data/forecast.
    assert llm.calls == {"intent": 1, "forecast": 0, "critic": 0}
    report = state["final_report"]
    assert "clarify" in report.lower()
    assert "ZZZZZ" in report
    assert "not investment advice" in report
