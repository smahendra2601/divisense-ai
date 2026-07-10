"""Tests for backtest.py (withholding logic + scoring), fully offline.

The pipeline itself is exercised elsewhere; here we pin the two things
backtest.py adds: correct redaction of the most recent complete fiscal
year (including dropping any later partial-year events), and hit/miss
scoring against the withheld actual.
"""

from __future__ import annotations

import pytest

import backtest


def _raw(dividends):
    return {"ticker": "TEST", "dividends": dividends, "current_price": 100.0}


# Four complete FYs (one June payout each) + a partial current FY. Interims
# for FY2023-26 land in Feb of the same calendar year (results_fy = year).
_COMPLETE_HISTORY = [
    {"date": "2022-02-01", "amount": 4.0},
    {"date": "2022-06-01", "amount": 6.0},   # FY2022 total 10
    {"date": "2023-02-01", "amount": 5.0},
    {"date": "2023-06-01", "amount": 6.0},   # FY2023 total 11
    {"date": "2024-02-01", "amount": 5.0},
    {"date": "2024-06-01", "amount": 7.0},   # FY2024 total 12
    {"date": "2025-02-01", "amount": 6.0},
    {"date": "2025-06-01", "amount": 7.0},   # FY2025 total 13
    {"date": "2026-02-01", "amount": 6.5},   # FY2026: interim only -> partial
]


def test_withholds_last_complete_fy_and_drops_partial_tail():
    redacted, fy, actual = backtest.withhold_latest_complete_fy(_raw(_COMPLETE_HISTORY))

    assert fy == 2025
    assert actual == 13.0
    dates = [d["date"] for d in redacted["dividends"]]
    # FY2025 events AND the partial FY2026 interim are gone.
    assert "2025-02-01" not in dates and "2025-06-01" not in dates
    assert "2026-02-01" not in dates
    # History up to FY2024 is intact.
    assert "2024-06-01" in dates and "2022-02-01" in dates


def test_withhold_raises_when_no_history():
    with pytest.raises(ValueError):
        backtest.withhold_latest_complete_fy(_raw([]))


def test_withhold_raises_when_nothing_would_remain():
    only_one_fy = [{"date": "2025-06-01", "amount": 7.0}]
    with pytest.raises(ValueError):
        backtest.withhold_latest_complete_fy(_raw(only_one_fy))


def _run_scored(monkeypatch, forecast_range, rag_calls=None):
    monkeypatch.setattr(
        backtest.graph.data_agent, "fetch_company_data", lambda t: _raw(_COMPLETE_HISTORY)
    )

    def fake_pipeline(query):
        # The redacted fetch and disabled RAG must be active *during* the run.
        served = backtest.graph.data_agent.fetch_company_data("TEST")
        assert all(d["date"] < "2025" for d in served["dividends"])
        if rag_calls is not None:
            rag_calls.append(backtest.graph.rag.retrieve("TEST"))
        return {
            "forecast": {"amount_range_inr": forecast_range, "confidence": "medium"},
            "retry_count": 0,
            "llm_calls": 2,
            "errors": [],
        }

    monkeypatch.setattr(backtest.graph, "run_pipeline", fake_pipeline)
    return backtest.backtest_ticker("TEST")


def test_backtest_ticker_scores_hit(monkeypatch):
    result = _run_scored(monkeypatch, {"low": 12.0, "high": 14.0})
    assert result["withheld_fy"] == 2025
    assert result["actual"] == 13.0
    assert result["hit"] is True
    assert result["miss_pct"] is None


def test_backtest_ticker_scores_miss_with_distance(monkeypatch):
    result = _run_scored(monkeypatch, {"low": 8.0, "high": 10.0})
    assert result["hit"] is False
    # actual 13 vs nearest bound 10 -> 3/13 = 23.1%
    assert result["miss_pct"] == pytest.approx(23.1)


def test_rag_disabled_during_run_and_restored_after(monkeypatch):
    rag_calls: list = []
    _run_scored(monkeypatch, {"low": 12.0, "high": 14.0}, rag_calls=rag_calls)
    assert rag_calls == [[]]  # retrieve() returned [] mid-run
    # and the original retrieve is restored afterwards
    from src import rag as rag_module

    assert backtest.graph.rag.retrieve is rag_module.retrieve


def test_fetch_restored_even_if_pipeline_raises(monkeypatch):
    sentinel = lambda t: _raw(_COMPLETE_HISTORY)  # noqa: E731
    monkeypatch.setattr(backtest.graph.data_agent, "fetch_company_data", sentinel)
    monkeypatch.setattr(
        backtest.graph, "run_pipeline",
        lambda q: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    with pytest.raises(RuntimeError):
        backtest.backtest_ticker("TEST")
    assert backtest.graph.data_agent.fetch_company_data is sentinel  # restored