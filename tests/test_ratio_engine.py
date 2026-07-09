"""Tests for src/ratio_engine.py (Tier 2 — Intelligence).

Two layers:

* **Unit tests** — deterministic, offline. They feed synthetic
  ``raw_data`` into ``compute_metrics`` to pin the pure logic (fiscal-year
  aggregation, trajectory, streak, CAGR, payout, coverage, yield, and the
  missing-input contract).
* **Integration tests** (``@pytest.mark.integration``) — hit live
  yfinance for ITC and COALINDIA and assert the results are structurally
  complete and land in plausible ranges. Deselect with
  ``pytest -m "not integration"``; they skip cleanly if the network or
  Yahoo data is unavailable.
"""

from __future__ import annotations

import pytest

from src.ratio_engine import compute_metrics

# ── fixtures ─────────────────────────────────────────────────────────
# A clean, strictly-rising annual dividend history (one payout per FY,
# ex-dates in June → each maps to that calendar year's results FY).
_RISING = {
    "ticker": "TESTCO",
    "current_price": 180.0,
    "dividends": [
        {"date": "2021-06-01", "amount": 5.0},
        {"date": "2022-06-01", "amount": 6.0},
        {"date": "2023-06-01", "amount": 7.0},
        {"date": "2024-06-01", "amount": 8.0},
        {"date": "2025-06-01", "amount": 9.0},
    ],
    "income_statement": [{"period": "2025-03-31", "data": {"Net Income": 1000.0}}],
    "cash_flow": [
        {
            "period": "2025-03-31",
            "data": {"Free Cash Flow": 800.0, "Cash Dividends Paid": -400.0},
        }
    ],
    "balance_sheet": [
        {"period": "2025-03-31", "data": {"Total Debt": 200.0, "Stockholders Equity": 2000.0}}
    ],
}


# ── unit tests (offline, deterministic) ──────────────────────────────
def test_rising_series_core_metrics():
    m = compute_metrics(_RISING)

    assert m["years_of_history"] == 5
    assert m["last_fiscal_year"] == 2025
    assert m["total_dividends_last_fy"] == pytest.approx(9.0)

    # Strictly rising annual totals.
    assert m["recent_trajectory"]["classification"] == "rising"
    assert m["recent_trajectory"]["amounts"] == [6.0, 7.0, 8.0, 9.0]

    # Increases every year → streak of 4 transitions.
    assert m["consecutive_increase_streak"] == 4

    # CAGR 5 → 9 over 4 years ≈ 15.8%.
    assert m["dividend_cagr_5yr"] == pytest.approx(15.83, abs=0.3)

    # yield = 9 / 180 = 5.0%.
    assert m["current_yield_pct"] == pytest.approx(5.0, abs=0.01)


def test_rising_series_statement_metrics():
    m = compute_metrics(_RISING)

    # Payout = |−400| / 1000 = 40%.
    assert m["payout_ratio_5yr"]["average_pct"] == pytest.approx(40.0)
    # FCF coverage = 800 / 400 = 2.0x.
    assert m["fcf_dividend_coverage"] == pytest.approx(2.0)
    # D/E = 200 / 2000 = 0.1.
    de = m["debt_to_equity_trend"]["per_year"][-1]["debt_to_equity"]
    assert de == pytest.approx(0.1)
    # Perfect history: paid every year, never cut → 100.
    assert m["dividend_consistency_score"] == 100


def test_interim_final_attribution_matches_screener_style():
    # ITC-style: interim (Feb) + final (May/Jun) collapse into one FY.
    raw = {
        "current_price": 400.0,
        "dividends": [
            {"date": "2025-02-12", "amount": 6.5},   # interim FY2025
            {"date": "2025-05-28", "amount": 7.85},  # final   FY2025
            {"date": "2026-02-04", "amount": 6.5},   # interim FY2026
            {"date": "2026-05-27", "amount": 8.0},   # final   FY2026
        ],
    }
    m = compute_metrics(raw)
    traj = m["recent_trajectory"]
    assert traj["fiscal_years"] == [2025, 2026]
    assert traj["amounts"] == [pytest.approx(14.35), pytest.approx(14.5)]
    assert m["total_dividends_last_fy"] == pytest.approx(14.5)


def test_falling_and_flat_classification():
    falling = {
        "dividends": [
            {"date": "2022-06-01", "amount": 10.0},
            {"date": "2023-06-01", "amount": 8.0},
            {"date": "2024-06-01", "amount": 6.0},
        ]
    }
    assert compute_metrics(falling)["recent_trajectory"]["classification"] == "falling"
    assert compute_metrics(falling)["consecutive_increase_streak"] == 0

    flat = {
        "dividends": [
            {"date": "2022-06-01", "amount": 5.0},
            {"date": "2023-06-01", "amount": 5.0},
            {"date": "2024-06-01", "amount": 5.0},
        ]
    }
    assert compute_metrics(flat)["recent_trajectory"]["classification"] == "flat"


def test_missing_inputs_return_none_with_warnings():
    m = compute_metrics({})
    # Every headline metric degrades to None, nothing raises.
    for key in (
        "payout_ratio_5yr",
        "dividend_cagr_5yr",
        "fcf_dividend_coverage",
        "consecutive_increase_streak",
        "current_yield_pct",
        "debt_to_equity_trend",
        "dividend_consistency_score",
        "total_dividends_last_fy",
        "years_of_history",
        "recent_trajectory",
    ):
        assert m[key] is None
    assert isinstance(m["warnings"], list) and m["warnings"]


def test_missing_price_only_disables_yield():
    raw = dict(_RISING)
    raw = {**_RISING, "current_price": None}
    m = compute_metrics(raw)
    assert m["current_yield_pct"] is None
    assert any("price" in w.lower() for w in m["warnings"])
    # Other metrics still compute.
    assert m["total_dividends_last_fy"] == pytest.approx(9.0)


# ── integration tests (live yfinance) ────────────────────────────────
def _fetch_or_skip(ticker: str) -> dict:
    try:
        from src.data_agent import InvalidTickerError, fetch_company_data

        return fetch_company_data(ticker)
    except InvalidTickerError as exc:  # pragma: no cover - env dependent
        pytest.skip(f"{ticker}: {exc}")
    except Exception as exc:  # pragma: no cover - network flakiness
        pytest.skip(f"{ticker}: live fetch failed ({exc})")


@pytest.mark.integration
def test_itc_live_metrics_are_plausible():
    m = compute_metrics(_fetch_or_skip("ITC"))

    assert m["years_of_history"] and m["years_of_history"] >= 10
    assert m["total_dividends_last_fy"] and m["total_dividends_last_fy"] > 0
    # ITC is a high-yield large cap.
    assert 1.0 < m["current_yield_pct"] < 12.0
    # ITC pays out most of its earnings.
    assert 40.0 < m["payout_ratio_5yr"]["average_pct"] < 130.0
    assert m["recent_trajectory"]["classification"] in {
        "rising",
        "flat",
        "falling",
        "mixed",
    }
    assert 0 <= m["dividend_consistency_score"] <= 100
    assert isinstance(m["warnings"], list)


@pytest.mark.integration
def test_coalindia_live_metrics_are_plausible():
    m = compute_metrics(_fetch_or_skip("COALINDIA"))

    assert m["years_of_history"] and m["years_of_history"] >= 5
    assert m["total_dividends_last_fy"] and m["total_dividends_last_fy"] > 0
    # COALINDIA is a very high-yield PSU.
    assert m["current_yield_pct"] and m["current_yield_pct"] > 3.0
    assert 0 <= m["dividend_consistency_score"] <= 100
    assert m["fcf_dividend_coverage"] is None or m["fcf_dividend_coverage"] > 0
