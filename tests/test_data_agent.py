"""Tests for src/data_agent.py (Tier 1 — yfinance fetcher).

yfinance is only touched in the two @integration tests at the bottom
(live network, ITC). Everything else replaces yf.Ticker with a
lightweight fake so the extraction/normalization logic, defensive
NaN/None handling, and InvalidTickerError paths are exercised
deterministically and offline.
"""

from __future__ import annotations

import json
from datetime import datetime

import pandas as pd
import pytest

from src import cache as cache_module
from src import data_agent
from src.data_agent import (
    InvalidTickerError,
    _company_meta,
    _current_price,
    _dividends,
    _period_label,
    _statement,
    _to_num,
    fetch_company_data,
)


# ── pure helper unit tests ───────────────────────────────────────────
@pytest.mark.parametrize(
    "raw,expected",
    [
        (None, None),
        (float("nan"), None),
        (float("inf"), None),
        (float("-inf"), None),
        (10, 10.0),
        (10.5, 10.5),
        ("12.3", 12.3),
        ("not a number", None),
    ],
)
def test_to_num(raw, expected):
    assert _to_num(raw) == expected


def test_period_label_formats_timestamp():
    assert _period_label(pd.Timestamp("2025-03-31")) == "2025-03-31"


def test_period_label_passes_through_plain_string():
    assert _period_label("some-label") == "some-label"


def test_statement_normalizes_dataframe_and_respects_limit():
    df = pd.DataFrame(
        {
            pd.Timestamp("2026-03-31"): {"Net Income": 100.0, "Total Revenue": float("nan")},
            pd.Timestamp("2025-03-31"): {"Net Income": 90.0, "Total Revenue": 400.0},
            pd.Timestamp("2024-03-31"): {"Net Income": 80.0, "Total Revenue": 380.0},
        }
    )
    records = _statement(df, limit=2)
    assert len(records) == 2
    assert records[0]["period"] == "2026-03-31"
    assert records[0]["data"]["Net Income"] == 100.0
    assert records[0]["data"]["Total Revenue"] is None  # NaN -> None


def test_statement_handles_none_and_empty():
    assert _statement(None) == []
    assert _statement(pd.DataFrame()) == []


def test_dividends_extracts_and_skips_bad_values():
    series = pd.Series(
        [1.5, float("nan"), 2.0],
        index=pd.to_datetime(["2024-06-01", "2024-09-01", "2025-06-01"]),
    )

    class _Tk:
        dividends = series

    assert _dividends(_Tk()) == [
        {"date": "2024-06-01", "amount": 1.5},
        {"date": "2025-06-01", "amount": 2.0},
    ]


def test_dividends_empty_series_returns_empty_list():
    class _Tk:
        dividends = pd.Series(dtype=float)

    assert _dividends(_Tk()) == []


def test_dividends_swallows_exceptions():
    class _Tk:
        @property
        def dividends(self):
            raise RuntimeError("network down")

    assert _dividends(_Tk()) == []


def test_company_meta_reads_info_fields():
    class _Tk:
        info = {"longName": "Test Co Ltd", "sector": "Testing", "currency": "INR"}

    assert _company_meta(_Tk()) == {
        "company_name": "Test Co Ltd",
        "sector": "Testing",
        "currency": "INR",
    }


def test_company_meta_swallows_exceptions():
    class _Tk:
        @property
        def info(self):
            raise RuntimeError("boom")

    assert _company_meta(_Tk()) == {"company_name": None, "sector": None, "currency": None}


def test_current_price_uses_fast_info_first():
    class _Tk:
        fast_info = {"last_price": 123.45}

    assert _current_price(_Tk()) == 123.45


def test_current_price_falls_back_to_info():
    class _Tk:
        fast_info = {}
        info = {"currentPrice": 250.0}

    assert _current_price(_Tk()) == 250.0


def test_current_price_falls_back_to_history():
    class _Tk:
        fast_info = {}
        info = {}

        def history(self, period="5d"):
            return pd.DataFrame({"Close": [10.0, 11.0, 12.5]})

    assert _current_price(_Tk()) == 12.5


def test_current_price_returns_none_when_all_sources_fail():
    class _Tk:
        fast_info = {}
        info = {}

        def history(self, period="5d"):
            return pd.DataFrame()

    assert _current_price(_Tk()) is None


# ── fetch_company_data: fully faked yf.Ticker ────────────────────────
class _FakeTickerGood:
    """A well-behaved company: price, dividends, and all 3 statements."""

    instances = []

    def __init__(self, symbol):
        self.symbol = symbol
        _FakeTickerGood.instances.append(symbol)

    fast_info = {"last_price": 283.1}
    info = {"longName": "Fake Good Ltd", "sector": "FMCG", "currency": "INR"}
    dividends = pd.Series([6.5, 7.85], index=pd.to_datetime(["2025-02-12", "2025-05-28"]))

    @staticmethod
    def _df():
        return pd.DataFrame({pd.Timestamp("2025-03-31"): {"Net Income": 100.0}})

    @property
    def income_stmt(self):
        return self._df()

    @property
    def balance_sheet(self):
        return self._df()

    @property
    def cashflow(self):
        return self._df()

    def history(self, period="5d"):
        return pd.DataFrame({"Close": [283.1]})


class _FakeTickerAllEmpty:
    """Simulates a bad/delisted symbol: nothing comes back from yfinance."""

    def __init__(self, symbol):
        self.symbol = symbol

    fast_info = {}
    info = {}
    dividends = pd.Series(dtype=float)
    income_stmt = None
    balance_sheet = None
    cashflow = None

    def history(self, period="5d"):
        return pd.DataFrame()


def test_fetch_company_data_happy_path(monkeypatch):
    monkeypatch.setattr(data_agent.yf, "Ticker", _FakeTickerGood)
    cache_module.clear()

    data = fetch_company_data("FAKEGOOD1")

    assert data["ticker"] == "FAKEGOOD1"
    assert data["yf_symbol"] == "FAKEGOOD1.NS"
    assert data["company_name"] == "Fake Good Ltd"
    assert data["current_price"] == 283.1
    assert len(data["dividends"]) == 2
    assert data["dividends"][0] == {"date": "2025-02-12", "amount": 6.5}
    assert len(data["income_statement"]) == 1
    assert len(data["balance_sheet"]) == 1
    assert len(data["cash_flow"]) == 1
    assert data["data_timestamp"].endswith("+05:30")
    datetime.fromisoformat(data["data_timestamp"])  # must be valid ISO 8601


def test_fetch_company_data_normalizes_whitespace_and_case(monkeypatch):
    monkeypatch.setattr(data_agent.yf, "Ticker", _FakeTickerGood)
    cache_module.clear()

    data = fetch_company_data("  fakegood5  ")
    assert data["ticker"] == "FAKEGOOD5"
    assert data["yf_symbol"] == "FAKEGOOD5.NS"


def test_fetch_company_data_tolerates_pre_suffixed_ticker(monkeypatch):
    monkeypatch.setattr(data_agent.yf, "Ticker", _FakeTickerGood)
    cache_module.clear()
    _FakeTickerGood.instances.clear()

    fetch_company_data("FAKEGOOD2")
    fetch_company_data("FAKEGOOD2.NS")  # different cache key, same resolved symbol

    assert len(_FakeTickerGood.instances) == 2
    assert all(sym == "FAKEGOOD2.NS" for sym in _FakeTickerGood.instances)


def test_fetch_company_data_is_json_serializable(monkeypatch):
    monkeypatch.setattr(data_agent.yf, "Ticker", _FakeTickerGood)
    cache_module.clear()

    data = fetch_company_data("FAKEGOOD3")
    json.dumps(data)  # must not raise


def test_fetch_company_data_raises_on_blank_ticker():
    with pytest.raises(InvalidTickerError):
        fetch_company_data("")
    with pytest.raises(InvalidTickerError):
        fetch_company_data("   ")


def test_fetch_company_data_raises_when_everything_is_empty(monkeypatch):
    monkeypatch.setattr(data_agent.yf, "Ticker", _FakeTickerAllEmpty)
    cache_module.clear()

    with pytest.raises(InvalidTickerError, match="ZZZFAKE"):
        fetch_company_data("ZZZFAKE")


def test_fetch_company_data_caches_repeated_calls(monkeypatch):
    monkeypatch.setattr(data_agent.yf, "Ticker", _FakeTickerGood)
    cache_module.clear()
    _FakeTickerGood.instances.clear()

    fetch_company_data("FAKECACHE")
    fetch_company_data("FAKECACHE")
    fetch_company_data("FAKECACHE")

    assert _FakeTickerGood.instances.count("FAKECACHE.NS") == 1


# ── integration: live NSE data via real yfinance ─────────────────────
@pytest.mark.integration
def test_itc_live_fetch_shape():
    cache_module.clear()
    data = fetch_company_data("ITC")

    assert data["ticker"] == "ITC"
    assert data["current_price"] and data["current_price"] > 0
    assert len(data["dividends"]) > 0
    assert len(data["income_statement"]) > 0
    assert data["data_timestamp"].endswith("+05:30")


@pytest.mark.integration
def test_invalid_ticker_raises_on_live_lookup():
    with pytest.raises(InvalidTickerError):
        fetch_company_data("THISISNOTAREALNSETICKERXYZ")
