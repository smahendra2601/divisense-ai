"""Shared test fixtures.

Every test runs against a small, fixed NSE symbol master so ticker
resolution is deterministic and never touches the network (the real
``_symbol_master`` would try a live NSE fetch, then the on-disk
snapshot). Tests that exercise the fetch/fallback chain itself reach the
real implementation via ``ticker_map._symbol_master.__wrapped__``.
"""

from __future__ import annotations

import pytest

from src import ticker_map

FIXTURE_MASTER = [
    ("CANBK", "Canara Bank"),
    ("CANHLIFE", "Canara HSBC Life Insurance Company Limited"),
    ("CRAMC", "Canara Robeco Asset Management Company Limited"),
    ("CANFINHOME", "Can Fin Homes Limited"),
    ("HDFCBANK", "HDFC Bank Limited"),
    ("ITC", "ITC Limited"),
]


@pytest.fixture(autouse=True)
def _offline_symbol_master(monkeypatch):
    monkeypatch.setattr(ticker_map, "_symbol_master", lambda: list(FIXTURE_MASTER))
    ticker_map._master_symbols_set.cache_clear()
    yield
    ticker_map._master_symbols_set.cache_clear()
