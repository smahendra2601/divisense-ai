"""Tests for src/ticker_map.py (Tier 1 — name/ticker resolution).

resolve() is pure and deterministic: case-insensitive, whitespace-
tolerant, exact-ticker-first matching over data/ticker_aliases.csv.
These run against the real alias CSV shipped in the repo, so a handful
of assertions double as a content sanity check for that file.
"""

from __future__ import annotations

import pytest

from src.ticker_map import resolve


@pytest.mark.parametrize(
    "query,expected",
    [
        ("Infosys", "INFY"),
        ("Infy", "INFY"),
        ("INFY", "INFY"),
        ("infosys", "INFY"),
        ("Coal India", "COALINDIA"),
        ("coal india", "COALINDIA"),
        ("SBI", "SBIN"),
        ("State Bank", "SBIN"),
        ("State Bank of India", "SBIN"),
        ("TCS", "TCS"),
        ("Tata Consultancy", "TCS"),
        ("Tata Consultancy Services", "TCS"),
        ("ITC", "ITC"),
        ("itc", "ITC"),
        ("HCL", "HCLTECH"),
        ("HCL Tech", "HCLTECH"),
        ("HCL Technologies", "HCLTECH"),
        ("ONGC", "ONGC"),
        ("L&T", "LT"),
        ("Larsen", "LT"),
        ("Larsen & Toubro", "LT"),
    ],
)
def test_resolves_known_aliases(query, expected):
    assert resolve(query) == expected


def test_case_insensitive():
    assert resolve("infosys") == resolve("INFOSYS") == resolve("InFoSyS") == "INFY"


def test_whitespace_tolerant():
    assert resolve("  State   Bank  ") == "SBIN"
    assert resolve("\tCoal India\n") == "COALINDIA"


def test_exact_ticker_match_wins_first():
    assert resolve("ITC") == "ITC"
    # A ticker that is nobody's alias should still resolve via the
    # ticker column itself.
    assert resolve("HDFCBANK") == "HDFCBANK"


def test_unknown_company_returns_none():
    assert resolve("Definitely Not A Real Company Ltd") is None


def test_empty_and_whitespace_only_return_none():
    assert resolve("") is None
    assert resolve("   ") is None


def test_none_input_handled_without_raising():
    assert resolve(None) is None  # type: ignore[arg-type]


# ── NSE symbol master: stage-2 exact symbol + fuzzy search ───────────
# The master is pinned to a small fixture in tests/conftest.py.
from src import config, ticker_map  # noqa: E402
from src.ticker_map import (  # noqa: E402
    _normalize_company,
    _parse_master_csv,
    resolve_with_suggestions,
    search,
)

# Captured at module-import time (before any fixture monkeypatches the
# module attribute) so fallback-chain tests can call the real function.
_REAL_SYMBOL_MASTER = ticker_map._symbol_master.__wrapped__


def test_resolve_exact_symbol_from_master_not_in_aliases():
    # CANBK is in the NSE master fixture but NOT in ticker_aliases.csv.
    assert resolve("CANBK") == "CANBK"
    assert resolve("canbk") == "CANBK"


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("Canara Bank", "CANARA BANK"),
        ("CANARA BANK LIMITED", "CANARA BANK"),
        ("HDFC Bank Ltd.", "HDFC BANK"),
        ("Larsen & Toubro Limited", "LARSEN & TOUBRO"),
        ("Canara Robeco Asset Management Company Limited", "CANARA ROBECO ASSET MANAGEMENT"),
    ],
)
def test_normalize_company(raw, expected):
    assert _normalize_company(raw) == expected


def test_search_exact_name_scores_one():
    hits = search("Canara Bank")
    assert hits[0]["ticker"] == "CANBK"
    assert hits[0]["score"] == 1.0


def test_search_partial_name_surfaces_all_group_companies():
    hits = search("canara")
    top3 = {h["ticker"] for h in hits[:3]}
    assert top3 == {"CANBK", "CANHLIFE", "CRAMC"}
    assert all(h["score"] >= config.TICKER_SUGGEST_MIN_SCORE for h in hits)


def test_search_gibberish_returns_empty():
    assert search("qqqxyzzy") == []
    assert search("") == []


def test_resolve_with_suggestions_auto_accepts_exact_name():
    ticker, suggestions = resolve_with_suggestions("Canara Bank")
    assert ticker == "CANBK"
    assert suggestions == []


def test_resolve_with_suggestions_ambiguous_returns_candidates():
    ticker, suggestions = resolve_with_suggestions("Canara")
    assert ticker is None
    assert {s["ticker"] for s in suggestions[:3]} == {"CANBK", "CANHLIFE", "CRAMC"}


def test_resolve_with_suggestions_nothing_matches():
    assert resolve_with_suggestions("qqqxyzzy") == (None, [])


# ── master CSV parsing + fallback chain (real implementation) ────────
_SAMPLE_MASTER_CSV = """SYMBOL,NAME OF COMPANY, SERIES, DATE OF LISTING
CANBK,Canara Bank,EQ,23-DEC-2002
FOO,Foo Industries Limited,EQ,01-JAN-2020
,Missing Symbol Co,EQ,01-JAN-2020
"""


def test_parse_master_csv_handles_padded_headers_and_blank_rows():
    rows = _parse_master_csv(_SAMPLE_MASTER_CSV)
    assert rows == [("CANBK", "Canara Bank"), ("FOO", "Foo Industries Limited")]


def test_parse_master_csv_wrong_headers_returns_empty():
    assert _parse_master_csv("a,b\n1,2\n") == []


def test_symbol_master_falls_back_to_snapshot(monkeypatch, tmp_path):
    real_symbol_master = _REAL_SYMBOL_MASTER

    def _boom():
        raise RuntimeError("network down")

    monkeypatch.setattr(ticker_map, "_fetch_symbol_master_live", _boom)
    snapshot = tmp_path / "nse_symbols.csv"
    snapshot.write_text(_SAMPLE_MASTER_CSV, encoding="utf-8")
    monkeypatch.setattr(config, "NSE_SYMBOLS_CSV", snapshot)

    rows = real_symbol_master()
    assert ("CANBK", "Canara Bank") in rows


def test_symbol_master_degrades_to_empty_without_snapshot(monkeypatch, tmp_path):
    real_symbol_master = _REAL_SYMBOL_MASTER

    monkeypatch.setattr(
        ticker_map, "_fetch_symbol_master_live",
        lambda: (_ for _ in ()).throw(RuntimeError("network down")),
    )
    monkeypatch.setattr(config, "NSE_SYMBOLS_CSV", tmp_path / "missing.csv")

    assert real_symbol_master() == []
