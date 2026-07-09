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
