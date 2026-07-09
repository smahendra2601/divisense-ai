"""Tests for src/corp_actions.py (Tier 1 — corporate actions).

Covers the CorporateActionsSource interface contract, the CSVSource MVP
implementation (against both an isolated temp CSV and the real shipped
data/corporate_actions.csv), the NSEScraperSource stub, and the
get_default_source() factory.
"""

from __future__ import annotations

import pytest

from src.corp_actions import (
    FIELDS,
    CorporateActionsSource,
    CSVSource,
    NSEScraperSource,
    get_default_source,
)

_SAMPLE_CSV = """ticker,action_type,amount,announcement_date,ex_date,record_date,source_note
ITC,Dividend (Final),7.85,2025-05-22,2025-05-28,2025-05-29,"Note with, an embedded comma"
ITC,Dividend (Interim),6.50,2025-02-05,2025-02-12,2025-02-13,
COALINDIA,Dividend (Interim),10.25,2025-10-28,2025-11-04,2025-11-05,SAMPLE
"""


@pytest.fixture
def sample_source(tmp_path):
    csv_path = tmp_path / "corporate_actions.csv"
    csv_path.write_text(_SAMPLE_CSV, encoding="utf-8")
    return CSVSource(csv_path=csv_path)


def test_source_is_abstract():
    with pytest.raises(TypeError):
        CorporateActionsSource()  # type: ignore[abstract]


def test_get_actions_returns_matching_rows(sample_source):
    actions = sample_source.get_actions("ITC")
    assert len(actions) == 2
    assert all(set(a.keys()) == set(FIELDS) for a in actions)


def test_case_insensitive_ticker_match(sample_source):
    assert len(sample_source.get_actions("itc")) == 2
    assert len(sample_source.get_actions("Itc")) == 2


def test_unknown_ticker_returns_empty_list(sample_source):
    assert sample_source.get_actions("NOPE") == []


def test_empty_string_returns_empty_list(sample_source):
    assert sample_source.get_actions("") == []


def test_missing_file_returns_empty_list_not_error(tmp_path):
    source = CSVSource(csv_path=tmp_path / "does_not_exist.csv")
    assert source.get_actions("ITC") == []


def test_amount_parsed_as_float(sample_source):
    action = sample_source.get_actions("COALINDIA")[0]
    assert action["amount"] == 10.25
    assert isinstance(action["amount"], float)


def test_blank_source_note_becomes_none(sample_source):
    interim = next(
        a for a in sample_source.get_actions("ITC") if a["action_type"] == "Dividend (Interim)"
    )
    assert interim["source_note"] is None


def test_quoted_comma_in_source_note_parses_intact(sample_source):
    final = next(
        a for a in sample_source.get_actions("ITC") if a["action_type"] == "Dividend (Final)"
    )
    assert final["source_note"] == "Note with, an embedded comma"


def test_sorted_most_recent_ex_date_first(sample_source):
    actions = sample_source.get_actions("ITC")
    assert actions[0]["ex_date"] == "2025-05-28"
    assert actions[1]["ex_date"] == "2025-02-12"


def test_nse_scraper_source_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="ARCHITECTURE.md"):
        NSEScraperSource().get_actions("ITC")


def test_default_source_is_csv_source_instance():
    source = get_default_source()
    assert isinstance(source, CSVSource)
    assert isinstance(source, CorporateActionsSource)


# ── against the real shipped CSV ─────────────────────────────────────
@pytest.mark.parametrize("ticker", ["ITC", "COALINDIA", "TCS", "INFY", "HCLTECH"])
def test_real_sample_csv_has_a_row_per_seeded_ticker(ticker):
    actions = get_default_source().get_actions(ticker)
    assert len(actions) >= 1
    assert actions[0]["ticker"] == ticker
    assert "SAMPLE DATA" in (actions[0]["source_note"] or "")
