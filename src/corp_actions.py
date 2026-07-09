"""Tier 1 — Data Acquisition: corporate-actions source interface + CSV impl.

Defines the abstract ``CorporateActionsSource`` (``get_actions(ticker)``)
and a ``CSVSource`` MVP implementation that reads
``data/corporate_actions.csv``. The interface exists so a future
``NSEScraperSource`` can slot in without touching any other module
(ARCHITECTURE.md §7). Use ``get_default_source()`` everywhere rather than
constructing a concrete source directly — that keeps the swap to a live
scraper a one-line change here.

CSV columns:
    ticker, action_type, amount, announcement_date, ex_date,
    record_date, source_note
"""

from __future__ import annotations

import csv
from abc import ABC, abstractmethod

from . import config

# CSV columns, in order — the contract every source's dicts follow.
FIELDS = [
    "ticker",
    "action_type",
    "amount",
    "announcement_date",
    "ex_date",
    "record_date",
    "source_note",
]


class CorporateActionsSource(ABC):
    """Interface for a provider of a company's corporate actions.

    Implementations return a list of action dicts (see ``FIELDS``) for a
    given NSE ticker. A source with no data for a ticker returns an empty
    list — never ``None`` and never an exception — so the pipeline can
    render "no corporate actions on file" gracefully.
    """

    @abstractmethod
    def get_actions(self, ticker: str) -> list[dict]:
        """Return corporate actions for ``ticker`` (may be an empty list)."""
        raise NotImplementedError


class CSVSource(CorporateActionsSource):
    """MVP source backed by a local CSV (``data/corporate_actions.csv``).

    Matching is case-insensitive on the ticker. A missing or empty file is
    treated as "no data" rather than an error. Results are sorted with the
    most recent ex-date first.
    """

    def __init__(self, csv_path=None):
        self.csv_path = csv_path or config.CORPORATE_ACTIONS_CSV

    def get_actions(self, ticker: str) -> list[dict]:
        if not ticker or not ticker.strip():
            return []
        wanted = ticker.strip().upper()

        try:
            with open(self.csv_path, newline="", encoding="utf-8") as fh:
                rows = list(csv.DictReader(fh))
        except FileNotFoundError:
            return []

        actions: list[dict] = []
        for row in rows:
            if (row.get("ticker") or "").strip().upper() != wanted:
                continue
            actions.append(
                {
                    "ticker": (row.get("ticker") or "").strip().upper(),
                    "action_type": (row.get("action_type") or "").strip(),
                    "amount": _to_float(row.get("amount")),
                    "announcement_date": (row.get("announcement_date") or "").strip() or None,
                    "ex_date": (row.get("ex_date") or "").strip() or None,
                    "record_date": (row.get("record_date") or "").strip() or None,
                    "source_note": (row.get("source_note") or "").strip() or None,
                }
            )

        # Most recent ex-date first; blanks sort last.
        actions.sort(key=lambda a: a["ex_date"] or "", reverse=True)
        return actions


class NSEScraperSource(CorporateActionsSource):
    """Placeholder for a live NSE/BSE corporate-actions scraper.

    Deliberately unimplemented for the MVP. Because it satisfies the same
    ``CorporateActionsSource`` interface, wiring it in later is a one-line
    change in ``get_default_source()`` with zero changes elsewhere — see
    ARCHITECTURE.md §7 (v1.1 enhancement roadmap).
    """

    def get_actions(self, ticker: str) -> list[dict]:
        raise NotImplementedError(
            "NSEScraperSource is a future enhancement (ARCHITECTURE.md §7): a live "
            "NSE/BSE corporate-actions scraper implementing CorporateActionsSource. "
            "Use get_default_source() / CSVSource for the MVP."
        )


def _to_float(value) -> float | None:
    """Parse a CSV amount cell to float, or None if blank/unparseable."""
    if value is None:
        return None
    text = str(value).strip().replace(",", "")
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def get_default_source() -> CorporateActionsSource:
    """Return the corporate-actions source the pipeline should use.

    The single seam where the MVP's ``CSVSource`` is swapped for a live
    ``NSEScraperSource`` in future.
    """
    return CSVSource()


if __name__ == "__main__":
    import json
    import sys

    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

    query = sys.argv[1] if len(sys.argv) > 1 else "ITC"
    source = get_default_source()
    result = source.get_actions(query)
    print(f"Corporate actions for {query.upper()} ({len(result)} found):\n")
    print(json.dumps(result, indent=2, ensure_ascii=False))
