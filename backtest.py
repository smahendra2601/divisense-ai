"""Backtest harness: hide the latest fiscal year's dividends, forecast, compare.

For each ticker, this:

1. fetches real market data via ``data_agent`` (cached as usual),
2. withholds every dividend attributed to the most recent **complete**
   fiscal year (and anything later, e.g. a half-collected current FY),
3. runs the *unmodified* pipeline (``graph.run_pipeline``) with the fetch
   monkeypatched to serve the redacted data, and
4. compares the forecast's ``amount_range_inr`` against the withheld
   actual total.

Reuses ``graph.py`` wholesale — no pipeline logic is duplicated.

Honesty caveats, by design:
- The **complete** FY is withheld (via ratio_engine's provisional-FY
  split), because scoring against a partially-collected year is
  meaningless.
- **RAG and news are both disabled** for the run: ingested annual reports
  post-date the cutoff, and a live news search would likely surface the
  actual dividend that was declared — either would leak the withheld
  answer into the forecast prompt.
- ``current_price`` is today's, not the historical price at the cutoff —
  yield is slightly anachronistic, but the forecast anchors on dividend
  history, which is correctly truncated.

Usage:
    python backtest.py            # ITC, TCS, COALINDIA
    python backtest.py INFY SBIN  # any tickers
"""

from __future__ import annotations

import copy
import logging
import sys

from src import graph
from src.ratio_engine import _annual_dividends, _results_fy, _split_provisional

logger = logging.getLogger(__name__)

DEFAULT_TICKERS = ["ITC", "TCS", "COALINDIA"]


def withhold_latest_complete_fy(raw: dict) -> tuple[dict, int, float]:
    """Return (redacted_raw, withheld_fy, actual_total).

    Removes every dividend event attributed to the most recent complete
    fiscal year *or later* (later events are "the future" from the
    forecast's point of view and must not leak).
    """
    dividends = raw.get("dividends") or []
    annual_full = _annual_dividends(dividends)
    complete, _provisional = _split_provisional(annual_full)
    if not complete:
        raise ValueError("no complete dividend fiscal year to withhold")

    target_fy, actual_total = complete[-1]

    redacted = copy.deepcopy(raw)
    redacted["dividends"] = [
        d for d in dividends if (_results_fy(d.get("date", "")) or 10**9) < target_fy
    ]
    if not redacted["dividends"]:
        raise ValueError(f"withholding FY{target_fy} would leave no dividend history")
    return redacted, target_fy, round(actual_total, 2)


def backtest_ticker(ticker: str) -> dict:
    """Run one withheld-year forecast for ``ticker`` and score it."""
    original_fetch = graph.data_agent.fetch_company_data
    original_retrieve = graph.rag.retrieve
    original_news = graph.news.fetch_recent_news

    raw = original_fetch(ticker)
    redacted, withheld_fy, actual = withhold_latest_complete_fy(raw)

    graph.data_agent.fetch_company_data = lambda t, _r=redacted: _r
    graph.rag.retrieve = lambda *a, **k: []  # reports post-date the cutoff
    graph.news.fetch_recent_news = lambda *a, **k: []  # news would leak the withheld actual
    try:
        state = graph.run_pipeline(ticker)
    finally:
        graph.data_agent.fetch_company_data = original_fetch
        graph.rag.retrieve = original_retrieve
        graph.news.fetch_recent_news = original_news

    forecast = state.get("forecast") or {}
    rng = forecast.get("amount_range_inr") or {}
    low = rng.get("low") if isinstance(rng, dict) else None
    high = rng.get("high") if isinstance(rng, dict) else None

    hit = low is not None and high is not None and low <= actual <= high
    # How far outside the range the actual landed (0 for a hit).
    miss_pct = None
    if low is not None and high is not None and not hit:
        nearest = low if actual < low else high
        miss_pct = round(abs(actual - nearest) / actual * 100, 1)

    return {
        "ticker": ticker,
        "withheld_fy": withheld_fy,
        "predicted_low": low,
        "predicted_high": high,
        "actual": actual,
        "hit": hit,
        "miss_pct": miss_pct,
        "confidence": forecast.get("confidence"),
        "retries": state.get("retry_count", 0),
        "llm_calls": state.get("llm_calls", 0),
        "errors": state.get("errors") or [],
    }


def run_backtest(tickers: list[str]) -> list[dict]:
    results = []
    for ticker in tickers:
        print(f"… backtesting {ticker}", flush=True)
        try:
            results.append(backtest_ticker(ticker))
        except Exception as exc:  # noqa: BLE001 - one bad ticker shouldn't kill the run
            logger.error("backtest: %s failed (%s)", ticker, exc)
            results.append({"ticker": ticker, "error": str(exc)})
    return results


def print_results(results: list[dict]) -> None:
    header = (
        f"{'Ticker':<11}{'Withheld':>9}{'Predicted range':>22}"
        f"{'Actual':>10}{'Result':>9}{'Off by':>9}{'Conf':>8}"
    )
    print("\n=== Backtest: forecast vs withheld fiscal year ===\n")
    print(header)
    print("-" * len(header))

    hits = scored = 0
    for r in results:
        if "error" in r:
            print(f"{r['ticker']:<11}  ERROR: {r['error']}")
            continue
        scored += 1
        hits += r["hit"]
        rng = (
            f"₹{r['predicted_low']} – ₹{r['predicted_high']}"
            if r["predicted_low"] is not None
            else "—"
        )
        off = f"{r['miss_pct']}%" if r["miss_pct"] is not None else "—"
        print(
            f"{r['ticker']:<11}{'FY' + str(r['withheld_fy']):>9}{rng:>22}"
            f"{'₹' + str(r['actual']):>10}{'✅ HIT' if r['hit'] else '❌ MISS':>9}"
            f"{off:>9}{(r['confidence'] or '—'):>8}"
        )
        if r["errors"]:
            print(f"{'':<11}  ⚠️  {'; '.join(r['errors'])}")

    if scored:
        print(f"\nHit rate: {hits}/{scored}")
    print(
        "\nNote: RAG and news disabled (would leak the withheld answer); price is "
        "current, not historical. Not investment advice."
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

    tickers = [t.upper() for t in sys.argv[1:]] or DEFAULT_TICKERS
    print_results(run_backtest(tickers))
