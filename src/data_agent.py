"""Tier 1 — Data Acquisition: live yfinance fetcher.

Given an NSE ticker (e.g. ``ITC``), appends ``.NS`` and fetches dividend
history, current price, annual income statement, balance sheet, and
cash flow via yfinance. Returns a normalized, JSON-serializable dict
including a ``data_timestamp`` (ISO 8601, IST). Raises a clear
``InvalidTickerError`` for bad tickers. Fetch-on-demand — freshness by
design.

yfinance is best-effort: any given field may be absent or ``NaN``. Every
extractor here degrades to ``None``/empty rather than raising, so a
partial-but-useful payload is always returned for a real ticker.
"""

from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeout
from datetime import datetime, timedelta, timezone

import yfinance as yf

from . import config
from .cache import disk_cached

# India Standard Time — every timestamp we stamp is IST so the UI's
# "data as of <time>" reads naturally for an Indian-market tool.
IST = timezone(timedelta(hours=5, minutes=30))

# How many most-recent annual periods to keep from each statement.
_ANNUAL_PERIODS = 4


class InvalidTickerError(ValueError):
    """Raised when a symbol resolves to no usable market data on NSE."""


class DataFetchTimeoutError(RuntimeError):
    """Raised when Yahoo Finance doesn't answer within the configured window."""


def _to_num(value) -> float | None:
    """Coerce a pandas/numpy cell to a plain float, or ``None`` if unusable."""
    if value is None:
        return None
    try:
        num = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(num) or math.isinf(num):
        return None
    return num


def _period_label(period) -> str:
    """Render a statement column label (usually a Timestamp) as YYYY-MM-DD."""
    if hasattr(period, "strftime"):
        return period.strftime("%Y-%m-%d")
    return str(period)


def _current_price(tk: "yf.Ticker") -> float | None:
    """Best-effort current price, trying fast_info then info then history."""
    # fast_info: cheapest and most reliable when it works.
    try:
        fast = tk.fast_info
        for key in ("last_price", "lastPrice", "regular_market_price"):
            try:
                price = _to_num(fast[key])
            except (KeyError, TypeError):
                price = None
            if price is not None:
                return price
        # FastInfo also exposes attribute access in some versions.
        price = _to_num(getattr(fast, "last_price", None))
        if price is not None:
            return price
    except Exception:
        pass

    # info: heavier call, but carries currentPrice / regularMarketPrice.
    try:
        info = tk.info or {}
        for key in ("currentPrice", "regularMarketPrice", "previousClose"):
            price = _to_num(info.get(key))
            if price is not None:
                return price
    except Exception:
        pass

    # history: last resort — the most recent close.
    try:
        hist = tk.history(period="5d")
        if hist is not None and not hist.empty and "Close" in hist:
            return _to_num(hist["Close"].dropna().iloc[-1])
    except Exception:
        pass

    return None


def _dividends(tk: "yf.Ticker") -> list[dict]:
    """Full dividend history as ``[{date, amount}]``, oldest → newest."""
    try:
        series = tk.dividends
    except Exception:
        return []
    if series is None or series.empty:
        return []

    out: list[dict] = []
    for date, amount in series.items():
        value = _to_num(amount)
        if value is None:
            continue
        out.append({"date": _period_label(date), "amount": value})
    return out


def _statement(df, limit: int = _ANNUAL_PERIODS) -> list[dict]:
    """Normalize a yfinance financial DataFrame to a list of period records.

    Columns are period-end dates (most recent first); rows are line
    items. Returns ``[{period, data: {line_item: value|None}}]`` for up
    to ``limit`` most-recent periods. Empty/missing → ``[]``.
    """
    if df is None or getattr(df, "empty", True):
        return []

    records: list[dict] = []
    for period in list(df.columns)[:limit]:
        series = df[period]
        data = {str(item): _to_num(val) for item, val in series.items()}
        records.append({"period": _period_label(period), "data": data})
    return records


def _company_meta(tk: "yf.Ticker") -> dict:
    """Best-effort descriptive metadata (name, sector, currency)."""
    meta = {"company_name": None, "sector": None, "currency": None}
    try:
        info = tk.info or {}
    except Exception:
        return meta
    meta["company_name"] = info.get("longName") or info.get("shortName")
    meta["sector"] = info.get("sector")
    meta["currency"] = info.get("currency")
    return meta


def _fetch_impl(ticker: str, yf_symbol: str, clean_ticker: str) -> dict:
    """The actual yfinance work; runs inside the timeout harness below."""
    tk = yf.Ticker(yf_symbol)

    dividends = _dividends(tk)
    current_price = _current_price(tk)
    income_statement = _statement(getattr(tk, "income_stmt", None))
    balance_sheet = _statement(getattr(tk, "balance_sheet", None))
    cash_flow = _statement(getattr(tk, "cashflow", None))

    # A real NSE ticker yields at least a price or some financial data.
    # If everything is empty, the symbol is almost certainly invalid.
    if (
        current_price is None
        and not dividends
        and not income_statement
        and not balance_sheet
        and not cash_flow
    ):
        raise InvalidTickerError(
            f"'{ticker}' did not return any data on NSE (tried '{yf_symbol}'). "
            "Check the spelling, or use the full NSE ticker (e.g. 'ITC', "
            "'COALINDIA', 'SBIN')."
        )

    meta = _company_meta(tk)

    return {
        "ticker": clean_ticker,
        "yf_symbol": yf_symbol,
        "company_name": meta["company_name"],
        "sector": meta["sector"],
        "currency": meta["currency"],
        "current_price": current_price,
        "dividends": dividends,
        "income_statement": income_statement,
        "balance_sheet": balance_sheet,
        "cash_flow": cash_flow,
        "data_timestamp": datetime.now(IST).isoformat(),
    }


@disk_cached()
def fetch_company_data(ticker: str) -> dict:
    """Fetch and normalize all market data for one NSE ticker.

    Appends ``.NS``, pulls dividend history, current price, and the last
    four annual income statements, balance sheets, and cash-flow
    statements. Result is JSON-serializable and cached on disk for one
    hour (``config.CACHE_TTL_SECONDS``) keyed on the ticker.

    The whole fetch runs under a wall-clock cap
    (``config.YFINANCE_TIMEOUT_SECONDS``) so a hung Yahoo endpoint can
    never stall the pipeline — ``DataFetchTimeoutError`` is raised
    instead. Errors and timeouts are never cached; only successful
    payloads are.

    Raises ``InvalidTickerError`` if the symbol yields no usable data
    (bad ticker, delisted, or not on NSE).
    """
    if not ticker or not ticker.strip():
        raise InvalidTickerError(
            "No ticker provided. Pass an NSE symbol such as 'ITC' or 'INFY'."
        )

    base = ticker.strip().upper()
    # Tolerate a user who already appended the suffix.
    yf_symbol = base if base.endswith(".NS") else f"{base}.NS"
    clean_ticker = yf_symbol[:-3]  # display ticker without ".NS"

    timeout = config.YFINANCE_TIMEOUT_SECONDS
    executor = ThreadPoolExecutor(max_workers=1)
    try:
        future = executor.submit(_fetch_impl, ticker, yf_symbol, clean_ticker)
        return future.result(timeout=timeout)
    except FuturesTimeout:
        raise DataFetchTimeoutError(
            f"Fetching '{clean_ticker}' timed out after {timeout}s — Yahoo Finance "
            "may be slow or unreachable. Please try again in a moment."
        ) from None
    finally:
        # Don't block on a hung worker; yfinance's own socket timeouts will
        # eventually reap it. cancel_futures is a no-op once it's running.
        executor.shutdown(wait=False, cancel_futures=True)


if __name__ == "__main__":
    import json
    import sys

    # Windows consoles often default to cp1252, which cannot encode ₹/…;
    # force UTF-8 so this demo prints cleanly everywhere.
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

    print("Fetching ITC data from NSE via yfinance…\n")
    try:
        data = fetch_company_data("ITC")
    except InvalidTickerError as exc:
        print(f"InvalidTickerError: {exc}")
    else:
        print(json.dumps(data, indent=2, default=str))
        print(
            f"\nSummary: {data['company_name']} ({data['ticker']}) — "
            f"price ₹{data['current_price']}, "
            f"{len(data['dividends'])} dividend records, "
            f"{len(data['income_statement'])} annual income statements. "
            f"Data as of {data['data_timestamp']}."
        )
