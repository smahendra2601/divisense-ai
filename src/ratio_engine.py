"""Tier 2 — Intelligence: deterministic financial-ratio computation.

Pure pandas/Python — **NO LLM involvement**. All financial numbers are
computed here, in code, so the LLM never invents or recalls a figure.

``compute_metrics(raw_data)`` consumes the dict returned by
``data_agent.fetch_company_data`` and returns a structured ``metrics``
dict. Every metric degrades gracefully: if its inputs are missing it is
set to ``None`` and an explanatory string is appended to
``metrics["warnings"]`` — the function never raises on incomplete data.

Fiscal-year attribution of dividends
------------------------------------
NSE companies typically pay an **interim** dividend during the fiscal
year (ex-date Oct–Mar) and a **final** dividend *after* the March
year-end (ex-date Apr–Sep). yfinance only gives ex-dates, so to line up
with how Screener.in reports "dividend for FY N" we attribute each
payout to its **results fiscal year**:

    results_fy(ex_date) = ex_date.year + 1   if ex_date.month >= 10   (Oct–Dec interim)
                          ex_date.year        otherwise               (Jan–Sep interim/final)

Example (ITC): final ex 2025-05-28 (₹7.85) and interim ex 2025-02-12
(₹6.50) both map to FY2025 → ₹14.35, matching Screener. This heuristic
suits the standard interim+final cadence; unusual schedules or special
dividends may be attributed a year off, which is documented rather than
hidden.

Dividend-consistency score (0–100)
----------------------------------
Computed over up to the last 10 completed fiscal years with any
dividend history::

    pay_rate    = (fiscal years with a dividend > 0) / (years in span)
    no_cut_rate = (year-over-year steps that did NOT cut the dividend)
                  / (number of year-over-year steps)
    score       = round(100 * (0.5 * pay_rate + 0.5 * no_cut_rate))

A perfect 100 means the company paid every year in the span and never
reduced its annual dividend. Needs at least two dividend-paying fiscal
years; otherwise ``None`` with a warning.
"""

from __future__ import annotations

import math
from datetime import date

# Relative tolerance for treating two dividend figures as "flat".
_FLAT_TOL = 0.01
# Look-back window (fiscal years) for the consistency score.
_CONSISTENCY_WINDOW = 10

# Line-item alias tables — yfinance labels vary by company/statement.
_NET_INCOME_KEYS = [
    "Net Income",
    "Net Income Common Stockholders",
    "Net Income Continuous Operations",
    "Net Income From Continuing Operations",
    "Net Income Including Noncontrolling Interests",
]
_DIVIDENDS_PAID_KEYS = ["Cash Dividends Paid", "Common Stock Dividend Paid"]
_FCF_KEYS = ["Free Cash Flow"]
_OCF_KEYS = ["Operating Cash Flow", "Total Cash From Operating Activities"]
_CAPEX_KEYS = ["Capital Expenditure"]
_TOTAL_DEBT_KEYS = ["Total Debt"]
_LONG_DEBT_KEYS = ["Long Term Debt", "Long Term Debt And Capital Lease Obligation"]
_SHORT_DEBT_KEYS = ["Current Debt", "Current Debt And Capital Lease Obligation"]
_EQUITY_KEYS = [
    "Stockholders Equity",
    "Common Stock Equity",
    "Total Equity Gross Minority Interest",
]


# ── small helpers ────────────────────────────────────────────────────
def _num(value) -> float | None:
    """Coerce to a finite float, else None."""
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return None if (math.isnan(f) or math.isinf(f)) else f


def _get(data: dict, keys: list[str]) -> float | None:
    """First finite value among ``keys`` in a statement's data dict."""
    if not data:
        return None
    for key in keys:
        val = _num(data.get(key))
        if val is not None:
            return val
    return None


def _round(value: float | None, ndigits: int = 2) -> float | None:
    return None if value is None else round(value, ndigits)


def _results_fy(iso_date: str) -> int | None:
    """Map a dividend ex-date to its results fiscal year (see module docs)."""
    try:
        y, m, d = (int(p) for p in iso_date.split("-")[:3])
        _ = date(y, m, d)  # validate
    except (ValueError, AttributeError):
        return None
    return y + 1 if m >= 10 else y


def _period_year(period: str) -> int | None:
    try:
        return int(period.split("-")[0])
    except (ValueError, AttributeError):
        return None


def _annual_dividends(dividends: list[dict]) -> list[tuple[int, float, int]]:
    """Aggregate raw payout events into (fiscal_year, total, event_count).

    Ascending by fiscal year; only fiscal years with a positive total are
    returned. ``event_count`` is how many payouts landed in that FY — used
    to spot a most-recent FY that is still being collected.
    """
    totals: dict[int, float] = {}
    counts: dict[int, int] = {}
    for item in dividends:
        amt = _num(item.get("amount"))
        fy = _results_fy(item.get("date", ""))
        if amt is None or fy is None or amt <= 0:
            continue
        totals[fy] = totals.get(fy, 0.0) + amt
        counts[fy] = counts.get(fy, 0) + 1
    return [(fy, totals[fy], counts[fy]) for fy in sorted(totals)]


def _split_provisional(
    annual_full: list[tuple[int, float, int]],
) -> tuple[list[tuple[int, float]], tuple[int, float, int] | None]:
    """Separate a still-collecting most-recent FY from the complete history.

    NSE finals for a just-closed FY often go ex-dividend months later, so
    mid-year the newest FY can show only its interim(s). If the latest FY
    has fewer payout events than the median of the last few prior years (at
    least three prior years to judge), it is treated as *provisional* and
    held back from the money metrics. The comparison uses only recent prior
    years, so a company that changed its payout cadence over time is judged
    against its current rhythm rather than its whole history. Returns
    ``(complete, provisional_or_None)`` where ``complete`` is
    ``[(fy, total)]``.
    """
    if len(annual_full) >= 4:
        *prior, last = annual_full
        import statistics

        recent_prior = prior[-5:]  # judge against current cadence, not ancient history
        median_prior = statistics.median(c for _, _, c in recent_prior)
        if last[2] < median_prior:
            complete = [(fy, t) for fy, t, _ in annual_full[:-1]]
            return complete, last
    return [(fy, t) for fy, t, _ in annual_full], None


def _classify(amounts: list[float]) -> str | None:
    """Classify a chronological series as rising/flat/falling/mixed."""
    if len(amounts) < 2:
        return None
    ups = downs = 0
    for prev, cur in zip(amounts, amounts[1:]):
        if prev == 0:
            if cur > 0:
                ups += 1
            continue
        rel = (cur - prev) / prev
        if rel > _FLAT_TOL:
            ups += 1
        elif rel < -_FLAT_TOL:
            downs += 1
    if ups and not downs:
        return "rising"
    if downs and not ups:
        return "falling"
    if not ups and not downs:
        return "flat"
    return "mixed"


# ── the public entry point ───────────────────────────────────────────
def compute_metrics(raw_data: dict) -> dict:
    """Compute the full deterministic metrics dict for one company.

    See the module docstring for fiscal-year attribution and the
    consistency-score formula. Missing inputs never raise: the affected
    metric is ``None`` and a note is appended to ``metrics["warnings"]``.
    """
    warnings: list[str] = []
    raw_data = raw_data or {}

    metrics: dict = {
        "payout_ratio_5yr": None,
        "dividend_cagr_5yr": None,
        "fcf_dividend_coverage": None,
        "consecutive_increase_streak": None,
        "current_yield_pct": None,
        "debt_to_equity_trend": None,
        "dividend_consistency_score": None,
        "total_dividends_last_fy": None,
        "years_of_history": None,
        "recent_trajectory": None,
        "last_fiscal_year": None,
        "provisional_fiscal_year": None,
        "warnings": warnings,
    }

    dividends = raw_data.get("dividends") or []
    income = raw_data.get("income_statement") or []
    cash = raw_data.get("cash_flow") or []
    balance = raw_data.get("balance_sheet") or []
    price = _num(raw_data.get("current_price"))

    annual_full = _annual_dividends(dividends)  # [(fy, total, count)] ascending
    annual, provisional = _split_provisional(annual_full)  # [(fy, total)], (fy,total,count)|None

    # years_of_history counts every dividend-paying FY (incl. provisional).
    if annual_full:
        metrics["years_of_history"] = len(annual_full)
    else:
        warnings.append("No dividend history found; dividend-based metrics unavailable.")

    if provisional is not None:
        prov_fy, prov_total, prov_events = provisional
        metrics["provisional_fiscal_year"] = {
            "fiscal_year": prov_fy,
            "partial_total": _round(prov_total),
            "events_so_far": prov_events,
        }
        warnings.append(
            f"FY{prov_fy} looks partial ({prov_events} payout(s), ₹{_round(prov_total)} "
            "so far) — final dividend likely not yet ex-dividend. Money metrics use the "
            "last complete fiscal year instead."
        )

    # last complete fiscal year --------------------------------------
    if annual:
        last_fy, last_fy_total = annual[-1]
        metrics["last_fiscal_year"] = last_fy
        metrics["total_dividends_last_fy"] = _round(last_fy_total)

    # recent_trajectory (last 4 annual fiscal-year totals) ------------
    # "Last 4 payouts" is interpreted as the last 4 annual dividend
    # totals: raw interim/final events zig-zag and would almost always
    # read as "mixed", which cannot power a "will it increase?" answer.
    if len(annual) >= 2:
        recent = annual[-4:]
        classification = _classify([t for _, t in recent])
        metrics["recent_trajectory"] = {
            "classification": classification,
            "fiscal_years": [fy for fy, _ in recent],
            "amounts": [_round(t) for _, t in recent],
            "based_on": "annual fiscal-year dividend totals (last 4 FYs)",
        }
    else:
        warnings.append(
            "Fewer than 2 dividend-paying fiscal years; recent trajectory unavailable."
        )

    # consecutive_increase_streak (annual totals, most recent back) ---
    if len(annual) >= 2:
        streak = 0
        for (_, prev), (_, cur) in zip(reversed(annual[:-1]), reversed(annual)):
            # walk newest → oldest: count while each FY > the one before it
            if prev == 0:
                break
            if (cur - prev) / prev > _FLAT_TOL:
                streak += 1
            else:
                break
        metrics["consecutive_increase_streak"] = streak
    elif annual:
        metrics["consecutive_increase_streak"] = 0

    # dividend_cagr_5yr (annual totals over up to last 5 FYs) ---------
    if len(annual) >= 2:
        window = annual[-5:]
        (begin_fy, begin_val), (end_fy, end_val) = window[0], window[-1]
        span = end_fy - begin_fy
        if begin_val and begin_val > 0 and span > 0:
            cagr = (end_val / begin_val) ** (1 / span) - 1
            metrics["dividend_cagr_5yr"] = _round(cagr * 100)
        else:
            warnings.append(
                "Dividend CAGR unavailable (zero base year or single-year span)."
            )
    else:
        warnings.append("Insufficient dividend history for CAGR.")

    # current_yield_pct ----------------------------------------------
    if metrics["total_dividends_last_fy"] is not None and price:
        metrics["current_yield_pct"] = _round(
            metrics["total_dividends_last_fy"] / price * 100
        )
    else:
        if price is None or price == 0:
            warnings.append("Current price missing; current yield unavailable.")

    # dividend_consistency_score -------------------------------------
    if len(annual) >= 2:
        window = annual[-_CONSISTENCY_WINDOW:]
        span_years = window[-1][0] - window[0][0] + 1
        years_paid = len(window)
        pay_rate = min(years_paid / span_years, 1.0) if span_years > 0 else 0.0
        steps = list(zip(window[:-1], window[1:]))
        no_cut = sum(
            1 for (_, prev), (_, cur) in steps if cur >= prev * (1 - _FLAT_TOL)
        )
        no_cut_rate = no_cut / len(steps) if steps else 1.0
        metrics["dividend_consistency_score"] = round(
            100 * (0.5 * pay_rate + 0.5 * no_cut_rate)
        )
    else:
        warnings.append("Insufficient history for a consistency score.")

    # payout_ratio_5yr (Cash Dividends Paid / Net Income, per FY) -----
    cash_by_period = {r.get("period"): r.get("data", {}) for r in cash}
    payout_rows: list[dict] = []
    for rec in income:
        period = rec.get("period")
        net_income = _get(rec.get("data", {}), _NET_INCOME_KEYS)
        div_paid = _get(cash_by_period.get(period, {}), _DIVIDENDS_PAID_KEYS)
        if net_income and net_income > 0 and div_paid is not None:
            payout_rows.append(
                {"period": period, "payout_ratio_pct": _round(abs(div_paid) / net_income * 100)}
            )
        else:
            payout_rows.append({"period": period, "payout_ratio_pct": None})
    valid = [r["payout_ratio_pct"] for r in payout_rows if r["payout_ratio_pct"] is not None]
    if valid:
        metrics["payout_ratio_5yr"] = {
            "per_year": sorted(payout_rows, key=lambda r: r["period"] or ""),
            "average_pct": _round(sum(valid) / len(valid)),
        }
    else:
        warnings.append(
            "Payout ratio unavailable (missing net income or cash dividends paid)."
        )

    # fcf_dividend_coverage (latest year: FCF / dividends paid) -------
    if cash:
        latest = cash[0].get("data", {})
        fcf = _get(latest, _FCF_KEYS)
        if fcf is None:
            ocf = _get(latest, _OCF_KEYS)
            capex = _get(latest, _CAPEX_KEYS)
            if ocf is not None and capex is not None:
                fcf = ocf + capex  # capex is reported negative
        div_paid = _get(latest, _DIVIDENDS_PAID_KEYS)
        if fcf is not None and div_paid and abs(div_paid) > 0:
            metrics["fcf_dividend_coverage"] = _round(fcf / abs(div_paid))
        else:
            warnings.append(
                "FCF/dividend coverage unavailable (missing FCF or dividends paid)."
            )
    else:
        warnings.append("No cash-flow statement; FCF/dividend coverage unavailable.")

    # debt_to_equity_trend (last 3 years) ----------------------------
    de_rows: list[dict] = []
    for rec in balance[:3]:
        data = rec.get("data", {})
        total_debt = _get(data, _TOTAL_DEBT_KEYS)
        if total_debt is None:
            lt = _get(data, _LONG_DEBT_KEYS) or 0.0
            st = _get(data, _SHORT_DEBT_KEYS) or 0.0
            total_debt = lt + st if (lt or st) else None
        equity = _get(data, _EQUITY_KEYS)
        de = _round(total_debt / equity) if (total_debt is not None and equity and equity > 0) else None
        de_rows.append({"period": rec.get("period"), "debt_to_equity": de})
    de_rows.sort(key=lambda r: r["period"] or "")
    de_values = [r for r in de_rows if r["debt_to_equity"] is not None]
    if de_values:
        oldest, newest = de_values[0]["debt_to_equity"], de_values[-1]["debt_to_equity"]
        if newest > oldest * 1.05:
            direction = "rising"
        elif newest < oldest * 0.95:
            direction = "falling"
        else:
            direction = "stable"
        metrics["debt_to_equity_trend"] = {"per_year": de_rows, "direction": direction}
    else:
        warnings.append("Debt-to-equity unavailable (missing debt or equity).")

    return metrics


if __name__ == "__main__":
    import sys

    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

    from .data_agent import InvalidTickerError, fetch_company_data

    tickers = sys.argv[1:] or ["ITC", "TCS", "COALINDIA", "HCLTECH", "INFY"]

    def _fmt(v, width, spec=""):
        return ("{:" + str(width) + spec + "}").format(v if v is not None else "—")

    header = (
        f"{'Ticker':<10}{'Yield%':>8}{'LastFY₹':>9}{'Payout%':>9}"
        f"{'CAGR5y%':>9}{'FCFcov':>8}{'Streak':>8}{'Consist':>8}"
        f"{'Trajectory':>12}{'YrsHist':>9}"
    )
    print("\nComputed metrics (verify against Screener.in):\n")
    print(header)
    print("-" * len(header))

    for tk in tickers:
        try:
            m = compute_metrics(fetch_company_data(tk))
        except InvalidTickerError as exc:
            print(f"{tk:<10}  ERROR: {exc}")
            continue

        yield_ = m["current_yield_pct"]
        last_fy = m["total_dividends_last_fy"]
        payout = m["payout_ratio_5yr"]["average_pct"] if m["payout_ratio_5yr"] else None
        cagr = m["dividend_cagr_5yr"]
        fcf = m["fcf_dividend_coverage"]
        streak = m["consecutive_increase_streak"]
        consist = m["dividend_consistency_score"]
        traj = m["recent_trajectory"]["classification"] if m["recent_trajectory"] else None
        yrs = m["years_of_history"]

        row = (
            f"{tk:<10}"
            f"{(f'{yield_:.2f}' if yield_ is not None else '—'):>8}"
            f"{(f'{last_fy:.2f}' if last_fy is not None else '—'):>9}"
            f"{(f'{payout:.1f}' if payout is not None else '—'):>9}"
            f"{(f'{cagr:.1f}' if cagr is not None else '—'):>9}"
            f"{(f'{fcf:.2f}' if fcf is not None else '—'):>8}"
            f"{(str(streak) if streak is not None else '—'):>8}"
            f"{(str(consist) if consist is not None else '—'):>8}"
            f"{(traj or '—'):>12}"
            f"{(str(yrs) if yrs is not None else '—'):>9}"
        )
        print(row)

    print("\nNote: dividends attributed to results fiscal year (interim + final).")
    print("Payout% = |Cash Dividends Paid| / Net Income, averaged over available FYs.")
