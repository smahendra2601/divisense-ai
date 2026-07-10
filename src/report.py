"""Tier 3 — Report Node: assemble the final markdown output.

``build_report(state)`` turns a ``DivisenseState`` into user-facing
markdown. It handles four shapes of result:

* **forecast_single / dividend_qa** — direct-answer banner (when the
  Forecast Agent gave one), forecast card (amount range, expected window,
  colour-coded confidence, reasoning, risks), a key-metrics table,
  corporate actions from the CSV source, the data-as-of timestamp, an
  agent trace, and the standard disclaimer.
* **clarify** — a friendly "please rephrase" message echoing the query.
* **out_of_scope** — explains the single-company scope and points to the
  roadmap.
* **error** — a kind, human-readable error message (never a stack trace).

Every branch ends with the disclaimer from ``config.DISCLAIMER``.
"""

from __future__ import annotations

import json

from . import config
from .corp_actions import get_default_source

_CONFIDENCE_BADGE = {"high": "🟢 High", "medium": "🟡 Medium", "low": "🔴 Low"}


# ── small formatting helpers ─────────────────────────────────────────
def _fmt_num(value, prefix: str = "", suffix: str = "") -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        value = round(value, 2)
    return f"{prefix}{value}{suffix}"


def _fmt_amount_range(amount_range) -> str:
    if isinstance(amount_range, dict):
        low, high = amount_range.get("low"), amount_range.get("high")
        if low is not None and high is not None:
            return f"₹{low} – ₹{high} per share"
        if low is not None:
            return f"₹{low} per share"
    if isinstance(amount_range, (int, float)):
        return f"₹{amount_range} per share"
    if isinstance(amount_range, str) and amount_range.strip():
        return amount_range
    return "—"


def _disclaimer() -> str:
    return f"---\n\n_{config.DISCLAIMER}_"


# ── metrics + corporate-actions tables ───────────────────────────────
def _metrics_table(metrics: dict) -> str:
    if not metrics:
        return "_No metrics available._"

    payout = metrics.get("payout_ratio_5yr") or {}
    traj = metrics.get("recent_trajectory") or {}
    de = metrics.get("debt_to_equity_trend") or {}

    rows = [
        ("Current yield", _fmt_num(metrics.get("current_yield_pct"), suffix=" %")),
        (
            f"Last-FY dividend (FY{metrics.get('last_fiscal_year')})"
            if metrics.get("last_fiscal_year")
            else "Last-FY dividend",
            _fmt_num(metrics.get("total_dividends_last_fy"), prefix="₹", suffix=" / share"),
        ),
        ("Payout ratio (avg)", _fmt_num(payout.get("average_pct"), suffix=" %")),
        ("Dividend CAGR (5y)", _fmt_num(metrics.get("dividend_cagr_5yr"), suffix=" %")),
        ("FCF / dividend coverage", _fmt_num(metrics.get("fcf_dividend_coverage"), suffix="×")),
        ("Consecutive-increase streak", _fmt_num(metrics.get("consecutive_increase_streak"), suffix=" yr")),
        ("Dividend consistency score", _fmt_num(metrics.get("dividend_consistency_score"), suffix=" / 100")),
        ("Recent trajectory", traj.get("classification") or "—"),
        ("Debt/equity trend", de.get("direction") or "—"),
        ("Years of dividend history", _fmt_num(metrics.get("years_of_history"))),
    ]

    lines = ["| Metric | Value |", "| --- | --- |"]
    lines += [f"| {name} | {value} |" for name, value in rows]

    warnings = metrics.get("warnings") or []
    if warnings:
        lines.append("")
        lines.append("> ⚠️ Data notes: " + " ".join(warnings))
    return "\n".join(lines)


def _corporate_actions_table(ticker: str) -> str:
    try:
        actions = get_default_source().get_actions(ticker)
    except Exception:
        actions = []
    if not actions:
        return "_No corporate actions on file._"

    lines = [
        "| Action | Amount (₹) | Announced | Ex-date | Record | Note |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for a in actions:
        lines.append(
            f"| {a.get('action_type') or '—'} | {_fmt_num(a.get('amount'))} | "
            f"{a.get('announcement_date') or '—'} | {a.get('ex_date') or '—'} | "
            f"{a.get('record_date') or '—'} | {a.get('source_note') or '—'} |"
        )
    return "\n".join(lines)


# ── agent trace ──────────────────────────────────────────────────────
def _agent_trace(state: dict) -> str:
    metrics = state.get("metrics") or {}
    rag = state.get("rag_context") or []
    critique = state.get("critique") or {}

    lines = ["## 🔍 Agent trace", ""]
    lines.append(f"- **Intent:** `{state.get('intent')}`  →  ticker `{state.get('ticker')}`")
    if state.get("horizon"):
        lines.append(f"- **Horizon:** {state.get('horizon')}")
    lines.append(
        f"- **Metrics:** {len(metrics)} fields computed"
        + (f" ({len(metrics.get('warnings') or [])} warning(s))" if metrics else "")
    )
    lines.append(f"- **RAG context:** {len(rag)} annual-report snippet(s)")
    if rag:
        for s in rag:
            lines.append(f"    - {s.get('source_file')} p{s.get('page')} (score {_fmt_num(s.get('score'))})")
    if critique:
        verdict = "approved ✅" if critique.get("approved") else "rejected ❌"
        lines.append(f"- **Critic:** {verdict}")
        for issue in critique.get("issues") or []:
            lines.append(f"    - {issue}")
    lines.append(f"- **Forecast retries:** {state.get('retry_count', 0)}")
    lines.append(f"- **LLM calls used:** {state.get('llm_calls', 0)} (budget: {config.MAX_LLM_CALLS_PER_QUERY})")
    return "\n".join(lines)


# ── message branches ─────────────────────────────────────────────────
def _clarify_report(state: dict) -> str:
    query = state.get("user_query") or ""
    return (
        "## 🤔 Could you clarify?\n\n"
        f"I couldn't identify a single NSE-listed company to analyse in: _\"{query}\"_.\n\n"
        "DiviSense AI answers about **one company at a time**. Try:\n\n"
        "- a ticker — e.g. `ITC`, `COALINDIA`, `INFY`\n"
        "- or a question — e.g. _\"Will Infosys increase its dividend next year?\"_\n\n"
        + _disclaimer()
    )


def _out_of_scope_report(state: dict) -> str:
    query = state.get("user_query") or ""
    return (
        "## 🧭 Out of scope (for now)\n\n"
        f"_\"{query}\"_ looks like a multi-company or screener-style question.\n\n"
        "DiviSense AI is a **single-company** dividend tool — screeners and rankings "
        "(e.g. \"top PSU dividend payers\") are on the roadmap, not in this version. "
        "Ask about one company, e.g. `COALINDIA` or _\"Is ITC's dividend sustainable?\"_.\n\n"
        + _disclaimer()
    )


def _error_report(state: dict) -> str:
    errors = state.get("errors") or ["Something went wrong."]
    ticker = state.get("ticker")
    heading = f"## ⚠️ Couldn't complete the analysis{f' for {ticker}' if ticker else ''}\n\n"
    # Keep the card readable: full provider payloads live in the logs.
    trimmed = [e if len(e) <= 220 else e[:220] + "…" for e in errors]
    body = "\n".join(f"- {e}" for e in trimmed)
    tip = (
        "\n\nPlease check the ticker/spelling and try again. If it's an LLM rate-limit, "
        "wait a moment — results are cached, so a retry is cheap."
    )
    return heading + body + tip + "\n\n" + _disclaimer()


# ── forecast / dividend_qa branch ────────────────────────────────────
def _forecast_report(state: dict) -> str:
    forecast = state.get("forecast") or {}
    ticker = state.get("ticker") or "?"
    metrics = state.get("metrics") or {}

    parts: list[str] = [f"# 📈 DiviSense AI — {ticker}"]

    # Direct-answer banner (dividend_qa).
    direct = forecast.get("direct_answer")
    likelihood = forecast.get("likelihood")
    if direct:
        banner = f"> ## 💬 {direct}"
        if likelihood:
            banner += f"\n>\n> **Likelihood:** {likelihood}"
        parts.append(banner)

    # Forecast card.
    confidence = (forecast.get("confidence") or "").lower()
    badge = _CONFIDENCE_BADGE.get(confidence, forecast.get("confidence") or "—")
    card = ["## 🔮 Dividend forecast", ""]
    card.append(f"- **Expected amount:** {_fmt_amount_range(forecast.get('amount_range_inr'))}")
    card.append(f"- **Expected window:** {forecast.get('expected_window') or '—'}")
    card.append(f"- **Confidence:** {badge}")
    reasoning = forecast.get("reasoning") or []
    if reasoning:
        card.append("")
        card.append("**Reasoning:**")
        card += [f"- {r}" for r in reasoning]
    risks = forecast.get("risks") or []
    if risks:
        card.append("")
        card.append("**Risks:**")
        card += [f"- {r}" for r in risks]
    parts.append("\n".join(card))

    # Metrics + corporate actions.
    parts.append("## 📊 Key metrics\n\n" + _metrics_table(metrics))
    parts.append("## 🗓️ Corporate actions\n\n" + _corporate_actions_table(ticker))

    # Timestamp.
    ts = state.get("data_timestamp")
    if ts:
        parts.append(f"_Data as of {ts}_")

    # If the critic step errored but we still have a forecast, note it.
    if state.get("errors"):
        parts.append("> ⚠️ Note: validation step was incomplete — " + "; ".join(state["errors"]))

    parts.append(_agent_trace(state))
    parts.append(_disclaimer())
    return "\n\n".join(parts)


# ── public entry point ───────────────────────────────────────────────
def build_report(state: dict) -> str:
    """Render the final markdown report for a completed pipeline state."""
    intent = state.get("intent")

    if intent == "clarify":
        return _clarify_report(state)
    if intent == "out_of_scope":
        return _out_of_scope_report(state)

    # Error with nothing to show → error card. If a forecast exists despite a
    # late (e.g. critic) error, fall through and show it with a note.
    if state.get("errors") and not state.get("forecast"):
        return _error_report(state)

    if state.get("forecast"):
        return _forecast_report(state)

    # Fallback: no forecast, no explicit message — treat as error.
    return _error_report(state)


if __name__ == "__main__":
    import sys

    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

    # Tiny offline demo with a hand-built state (no LLM/network).
    demo_state = {
        "user_query": "ITC",
        "intent": "forecast_single",
        "ticker": "ITC",
        "metrics": {
            "current_yield_pct": 5.12,
            "total_dividends_last_fy": 14.5,
            "last_fiscal_year": 2026,
            "payout_ratio_5yr": {"average_pct": 79.3},
            "dividend_cagr_5yr": 6.0,
            "recent_trajectory": {"classification": "mixed"},
            "years_of_history": 31,
            "warnings": [],
        },
        "forecast": {
            "amount_range_inr": {"low": 14.0, "high": 15.5},
            "expected_window": "Interim ~Feb 2027, final ~Jun 2027",
            "confidence": "medium",
            "reasoning": ["Last-FY dividend was ₹14.5", "5y CAGR ~6%"],
            "risks": ["Cigarette-tax changes could pressure payout"],
        },
        "critique": {"approved": True, "issues": []},
        "rag_context": [],
        "retry_count": 0,
        "data_timestamp": "2026-07-09T12:00:00+05:30",
    }
    print(build_report(demo_state))
