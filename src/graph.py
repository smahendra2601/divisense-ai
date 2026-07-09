"""Tier 3 — Agentic Orchestration: the LangGraph StateGraph.

Wires the whole pipeline (ARCHITECTURE.md §3):

    intent → [forecast_single | dividend_qa] → data → ratio → rag →
    forecast → critic → report

Routing:
- ``intent_node`` classifies the query; ``forecast_single`` /
  ``dividend_qa`` proceed to ``data_node``, while ``clarify`` /
  ``out_of_scope`` (and any error) go straight to ``report_node``.
- ``critic_node`` may loop back to ``forecast_node`` exactly once
  (when it rejects and ``retry_count == 0``), injecting its critique;
  otherwise it goes to ``report_node``.
- An error recorded at *any* node routes to ``report_node``, which
  renders a human-readable message — never a stack trace.

Golden rules enforced here: the LLM interprets deterministic metrics but
never computes them; at most 3 LLM calls per query (intent + forecast +
critic), 2 for a bare ticker; every output carries a timestamp and the
disclaimer (added by ``report``).
"""

from __future__ import annotations

import json
import logging
from typing import Optional, TypedDict

from langgraph.graph import END, START, StateGraph

from . import data_agent, intent, llm_router, ratio_engine, rag, report

logger = logging.getLogger(__name__)


class DivisenseState(TypedDict, total=False):
    """Shared state threaded through the graph (ARCHITECTURE.md §3)."""

    user_query: str
    intent: str
    ticker: Optional[str]
    question: Optional[str]
    horizon: Optional[str]
    raw_data: Optional[dict]
    metrics: Optional[dict]
    rag_context: list
    forecast: Optional[dict]
    critique: Optional[dict]
    retry_count: int
    final_report: Optional[str]
    errors: list
    data_timestamp: Optional[str]


# ── prompt schemas ───────────────────────────────────────────────────
_FORECAST_SCHEMA = (
    '{"direct_answer": "<string, or null>", '
    '"likelihood": "likely | unlikely | unclear, or null", '
    '"amount_range_inr": {"low": <number>, "high": <number>}, '
    '"expected_window": "<timing text>", '
    '"confidence": "high | medium | low", '
    '"reasoning": ["<string>", "..."], '
    '"risks": ["<string>", "..."]}'
)

_CRITIC_SCHEMA = '{"approved": true or false, "issues": ["<string>", "..."]}'


def _append_error(state: dict, message: str) -> dict:
    return {"errors": (state.get("errors") or []) + [message]}


# ── prompt builders ──────────────────────────────────────────────────
def _forecast_prompt(state: dict) -> str:
    metrics_json = json.dumps(state.get("metrics") or {}, indent=2, default=str)
    rag_context = state.get("rag_context") or []
    if rag_context:
        rag_text = "\n\n".join(
            f"[{s.get('source_file')} p{s.get('page')}] {s.get('text')}" for s in rag_context
        )
    else:
        rag_text = "(No annual-report context available for this company.)"

    ticker = state.get("ticker")
    horizon = state.get("horizon") or "the next fiscal year"

    base = (
        f"You are the dividend Forecast Agent for DiviSense AI, analysing the "
        f"NSE-listed company {ticker}.\n\n"
        "You are given DETERMINISTIC metrics computed in code. Use ONLY numbers that "
        "appear in these metrics — never invent, recall, or compute new figures.\n\n"
        f"METRICS (JSON):\n{metrics_json}\n\n"
        f"ANNUAL-REPORT CONTEXT (may be empty):\n{rag_text}\n\n"
        "INDIAN DIVIDEND CADENCE RULE: most NSE companies pay an INTERIM dividend "
        "(ex-date ~Oct–Mar) and a FINAL dividend after the March year-end "
        "(ex-date ~Jun–Aug) — NOT US-style quarterly dividends. If the user refers to "
        "\"next quarter\", map it to this company's actual interim/final cadence and "
        "say so explicitly.\n\n"
        "HOW TO BUILD amount_range_inr: anchor it on `total_dividends_last_fy` — do not "
        "propose a range far below it without reason. Then adjust for direction using "
        "`dividend_cagr_5yr`, `recent_trajectory`, and `consecutive_increase_streak`: "
        "if trajectory is rising with a positive streak, the range should skew AT OR "
        "ABOVE last year's total (e.g. last_fy to last_fy × (1 + cagr%)); if trajectory "
        "is falling or the payout ratio is stretched (>90%) with weak FCF coverage "
        "(<1×), the range should skew AT OR BELOW last year's total instead; if mixed or "
        "unclear, center the range near last year's total with modest width. The low end "
        "should never be implausibly small (e.g. near zero) unless the metrics show "
        "genuine distress (falling trajectory AND weak coverage AND stretched payout).\n"
    )

    if state.get("intent") == "dividend_qa":
        task = (
            f"\nThe user asked: {state.get('question')!r}\nHorizon: {horizon}\n\n"
            "Answer the user's question FIRST and directly — set `direct_answer` to a "
            "clear sentence and `likelihood` to likely/unlikely/unclear — THEN support it "
            "with the forecast and metrics. Apply the cadence rule explicitly if the "
            "question mentions \"next quarter\".\n"
        )
    else:
        task = (
            f"\nTask: forecast {ticker}'s TOTAL dividend per share for the next fiscal "
            f"year ({horizon}) as a range (`amount_range_inr` low/high, ₹ per share). "
            "State the likely interim/final split and the timing window "
            "(`expected_window`). `direct_answer` and `likelihood` may be null.\n"
        )

    critique = state.get("critique")
    retry_note = ""
    if critique and not critique.get("approved"):
        issues = "; ".join(critique.get("issues") or [])
        retry_note = (
            "\nA PRIOR DRAFT WAS REJECTED by the critic for these issues — fix them and "
            f"do not repeat the mistakes:\n{issues}\n"
        )

    return base + task + retry_note


def _critic_prompt(state: dict) -> str:
    metrics_json = json.dumps(state.get("metrics") or {}, default=str)
    forecast_json = json.dumps(state.get("forecast") or {}, default=str)
    return (
        "You are the Critic Agent for DiviSense AI. Verify the Forecast Agent's output "
        "against the deterministic metrics.\n\n"
        f"METRICS (JSON): {metrics_json}\n\n"
        f"FORECAST (JSON): {forecast_json}\n\n"
        "Check, and list any problems in `issues`:\n"
        "1. Every numeric claim in the forecast's reasoning traces to a value present in "
        "METRICS. The forecast RANGE itself is a projection, but it must be consistent "
        "with `total_dividends_last_fy` and the growth signals (`recent_trajectory`, "
        "`dividend_cagr_5yr`, `consecutive_increase_streak`).\n"
        "2. The `direct_answer` / `likelihood` is consistent with the metrics. Flag "
        "contradictions — e.g. claiming a \"likely increase\" while `recent_trajectory` is "
        "\"falling\" AND the payout ratio is stretched, UNLESS the reasoning explicitly "
        "acknowledges and addresses that tension.\n"
        "3. Flag any number that does not appear in, or follow from, the metrics.\n\n"
        "Set `approved` true only if there are no material issues."
    )


# ── nodes ────────────────────────────────────────────────────────────
def intent_node(state: dict) -> dict:
    parsed = intent.parse_query(state.get("user_query", ""))
    return {
        "intent": parsed.get("intent"),
        "ticker": parsed.get("ticker"),
        "question": parsed.get("question"),
        "horizon": parsed.get("horizon"),
    }


def data_node(state: dict) -> dict:
    ticker = state.get("ticker")
    try:
        raw = data_agent.fetch_company_data(ticker)
    except data_agent.InvalidTickerError as exc:
        return _append_error(state, f"Couldn't find market data for '{ticker}': {exc}")
    except Exception as exc:  # noqa: BLE001
        return _append_error(state, f"Data fetch failed for '{ticker}': {exc}")
    return {"raw_data": raw, "data_timestamp": raw.get("data_timestamp")}


def ratio_node(state: dict) -> dict:
    try:
        metrics = ratio_engine.compute_metrics(state.get("raw_data") or {})
    except Exception as exc:  # noqa: BLE001
        return _append_error(state, f"Metric computation failed: {exc}")
    return {"metrics": metrics}


def rag_node(state: dict) -> dict:
    # rag.retrieve is fail-soft (returns []), so this never blocks the pipeline.
    return {"rag_context": rag.retrieve(state.get("ticker") or "")}


def forecast_node(state: dict) -> dict:
    try:
        forecast = llm_router.invoke_json(
            _forecast_prompt(state), _FORECAST_SCHEMA, task_type="reasoning"
        )
    except Exception as exc:  # noqa: BLE001
        return _append_error(state, f"Forecast step failed: {exc}")

    updates = {"forecast": forecast}
    critique = state.get("critique")
    if critique and not critique.get("approved"):
        # We only reach here on a retry; count it so the critic loops at most once.
        updates["retry_count"] = (state.get("retry_count") or 0) + 1
    return updates


def critic_node(state: dict) -> dict:
    try:
        critique = llm_router.invoke_json(
            _critic_prompt(state), _CRITIC_SCHEMA, task_type="reasoning"
        )
    except Exception as exc:  # noqa: BLE001
        return _append_error(state, f"Critic step failed: {exc}")
    return {"critique": critique}


def report_node(state: dict) -> dict:
    return {"final_report": report.build_report(state)}


# ── routers ──────────────────────────────────────────────────────────
def route_after_intent(state: dict) -> str:
    if state.get("errors"):
        return "report_node"
    return "data_node" if state.get("intent") in ("forecast_single", "dividend_qa") else "report_node"


def _error_or(next_node: str):
    def router(state: dict) -> str:
        return "report_node" if state.get("errors") else next_node

    return router


def route_after_critic(state: dict) -> str:
    if state.get("errors"):
        return "report_node"
    critique = state.get("critique") or {}
    if critique.get("approved"):
        return "report_node"
    if (state.get("retry_count") or 0) == 0:
        return "forecast_node"  # loop back once with the critique injected
    return "report_node"


# ── graph assembly ───────────────────────────────────────────────────
def build_graph():
    """Build and compile the DiviSense StateGraph."""
    g = StateGraph(DivisenseState)

    g.add_node("intent_node", intent_node)
    g.add_node("data_node", data_node)
    g.add_node("ratio_node", ratio_node)
    g.add_node("rag_node", rag_node)
    g.add_node("forecast_node", forecast_node)
    g.add_node("critic_node", critic_node)
    g.add_node("report_node", report_node)

    g.add_edge(START, "intent_node")
    g.add_conditional_edges(
        "intent_node", route_after_intent, {"data_node": "data_node", "report_node": "report_node"}
    )
    g.add_conditional_edges(
        "data_node", _error_or("ratio_node"),
        {"ratio_node": "ratio_node", "report_node": "report_node"},
    )
    g.add_conditional_edges(
        "ratio_node", _error_or("rag_node"),
        {"rag_node": "rag_node", "report_node": "report_node"},
    )
    g.add_conditional_edges(
        "rag_node", _error_or("forecast_node"),
        {"forecast_node": "forecast_node", "report_node": "report_node"},
    )
    g.add_conditional_edges(
        "forecast_node", _error_or("critic_node"),
        {"critic_node": "critic_node", "report_node": "report_node"},
    )
    g.add_conditional_edges(
        "critic_node", route_after_critic,
        {"forecast_node": "forecast_node", "report_node": "report_node"},
    )
    g.add_edge("report_node", END)

    return g.compile()


# Compile once at import; reused across queries.
_graph = None


def get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


def run_pipeline(user_query: str) -> dict:
    """Run one query end-to-end and return the final ``DivisenseState``."""
    initial: dict = {
        "user_query": user_query,
        "retry_count": 0,
        "errors": [],
        "rag_context": [],
    }
    return get_graph().invoke(initial)


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

    query = " ".join(sys.argv[1:]) or "ITC"
    final_state = run_pipeline(query)
    print("\n" + (final_state.get("final_report") or "(no report produced)"))
