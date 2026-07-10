"""Tier 4 — Presentation: Streamlit UI.

Single-page app with a natural-language question box and example chips.
Renders the direct-answer banner (for dividend_qa), forecast card
(amount range, expected window, colour-coded confidence badge, reasoning,
risks), key metrics table, Plotly dividend-history chart, the "data as of
<timestamp>" caption, the standard disclaimer, and an expandable
agent-trace panel. Handles clarify / out_of_scope / error results
gracefully.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from src import config
from src.ratio_engine import _annual_dividends  # FY-attributed dividend totals

# ── page + minimal styling ───────────────────────────────────────────
st.set_page_config(page_title="DiviSense AI", page_icon="📈", layout="centered")

st.markdown(
    """
    <style>
    .ds-badge{color:#fff;padding:2px 12px;border-radius:12px;font-size:0.85rem;font-weight:600;}
    .ds-answer{border-left:4px solid #0969da;background:rgba(9,105,218,0.08);
               padding:14px 18px;border-radius:6px;margin:10px 0;font-size:1.05rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

EXAMPLES = [
    "Will Infosys increase its dividend next quarter?",
    "Forecast ITC's dividend for next year",
    "COALINDIA",
]

# Friendly per-node captions shown while the graph streams.
NODE_STEPS = {
    "intent_node": "🧭 Understanding your question…",
    "data_node": "📥 Fetching live market data (yfinance)…",
    "ratio_node": "🧮 Computing dividend metrics (deterministic)…",
    "rag_node": "📚 Retrieving annual-report context…",
    "forecast_node": "🔮 Forecasting the dividend…",
    "critic_node": "🔍 Validating every number against the metrics…",
    "report_node": "📝 Composing the report…",
}

_CONF_COLORS = {"high": "#1a7f37", "medium": "#9a6700", "low": "#cf222e"}


# ── small formatters ─────────────────────────────────────────────────
def _amount_range(amount_range) -> str:
    if isinstance(amount_range, dict):
        low, high = amount_range.get("low"), amount_range.get("high")
        if low is not None and high is not None:
            return f"₹{low} – ₹{high}"
        if low is not None:
            return f"₹{low}"
    if isinstance(amount_range, (int, float)):
        return f"₹{amount_range}"
    if isinstance(amount_range, str) and amount_range.strip():
        return amount_range
    return "—"


def _confidence_badge(conf) -> str:
    key = (conf or "").lower()
    color = _CONF_COLORS.get(key, "#57606a")
    label = (conf or "—").title() if conf else "—"
    return f'<span class="ds-badge" style="background:{color}">{label} confidence</span>'


def _fmt(value, prefix="", suffix=""):
    if value is None:
        return "—"
    if isinstance(value, float):
        value = round(value, 2)
    return f"{prefix}{value}{suffix}"


# ── pipeline execution (streamed, with per-node progress) ────────────
def execute(query: str) -> dict:
    """Stream the LangGraph pipeline, showing per-node progress; return final state."""
    from src.graph import get_graph  # heavy imports deferred until a query runs

    initial = {"user_query": query, "retry_count": 0, "llm_calls": 0, "errors": [], "rag_context": []}
    state = dict(initial)

    with st.status("Running the DiviSense agent pipeline…", expanded=True) as status:
        try:
            for chunk in get_graph().stream(initial, stream_mode="updates"):
                for node, update in chunk.items():
                    st.write(NODE_STEPS.get(node, f"… {node}"))
                    if update:
                        state.update(update)
            status.update(label="✅ Analysis complete", state="complete", expanded=False)
        except Exception as exc:  # noqa: BLE001 - surface as a friendly message
            state.setdefault("errors", []).append(f"Unexpected error: {exc}")
            status.update(label="⚠️ Something went wrong", state="error")

    return state


# ── result renderers ─────────────────────────────────────────────────
def render_direct_answer(forecast: dict) -> None:
    direct = forecast.get("direct_answer")
    if not direct:
        return
    likelihood = forecast.get("likelihood")
    extra = f"<br><small>Likelihood: <strong>{likelihood}</strong></small>" if likelihood else ""
    st.markdown(f'<div class="ds-answer">💬 {direct}{extra}</div>', unsafe_allow_html=True)


def render_forecast_card(forecast: dict) -> None:
    st.subheader("🔮 Dividend forecast")
    st.metric("Expected dividend (next FY)", _amount_range(forecast.get("amount_range_inr")))
    st.markdown(f"**Expected window:** {forecast.get('expected_window') or '—'}")
    st.markdown(_confidence_badge(forecast.get("confidence")), unsafe_allow_html=True)

    reasoning = forecast.get("reasoning") or []
    if reasoning:
        st.markdown("**Reasoning**")
        for r in reasoning:
            st.markdown(f"- {r}")

    risks = forecast.get("risks") or []
    if risks:
        st.markdown("**Risks**")
        for r in risks:
            st.markdown(f"- {r}")


def render_metrics(metrics: dict) -> None:
    if not metrics:
        return
    st.subheader("📊 Key metrics")
    payout = metrics.get("payout_ratio_5yr") or {}
    traj = metrics.get("recent_trajectory") or {}
    de = metrics.get("debt_to_equity_trend") or {}
    last_fy = metrics.get("last_fiscal_year")

    rows = [
        ("Current yield", _fmt(metrics.get("current_yield_pct"), suffix=" %")),
        (
            f"Last-FY dividend (FY{last_fy})" if last_fy else "Last-FY dividend",
            _fmt(metrics.get("total_dividends_last_fy"), prefix="₹", suffix=" / share"),
        ),
        ("Payout ratio (avg)", _fmt(payout.get("average_pct"), suffix=" %")),
        ("Dividend CAGR (5y)", _fmt(metrics.get("dividend_cagr_5yr"), suffix=" %")),
        ("FCF / dividend coverage", _fmt(metrics.get("fcf_dividend_coverage"), suffix="×")),
        ("Consecutive-increase streak", _fmt(metrics.get("consecutive_increase_streak"), suffix=" yr")),
        ("Dividend consistency score", _fmt(metrics.get("dividend_consistency_score"), suffix=" / 100")),
        ("Recent trajectory", traj.get("classification") or "—"),
        ("Debt/equity trend", de.get("direction") or "—"),
        ("Years of dividend history", _fmt(metrics.get("years_of_history"))),
    ]
    df = pd.DataFrame(rows, columns=["Metric", "Value"]).set_index("Metric")
    st.table(df)

    warnings = metrics.get("warnings") or []
    if warnings:
        st.caption("⚠️ " + " ".join(warnings))


def render_chart(raw_data: dict | None) -> None:
    dividends = (raw_data or {}).get("dividends") or []
    annual = _annual_dividends(dividends)  # [(fy, total, count)] ascending
    if not annual:
        st.caption("No dividend history available to chart.")
        return

    last10 = annual[-10:]
    df = pd.DataFrame(
        {
            "Fiscal Year": [f"FY{fy}" for fy, _, _ in last10],
            "Dividend (₹/share)": [round(total, 2) for _, total, _ in last10],
        }
    )
    st.subheader("📉 Dividend history (last 10 FYs)")
    fig = px.bar(df, x="Fiscal Year", y="Dividend (₹/share)", text="Dividend (₹/share)")
    fig.update_traces(marker_color="#0969da", textposition="outside")
    fig.update_layout(height=340, margin=dict(l=10, r=10, t=10, b=10), yaxis_title="₹ / share")
    st.plotly_chart(fig, width="stretch")


def render_agent_trace(state: dict) -> None:
    metrics = state.get("metrics") or {}
    rag = state.get("rag_context") or []
    critique = state.get("critique") or {}

    with st.expander("🔍 Agent trace"):
        st.markdown(f"**Intent:** `{state.get('intent')}`  →  ticker `{state.get('ticker')}`")
        if state.get("horizon"):
            st.markdown(f"**Horizon:** {state.get('horizon')}")
        if state.get("question"):
            st.markdown(f"**Question:** {state.get('question')}")
        st.markdown(
            f"**Metrics:** {len(metrics)} fields computed, "
            f"{len(metrics.get('warnings') or [])} warning(s)"
        )
        st.markdown(f"**RAG context:** {len(rag)} annual-report snippet(s)")
        for s in rag:
            st.caption(f"• {s.get('source_file')} p{s.get('page')} (score {_fmt(s.get('score'))})")
        if critique:
            verdict = "approved ✅" if critique.get("approved") else "rejected ❌"
            st.markdown(f"**Critic:** {verdict}")
            for issue in critique.get("issues") or []:
                st.caption(f"• {issue}")
        st.markdown(f"**Forecast retries:** {state.get('retry_count', 0)}")
        if state.get("forecast"):
            st.markdown("**Forecast (raw JSON):**")
            st.json(state["forecast"])


def render_clarify(state: dict) -> None:
    st.warning(
        f'🤔 I couldn\'t identify a single NSE-listed company in: _"{state.get("user_query", "")}"_'
    )
    st.markdown(
        "DiviSense AI answers about **one company at a time**. Try a ticker like "
        "`ITC`, `COALINDIA`, `INFY`, or a question like "
        '_"Will Infosys increase its dividend next year?"_'
    )


def render_out_of_scope(state: dict) -> None:
    st.info(
        f'🧭 _"{state.get("user_query", "")}"_ looks like a multi-company or screener-style question.'
    )
    st.markdown(
        "DiviSense AI is a **single-company** dividend tool — screeners and rankings "
        '(e.g. "top PSU dividend payers") are on the roadmap, not in this version. '
        "Ask about one company, e.g. `COALINDIA`."
    )


def render_error(state: dict) -> None:
    for err in state.get("errors") or ["Something went wrong."]:
        st.error(err)
    st.caption(
        "Check the ticker/spelling and try again. If it's an LLM rate-limit, wait a "
        "moment — results are cached, so a retry is cheap."
    )


def render(state: dict) -> None:
    intent = state.get("intent")

    if intent == "clarify":
        render_clarify(state)
        return
    if intent == "out_of_scope":
        render_out_of_scope(state)
        return
    if state.get("errors") and not state.get("forecast"):
        render_error(state)
        return
    if not state.get("forecast"):
        render_error(state)
        return

    # Forecast / dividend_qa result.
    st.header(f"📈 {state.get('ticker')}")
    render_direct_answer(state["forecast"])
    render_forecast_card(state["forecast"])
    render_metrics(state.get("metrics") or {})
    render_chart(state.get("raw_data"))
    if state.get("data_timestamp"):
        st.caption(f"Data as of {state['data_timestamp']}")
    if state.get("errors"):  # e.g. critic failed but a forecast exists
        st.caption("⚠️ Validation was incomplete: " + "; ".join(state["errors"]))
    render_agent_trace(state)


# ── page body ────────────────────────────────────────────────────────
def main() -> None:
    st.title("📈 DiviSense AI")
    st.markdown("_Agentic dividend forecasting for NSE-listed Indian companies._")
    st.warning(config.DISCLAIMER)

    with st.form("query_form"):
        query_input = st.text_input(
            "Ask DiviSense",
            placeholder="Enter a ticker or ask a question…",
            label_visibility="collapsed",
        )
        submitted = st.form_submit_button("Analyze dividend →", type="primary")

    st.caption("Or try an example:")
    chip_cols = st.columns(len(EXAMPLES))
    for col, example in zip(chip_cols, EXAMPLES):
        if col.button(example, width="stretch"):
            st.session_state["pending_query"] = example

    if submitted and query_input.strip():
        st.session_state["pending_query"] = query_input.strip()

    pending = st.session_state.get("pending_query")
    if not pending:
        st.info("👆 Enter a ticker (e.g. `ITC`) or ask a dividend question to begin.")
        return

    # Only re-run the pipeline when the query actually changes; otherwise
    # re-render the stored result (so toggling the trace doesn't re-run LLMs).
    if st.session_state.get("result_query") != pending:
        st.session_state["result_state"] = execute(pending)
        st.session_state["result_query"] = pending

    render(st.session_state["result_state"])


main()
