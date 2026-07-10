"""Tier 4 — Presentation: Streamlit UI.

Dashboard layout: a branded navy sidebar owns all input (question box,
example queries that pre-fill the box, the Analyze button, and a short
"how it works"), while the full-width main panel renders the result —
company header with live price, direct-answer banner (dividend_qa),
forecast cards, reasoning/risks, key-metrics table beside the Plotly
dividend-history chart, corporate actions, the "data as of" caption,
an expandable agent trace, and the standard disclaimer. Clarify /
out-of-scope / error results render as friendly cards.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from src import config
from src.ratio_engine import _annual_dividends  # FY-attributed dividend totals

# ── page + styling ───────────────────────────────────────────────────
st.set_page_config(
    page_title="DiviSense AI",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    /* hide default chrome for a clean demo */
    #MainMenu, footer {visibility: hidden;}

    /* navy sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0a2b45 0%, #10395c 100%);
        min-width: 340px;
    }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3, [data-testid="stSidebar"] h4,
    [data-testid="stSidebar"] p, [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] li, [data-testid="stSidebar"] summary {
        color: #eef4fa !important;
    }
    [data-testid="stSidebar"] hr {border-color: rgba(255,255,255,0.25);}

    /* metric tiles as cards */
    [data-testid="stMetric"] {
        border: 1px solid rgba(49,51,63,0.14);
        border-radius: 12px;
        padding: 14px 16px;
        background: rgba(9,105,218,0.04);
        box-shadow: 0 1px 3px rgba(16,24,40,0.06);
    }

    /* generic card */
    .ds-card {
        border: 1px solid rgba(49,51,63,0.14);
        border-radius: 12px;
        padding: 16px 20px;
        background: rgba(9,105,218,0.03);
        box-shadow: 0 1px 3px rgba(16,24,40,0.06);
        margin-bottom: 8px;
    }
    .ds-card h4 {margin: 0 0 8px 0;}

    /* direct-answer banner */
    .ds-answer {
        border-left: 5px solid #0969da;
        background: rgba(9,105,218,0.08);
        padding: 16px 20px;
        border-radius: 8px;
        margin: 4px 0 14px 0;
        font-size: 1.12rem;
    }

    /* confidence badge */
    .ds-badge {
        color: #fff; padding: 3px 14px; border-radius: 14px;
        font-size: 0.9rem; font-weight: 600; display: inline-block;
    }

    /* intent chip in the header */
    .ds-chip {
        background: rgba(9,105,218,0.12); color: #0a3d62;
        padding: 2px 12px; border-radius: 12px; font-size: 0.8rem;
        font-weight: 600; display: inline-block; margin-left: 8px;
        vertical-align: middle;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

EXAMPLES = [
    "Will Infosys increase its dividend next quarter?",
    "Forecast ITC's dividend for next year",
    "COALINDIA",
]

NODE_STEPS = {
    "intent_node": "🧭 Understanding your question…",
    "data_node": "📥 Fetching live market data (yfinance)…",
    "ratio_node": "🧮 Computing dividend metrics (deterministic)…",
    "rag_node": "📚 Retrieving annual-report context…",
    "news_node": "📰 Checking recent news…",
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


# ── sidebar (all input lives here) ───────────────────────────────────
def _fill_example(example: str) -> None:
    st.session_state["query_box"] = example


def _submit_query() -> None:
    query = (st.session_state.get("query_box") or "").strip()
    if query:
        st.session_state["pending_query"] = query


def render_sidebar() -> None:
    with st.sidebar:
        st.markdown("## 📈 DiviSense AI")
        st.markdown("Agentic dividend forecasting for NSE-listed Indian companies.")
        st.divider()

        st.markdown("#### Ask DiviSense")
        st.text_area(
            "Your question",
            key="query_box",
            placeholder="Enter a ticker or ask a question…",
            label_visibility="collapsed",
            height=90,
        )
        st.button(
            "Analyze dividend →",
            type="primary",
            width="stretch",
            on_click=_submit_query,
        )

        st.markdown("**Or try an example** *(fills the box above)*")
        for i, example in enumerate(EXAMPLES):
            st.button(
                example,
                key=f"example_{i}",
                width="stretch",
                on_click=_fill_example,
                args=(example,),
            )

        st.divider()
        with st.expander("ℹ️ How it works"):
            st.markdown(
                "1. **Intent** — parses your question; the ticker is resolved "
                "by a lookup table, never guessed by AI.\n"
                "2. **Data** — live Yahoo Finance fetch (1-hour cache).\n"
                "3. **Metrics** — pure-code financial ratios; no AI arithmetic.\n"
                "4. **RAG** — annual-report context, when ingested.\n"
                "5. **Forecast + Critic** — an AI forecast, cross-checked by a "
                "second AI pass. At most 3 AI calls per query.\n"
                "6. **Report** — reasoning, confidence, and timestamp, always."
            )
        st.caption("Research tool — not investment advice.")


# ── pipeline execution (streamed, with per-node progress) ────────────
def execute(query: str) -> dict:
    """Stream the LangGraph pipeline, showing per-node progress; return final state."""
    from src.graph import get_graph  # heavy imports deferred until a query runs

    initial = {
        "user_query": query, "retry_count": 0, "llm_calls": 0,
        "errors": [], "rag_context": [], "news_context": [],
    }
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
def render_company_header(state: dict) -> None:
    raw = state.get("raw_data") or {}
    ticker = state.get("ticker") or "?"
    name = raw.get("company_name") or ticker
    intent = state.get("intent")

    left, right = st.columns([4, 1])
    with left:
        st.markdown(
            f"## {name} <span class='ds-chip'>{ticker}</span>"
            + (f"<span class='ds-chip'>{intent}</span>" if intent else ""),
            unsafe_allow_html=True,
        )
        sector = raw.get("sector")
        if sector:
            st.caption(sector)
    with right:
        price = raw.get("current_price")
        if price is not None:
            st.metric("Current price", f"₹{round(price, 2)}")


def render_direct_answer(forecast: dict) -> None:
    direct = forecast.get("direct_answer")
    if not direct:
        return
    likelihood = forecast.get("likelihood")
    extra = f"<br><small>Likelihood: <strong>{likelihood}</strong></small>" if likelihood else ""
    st.markdown(f'<div class="ds-answer">💬 {direct}{extra}</div>', unsafe_allow_html=True)


def render_forecast_card(forecast: dict) -> None:
    st.subheader("🔮 Dividend forecast")
    c1, c2, c3 = st.columns([1.2, 1.4, 1])
    with c1:
        st.metric("Expected dividend (next FY)", _amount_range(forecast.get("amount_range_inr")))
    with c2:
        st.markdown(
            "<div class='ds-card'><h4>🗓️ Expected window</h4>"
            f"{forecast.get('expected_window') or '—'}</div>",
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            "<div class='ds-card'><h4>Confidence</h4>"
            f"{_confidence_badge(forecast.get('confidence'))}</div>",
            unsafe_allow_html=True,
        )

    r1, r2 = st.columns(2)
    reasoning = forecast.get("reasoning") or []
    risks = forecast.get("risks") or []
    with r1:
        if reasoning:
            st.markdown("**Reasoning**")
            for r in reasoning:
                st.markdown(f"- {r}")
    with r2:
        if risks:
            st.markdown("**⚠️ Risks**")
            for r in risks:
                st.markdown(f"- {r}")


def render_metrics(metrics: dict) -> None:
    if not metrics:
        return
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
    fig = px.bar(df, x="Fiscal Year", y="Dividend (₹/share)", text="Dividend (₹/share)")
    fig.update_traces(marker_color="#0969da", textposition="outside")
    fig.update_layout(
        height=360,
        margin=dict(l=10, r=10, t=10, b=10),
        yaxis_title="₹ / share",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})


def render_corporate_actions(ticker: str) -> None:
    from src.corp_actions import get_default_source

    try:
        actions = get_default_source().get_actions(ticker)
    except Exception:
        actions = []
    with st.expander("🗓️ Corporate actions on file"):
        if not actions:
            st.caption("No corporate actions on file.")
            return
        df = pd.DataFrame(actions)
        df = df.rename(
            columns={
                "action_type": "Action",
                "amount": "Amount (₹)",
                "announcement_date": "Announced",
                "ex_date": "Ex-date",
                "record_date": "Record",
                "source_note": "Note",
            }
        ).drop(columns=["ticker"], errors="ignore")
        st.dataframe(df, width="stretch", hide_index=True)


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
        st.markdown(
            f"**LLM calls used:** {state.get('llm_calls', 0)} "
            f"(budget: {config.MAX_LLM_CALLS_PER_QUERY})"
        )
        if state.get("forecast"):
            st.markdown("**Forecast (raw JSON):**")
            st.json(state["forecast"], expanded=False)


def _select_suggestion(ticker: str) -> None:
    # Selection = confirmation: fill the sidebar box and run immediately.
    st.session_state["query_box"] = ticker
    st.session_state["pending_query"] = ticker


def render_clarify(state: dict) -> None:
    suggestions = state.get("suggestions") or []
    if suggestions:
        st.warning(
            f'🤔 I couldn\'t find an exact NSE match for: _"{state.get("user_query", "")}"_'
        )
        st.markdown("**Did you mean one of these?** Click to run the analysis:")
        cols = st.columns(min(len(suggestions), 3))
        for i, s in enumerate(suggestions):
            with cols[i % len(cols)]:
                st.button(
                    f"{s.get('ticker')} — {s.get('company_name')}",
                    key=f"suggestion_{i}",
                    width="stretch",
                    on_click=_select_suggestion,
                    args=(s.get("ticker"),),
                )
                st.caption(f"match {round((s.get('score') or 0) * 100)}%")
        return
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


def render_welcome() -> None:
    st.title("📈 DiviSense AI")
    st.markdown(
        "#### Agentic dividend forecasting for NSE-listed Indian companies"
    )
    st.markdown(
        "Use the **sidebar** to enter a ticker (like `ITC`) or ask a question in "
        "plain English — or click an example to pre-fill the box."
    )
    st.markdown("")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            "<div class='ds-card'><h4>🧮 Deterministic numbers</h4>"
            "Every financial metric is computed in code from live market data. "
            "The AI interprets numbers — it never invents them.</div>",
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            "<div class='ds-card'><h4>🤖 Agentic pipeline</h4>"
            "Intent parsing, live data, ratio engine, annual-report retrieval, "
            "an AI forecaster and an AI critic — orchestrated with LangGraph.</div>",
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            "<div class='ds-card'><h4>🔍 Transparent output</h4>"
            "Reasoning chain, confidence level, data timestamp, and a full "
            "agent trace on every single answer.</div>",
            unsafe_allow_html=True,
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
    render_company_header(state)
    render_direct_answer(state["forecast"])
    render_forecast_card(state["forecast"])

    st.divider()
    left, right = st.columns([1, 1.2])
    with left:
        st.subheader("📊 Key metrics")
        render_metrics(state.get("metrics") or {})
    with right:
        st.subheader("📉 Dividend history (last 10 FYs)")
        render_chart(state.get("raw_data"))

    render_corporate_actions(state.get("ticker") or "")
    render_agent_trace(state)

    if state.get("data_timestamp"):
        st.caption(f"Data as of {state['data_timestamp']}")
    if state.get("errors"):  # e.g. critic failed but a forecast exists
        st.caption("⚠️ Validation was incomplete: " + "; ".join(state["errors"]))


# ── page body ────────────────────────────────────────────────────────
def main() -> None:
    render_sidebar()

    pending = st.session_state.get("pending_query")
    if not pending:
        render_welcome()
        st.warning(config.DISCLAIMER)
        return

    # Only re-run the pipeline when the query actually changes; otherwise
    # re-render the stored result (so toggling the trace doesn't re-run LLMs).
    if st.session_state.get("result_query") != pending:
        st.session_state["result_state"] = execute(pending)
        st.session_state["result_query"] = pending

    render(st.session_state["result_state"])
    st.warning(config.DISCLAIMER)


main()
