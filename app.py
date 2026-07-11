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

import logging

import pandas as pd
import plotly.express as px
import streamlit as st

from src import config
from src.ratio_engine import _annual_dividends  # FY-attributed dividend totals

# streamlit run app.py never triggers any module's __main__ guard, so this
# is the only place that configures logging for the whole process — without
# it, llm_router's routing/quota-skip logs and the logger.exception() calls
# in graph.py are silently dropped instead of reaching Render's log stream.
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

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
    /* ── base chrome: clean, demo-ready ─────────────────────────── */
    #MainMenu, footer {visibility: hidden;}
    [data-testid="stAppDeployButton"] {display: none !important;}  /* remove Deploy */
    [data-testid="stDecoration"] {display: none;}                  /* top accent bar */
    [data-testid="stHeader"] {background: transparent;}
    [data-testid="stAppViewContainer"] {
        background: radial-gradient(1100px 550px at 85% -10%, #1a1c20 0%, #0e0f11 50%, #0a0b0d 100%);
    }
    /* main panel fills the space left of the sidebar */
    .block-container {
        padding-top: 2rem; padding-bottom: 3rem;
        padding-left: 3rem; padding-right: 3rem;
        max-width: 100%;
    }

    /* typography */
    h1, h2, h3 {letter-spacing: -0.4px;}
    [data-testid="stMainBlockContainer"] h1 {font-size: 2.6rem; font-weight: 800;}
    [data-testid="stMainBlockContainer"] h2 {font-weight: 750;}

    /* ── graphite sidebar ───────────────────────────────────────── */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0c0d0f 0%, #141619 100%);
        min-width: 410px;
        border-right: 1px solid rgba(255,255,255,0.08);
    }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3, [data-testid="stSidebar"] h4,
    [data-testid="stSidebar"] p, [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] li, [data-testid="stSidebar"] summary {
        color: #ededf0 !important;
    }
    /* sidebar type matches the main panel body size (1rem = baseFontSize) */
    [data-testid="stSidebar"] p, [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] li, [data-testid="stSidebar"] summary,
    [data-testid="stSidebar"] button p {
        font-size: 1rem;
    }
    [data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {
        font-size: 0.9rem; color: #9aa0a8 !important;
    }
    [data-testid="stSidebar"] hr {border-color: rgba(255,255,255,0.14);}
    /* question box: solid dark field + white text so typing is always
       visible (semi-transparent backgrounds inherit the theme's white
       widget background and go white-on-white) */
    [data-testid="stSidebar"] [data-baseweb="textarea"] {
        background: #1d1f23;
        border: 1px solid rgba(255,255,255,0.22); border-radius: 10px;
    }
    [data-testid="stSidebar"] textarea {
        background: #1d1f23 !important; color: #ffffff !important;
        caret-color: #e6bc63; font-size: 1rem;
    }
    [data-testid="stSidebar"] textarea::placeholder {color: rgba(255,255,255,0.45);}

    /* brand lockup — big logo + wordmark */
    .ds-brand {display: flex; align-items: center; gap: 14px; margin: 2px 0 6px;}
    .ds-brand-logo {
        font-size: 2.9rem; line-height: 1;
        filter: drop-shadow(0 3px 8px rgba(0,0,0,0.35));
    }
    .ds-brand-name {
        font-size: 2.05rem; font-weight: 800; letter-spacing: -0.6px;
        color: #ffffff; line-height: 1.02;
    }
    .ds-brand-name span {
        background: linear-gradient(90deg, #d9a441, #f0d08a);
        -webkit-background-clip: text; background-clip: text; color: transparent;
    }
    .ds-brand-tag {font-size: 0.92rem; color: #a3a9b2 !important; margin-top: 3px;}

    /* sidebar buttons */
    [data-testid="stSidebar"] button[kind="primary"],
    [data-testid="stSidebar"] [data-testid="stBaseButton-primary"] {
        background: linear-gradient(90deg, #c9962f, #e6bc63);
        border: none; border-radius: 10px; font-weight: 700;
        box-shadow: 0 3px 12px rgba(201,150,47,0.35);
        transition: transform .06s ease, box-shadow .2s ease;
    }
    [data-testid="stSidebar"] button[kind="primary"] p,
    [data-testid="stSidebar"] [data-testid="stBaseButton-primary"] p {
        color: #17130a !important;   /* dark text on the gold CTA */
    }
    [data-testid="stSidebar"] button[kind="primary"]:hover,
    [data-testid="stSidebar"] [data-testid="stBaseButton-primary"]:hover {
        box-shadow: 0 5px 18px rgba(201,150,47,0.55); transform: translateY(-1px);
    }
    [data-testid="stSidebar"] button[kind="secondary"],
    [data-testid="stSidebar"] [data-testid="stBaseButton-secondary"] {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.14);
        color: #e2e4e8; text-align: left; font-weight: 500;
        white-space: normal; border-radius: 10px;
    }
    [data-testid="stSidebar"] button[kind="secondary"]:hover,
    [data-testid="stSidebar"] [data-testid="stBaseButton-secondary"]:hover {
        background: rgba(255,255,255,0.10);
        border-color: rgba(230,188,99,0.55); color: #ffffff;
    }

    /* ── main-panel surfaces ────────────────────────────────────── */
    [data-testid="stMetric"] {
        border: 1px solid rgba(255,255,255,0.09); border-radius: 14px;
        padding: 16px 18px; background: #17181b;
        box-shadow: 0 2px 8px rgba(0,0,0,0.35);
    }
    [data-testid="stMetricValue"] {font-weight: 750;}

    .ds-card {
        border: 1px solid rgba(255,255,255,0.09); border-radius: 14px;
        padding: 16px 20px; background: #17181b;
        box-shadow: 0 2px 8px rgba(0,0,0,0.35); margin-bottom: 8px;
    }
    .ds-card h4 {margin: 0 0 8px 0; color: #eceef1;}

    /* direct-answer banner */
    .ds-answer {
        border-left: 5px solid #d9a441;
        background: linear-gradient(90deg, rgba(217,164,65,0.13), rgba(217,164,65,0.03));
        padding: 16px 20px; border-radius: 10px;
        margin: 4px 0 14px 0; font-size: 1.12rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.35);
    }

    /* confidence badge */
    .ds-badge {
        color: #fff; padding: 3px 14px; border-radius: 14px;
        font-size: 0.9rem; font-weight: 600; display: inline-block;
    }

    /* intent chip in the header */
    .ds-chip {
        background: rgba(217,164,65,0.16); color: #e6c078;
        padding: 2px 12px; border-radius: 12px; font-size: 0.85rem;
        font-weight: 600; display: inline-block; margin-left: 8px;
        vertical-align: middle;
    }

    /* expanders as clean cards — scoped to the MAIN panel only */
    [data-testid="stMainBlockContainer"] [data-testid="stExpander"] {
        border: 1px solid rgba(255,255,255,0.09); border-radius: 12px;
        background: #17181b; box-shadow: 0 2px 8px rgba(0,0,0,0.3);
    }
    /* sidebar expander ("How it works"): its own dark surface so the
       light text reads crisply instead of washing out */
    [data-testid="stSidebar"] [data-testid="stExpander"] {
        border: 1px solid rgba(255,255,255,0.18); border-radius: 10px;
        background: rgba(255,255,255,0.045);
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

_CONF_COLORS = {"high": "#2da44e", "medium": "#bf8700", "low": "#e5534b"}  # bright enough for dark bg


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
        st.markdown(
            "<div class='ds-brand'>"
            "<div class='ds-brand-logo'>📈</div>"
            "<div><div class='ds-brand-name'>DiviSense<span> AI</span></div>"
            "<div class='ds-brand-tag'>Agentic dividend forecasting · NSE</div></div>"
            "</div>",
            unsafe_allow_html=True,
        )
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
        except Exception:  # noqa: BLE001 - surface as a friendly message
            logger.exception("execute(): pipeline run failed for query %r", query)
            state.setdefault("errors", []).append(
                "Something went wrong while running the analysis. Try again in a moment."
            )
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
    fig.update_traces(marker_color="#d9a441", textposition="outside")
    fig.update_layout(
        height=360,
        margin=dict(l=10, r=10, t=10, b=10),
        yaxis_title="₹ / share",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="#d7dae0",
    )
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.08)")
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
    news_items = state.get("news_context") or []
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
        st.markdown(f"**News context:** {len(news_items)} recent article(s)")
        for n in news_items:
            st.caption(f"• [{n.get('title')}]({n.get('url')})")
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
