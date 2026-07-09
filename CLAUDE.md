# CLAUDE.md — DiviSense AI project context

Claude Code automatically reads this file. Keep it short; it points to the binding docs.

## What this project is
Agentic dividend forecasting platform for Indian (NSE) stocks — **single-company tool**.
Accepts a ticker OR a natural-language question about one company, e.g.
"Will Infosys increase its dividend next quarter?" or "Forecast ITC's dividend for
next year". LangGraph pipeline: Intent Agent (query → intent/ticker/question) →
live yfinance fetch → deterministic pandas ratio engine → Chroma RAG over annual
reports → question-aware LLM Forecast Agent → LLM Critic Agent → report.
Multi-company screeners ("top dividend payers") are OUT OF SCOPE for MVP — the
Intent Agent classifies them out_of_scope; they're a v1.1 enhancement (ARCHITECTURE.md §7).
Free-tier LLMs only (Groq primary, Gemini fallback). Streamlit UI. Runs locally.

## Binding documents
- **ARCHITECTURE.md** — source of truth. Read it before designing or writing anything.
  Tier boundaries (§2–§3), project structure (§5), and design principles (§6) are binding.
  Enhancements must follow the plug-in points in §7.
- **PROMPTS.md** — the sequenced build plan (3-day schedule).

## Golden rules (never violate)
1. LLMs interpret numbers; they NEVER compute or recall financial figures.
   All ratios come from `src/ratio_engine.py` (pure pandas).
2. The LLM never has final say on ticker resolution — `src/ticker_map.py`
   + yfinance validation decide.
3. Fetch-on-demand: no stored market data beyond the 1-hour disk cache.
4. Every user-facing output includes: reasoning chain, confidence level,
   `data as of <timestamp>`, and the not-investment-advice disclaimer.
   For dividend_qa, the direct answer comes first, then the support.
5. Graceful degradation: bad ticker, empty RAG, out-of-scope query, or a
   rate-limited LLM provider must yield a friendly message, never a traceback.
6. Quota discipline: ≤3 LLM calls per query (2 for bare tickers); cache aggressively.
7. Data sources and LLM providers stay behind their interfaces
   (`CorporateActionsSource`, `llm_router.invoke`) — swaps touch one file only.

## Conventions
- Python 3.11+, type hints everywhere, docstrings state the module's tier.
- Config values (models, TTLs, paths, disclaimer text) live in `src/config.py` only.
- Secrets in `.env` (see `.env.example`); never hardcode keys.
- Tests in `tests/`; LLM calls mocked in unit tests, live calls only in
  integration tests marked as such.

## Commands
- CLI: `python forecast.py ITC`  or
       `python forecast.py "Will Infosys increase its dividend next quarter?"`
- UI: `streamlit run app.py`
- Ingest annual report: `python -m src.pdf_ingest ITC data/annual_reports/itc_fy25.pdf`
- Backtest: `python backtest.py`
