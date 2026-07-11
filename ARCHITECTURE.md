# DiviSense AI — Architecture Reference

> **Purpose of this file:** Master architecture reference for the DiviSense AI project.
> Claude Code should treat this document as the source of truth for design decisions.
> Do not deviate from the tier boundaries or golden rules below without explicit instruction.

---

## 1. Project Overview

**DiviSense AI** is an agentic dividend forecasting platform for the **Indian stock market (NSE-listed companies)**. It is a **single-company tool**: given a ticker or a natural-language question about one company, it fetches live public data (dividend history, financial statements), computes fundamental ratios deterministically, retrieves qualitative context from annual reports via RAG and recent dividend-relevant news via a web-search source, and uses a multi-agent LLM workflow (LangGraph) to produce a dividend forecast or a direct answer — always with transparent reasoning and a confidence level.

**Supported query types (intents):**
| Intent | Example inputs | Output |
|---|---|---|
| `forecast_single` | "ITC", "Forecast Coal India's next dividend", "What dividend will TCS pay next year?" | Next-FY dividend forecast: amount range (₹/share), expected window, confidence, reasoning, risks |
| `dividend_qa` | "Will Infosys increase its dividend next quarter?", "Is HCL's dividend sustainable?" | Direct answer (likely yes / likely no / unclear + likelihood) backed by the standard forecast and metrics |
| `clarify` | Ambiguous/unparseable input | Friendly request to rephrase, with examples |

**Constraints:**
- Runs entirely on a local laptop (no cloud infra required) — and, unchanged
  architecturally, optionally deploys as-is to Render's free tier behind a custom
  domain (see README's "Deploy to Render"); RAG data ships in the repo instead of a
  managed store, so there's still no infra dependency beyond the two free LLM APIs
- Free-tier LLM APIs only (Groq primary, Google Gemini fallback)
- Fetch-on-demand data model — no pre-ingested database of market data (freshness by design)
- 3-day MVP timeline; architecture must leave enhancement paths open (see §7)

**Non-goals (MVP):** multi-company screeners/rankings ("top dividend payers…"), live NSE/BSE scraping, portfolio management, buy/sell signals, multi-user auth. If a screener-type question arrives, the Intent Agent classifies it as out-of-scope and the Report Node explains this is a single-company tool (screeners are a v1.1 enhancement — §7).

**Disclaimer requirement:** Every user-facing output must carry a "research tool, not investment advice" disclaimer.

---

## 2. High-Level Architecture (4 Tiers)

```
┌────────────────────────────────────────────────────────────────┐
│  TIER 4: PRESENTATION                                          │
│  Streamlit app (app.py) — natural-language question box        │
│  - Direct-answer banner + forecast card + history chart        │
│  - "Data as of <timestamp>" stamp on every result              │
└───────────────────────────┬────────────────────────────────────┘
                            │
┌───────────────────────────▼────────────────────────────────────┐
│  TIER 3: AGENTIC ORCHESTRATION (LangGraph)                     │
│  graph.py — StateGraph with typed DivisenseState               │
│                                                                │
│  [Intent Agent: query → intent, ticker, question, horizon]     │
│         │                                                      │
│         ▼                                                      │
│  [Data Node] → [Ratio Node] → [RAG Node] → [News Node] →       │
│  [Forecast Agent (question-aware)] → [Critic Agent]            │
│         → [Report Node]                                        │
│  (Critic can loop back to Forecast once on failure;            │
│   clarify / out-of-scope / errors route straight to Report)    │
└───────┬───────────────────┬───────────────────┬────────────────┘
        │                   │                   │
┌───────▼───────┐   ┌───────▼───────┐   ┌───────▼────────────────┐
│ TIER 2:       │   │ TIER 2:       │   │ TIER 2:                │
│ INTELLIGENCE  │   │ KNOWLEDGE     │   │ LLM SERVICE            │
│ ratio_engine  │   │ ChromaDB +    │   │ llm_router.py          │
│ (pure pandas, │   │ local embeds  │   │ Groq (fast, primary)   │
│ deterministic)│   │ (annual rpts) │   │ Gemini (large-context  │
│               │   │ + news.py     │   │ fallback)              │
│               │   │ (Tavily web   │   │ + quota tracker        │
│               │   │ search)       │   │ + response cache       │
└───────┬───────┘   └───────┬───────┘   └────────────────────────┘
        │                   │
┌───────▼───────────────────▼────────────────────────────────────┐
│  TIER 1: DATA ACQUISITION                                      │
│  data_agent.py  — live yfinance fetch (.NS tickers):           │
│                   dividends, prices, P&L, balance sheet, CF    │
│  ticker_map.py  — data/ticker_aliases.csv: company name →      │
│                   NSE ticker ("Infosys"→INFY); prevents the    │
│                   LLM from guessing tickers                    │
│  corp_actions.py — CorporateActionsSource interface:           │
│                   CSVSource (MVP) | NSEScraperSource (future)  │
│  cache.py        — 1-hour TTL disk cache (demo protection)     │
│  pdf_ingest.py   — one-time annual report → Chroma pipeline    │
└────────────────────────────────────────────────────────────────┘
```

---

## 3. Tier-by-Tier Detail

### Tier 1 — Data Acquisition
| Module | Responsibility | Key decisions |
|---|---|---|
| `data_agent.py` | Given ticker (e.g. `ITC`), append `.NS`, fetch via `yfinance`: dividend history, current price, annual income statement, balance sheet, cash flow. Returns a normalized dict with `data_timestamp`. | Live fetch per query = always fresh. Raise clear `InvalidTickerError` for bad tickers. |
| `ticker_map.py` | Loads `data/ticker_aliases.csv` (columns: alias, ticker, company_name) covering ~100 common company names/variants ("Infosys", "Infy", "Coal India", "SBI"…). `resolve(name_or_ticker) -> ticker | None`. | Deterministic name→ticker mapping. The Intent Agent proposes; `ticker_map` (then a yfinance validity check) confirms. **The LLM never has final say on a ticker.** |
| `cache.py` | Disk cache (`diskcache`) with **1-hour TTL** keyed on ticker. | Protects demo from rate limits & repeated fetches; TTL short enough that data is never meaningfully stale. |
| `corp_actions.py` | Abstract `CorporateActionsSource` with `get_actions(ticker)`. MVP implementation reads `data/corporate_actions.csv`. | Interface exists so an NSE scraper can slot in later **without touching any other module**. |
| `pdf_ingest.py` | One-time script: parse annual report PDFs (`pdfplumber`), chunk (~800 tokens, 100 overlap), embed, store in Chroma. | Annual reports are static documents — pre-embedding them does NOT violate the freshness principle. |

### Tier 2 — Intelligence, Knowledge, LLM Service
| Module | Responsibility | Key decisions |
|---|---|---|
| `ratio_engine.py` | Pure pandas computation: payout ratio (5yr), dividend CAGR, FCF/dividend coverage, consecutive-increase streak, current yield, debt/equity trend, dividend consistency score, **recent dividend trajectory (last 4 payouts: rising/flat/falling)** — the trajectory feature directly powers "will it increase?" answers. Returns structured `metrics` dict. | **NO LLM involvement.** All financial numbers are computed in code. |
| `rag.py` | Query Chroma for dividend-policy / capital-allocation snippets for the ticker. Local embedding model (`all-MiniLM-L6-v2`) via Chroma's built-in ONNX runtime — zero API cost, and deliberately not `sentence-transformers`/`torch` (that combination measured ~330MB RSS, enough to OOM Render's 512MB free-tier instance). | If no documents exist for a ticker, return empty context gracefully (pipeline must still work). |
| `news.py` | Fetch recent dividend-relevant news snippets via the Tavily web-search API (`fetch_recent_news(ticker, company_name)`) — special-dividend rumors, board announcements, regulatory/tax risk that a once-a-year annual report can't see. Deterministic retrieval (fixed query `"{company} dividend announcement"`, `topic=general`, `time_range=year`); **zero LLM calls**; stdlib `urllib` (no new dep); 1-hour disk cache. | **Optional context source, fail-soft like `rag.py`:** no `TAVILY_API_KEY`, a network error, or a timeout all degrade to `[]`, never raising. Snippets are **qualitative context only — never a source of numbers** (see Tier 3 node 3b). |
| `llm_router.py` | Single `invoke(prompt, task_type)` entry point + `invoke_json(prompt, schema_hint)`. Routes: short reasoning → Groq (`openai/gpt-oss-120b`, an open-weight **reasoning** model — chain-of-thought kept out of the response via `reasoning_format="hidden"`, depth capped via `reasoning_effort="medium"`); long-context → Gemini Flash. Tracks per-provider RPM/RPD counters; auto-fallback on 429; caches identical prompts. | Free tiers: Groq gpt-oss-120b ≈ 200K tokens/day (vs 100K for llama-3.3-70b, the prior default — directly addresses token-cap exhaustion seen in large backtests) / low TPM; Gemini Flash ≈ 1,500 req/day. Both exhaust fast — cache aggressively. |

### Tier 3 — Agentic Orchestration (LangGraph)
State object (`DivisenseState`, TypedDict): `user_query`, `intent`, `ticker`, `question`, `horizon`, `raw_data`, `metrics`, `rag_context`, `news_context`, `forecast`, `critique`, `retry_count`, `final_report`, `errors`, `data_timestamp`.

Nodes:
0. **Intent Agent** — parses the raw user query into structured JSON:
   `{intent: "forecast_single" | "dividend_qa" | "clarify" | "out_of_scope", company_mention, question, horizon}`.
   - A bare ticker input (regex + yfinance validity) **skips the LLM entirely** → `forecast_single`. Saves quota.
   - Otherwise one small LLM call classifies intent and extracts the company mention; `ticker_map.resolve()` converts it to a ticker deterministically. Unresolvable company → `clarify`.
   - Multi-company/screener questions → `out_of_scope` (Report Node explains + points to roadmap).
1. **Data Node** — calls `data_agent` (+cache). On failure → error path to Report Node.
2. **Ratio Node** — calls `ratio_engine`.
3. **RAG Node** — calls `rag.retrieve(ticker)`.
3b. **News Node** — calls `news.fetch_recent_news(ticker, company_name)`. Deterministic retrieval, **no LLM call**; fails soft to `[]` (missing key / network error / timeout). The snippets are passed to the Forecast Agent as **qualitative context only — never a source of numbers**; the prompt says so explicitly and the Critic's number-tracing rule (node 5) enforces it. **Disabled during backtests** (`backtest.py`) — a live search would surface the very dividend being withheld.
4. **Forecast Agent (LLM, question-aware)** — receives `metrics` as JSON + RAG snippets + recent-news snippets + (for `dividend_qa`) the user's exact question. Output JSON:
   `{direct_answer?, likelihood?: "likely"|"unlikely"|"unclear", amount_range_inr, expected_window, confidence: high|medium|low, reasoning: [...], risks: [...]}`.
   - For `forecast_single`: forecast the **next fiscal year's total dividend per share** (range), plus likely interim/final split and timing window.
   - For `dividend_qa`: answer the question **first and directly** (e.g. "Likely yes — increase probability: medium"), then support it with the forecast.
   - **Indian cadence rule:** most NSE companies pay interim + final dividends, not US-style quarterly. If the question says "next quarter", the agent must map it to the company's actual payout cadence and say so explicitly.
   - The model must ONLY use numbers present in the provided metrics.
5. **Critic Agent (LLM)** — verifies every number cited exists in `metrics`/`raw_data` AND that the direct answer is consistent with the metrics (e.g. it should not say "likely increase" when payout ratio is stretched and the trajectory is falling, without acknowledging the tension). Output: `{approved, issues}`. If rejected and `retry_count == 0` → loop back once with critique attached; then flag low confidence.
6. **Report Node** — assemble final markdown (direct-answer banner when present + forecast card + metrics + corporate actions from CSVSource) + timestamp + disclaimer. Also handles `clarify`, `out_of_scope`, and error messages.

**Golden rules (enforce everywhere):**
- LLMs **interpret** numbers; they never compute or recall financial figures.
- The LLM never has final say on ticker resolution — `ticker_map` + yfinance validation do.
- Every forecast/answer shows its reasoning chain — never a bare number or bare yes/no.
- Every output carries `data as of <timestamp>` and the disclaimer.
- Per query, LLM calls are at most 3 (intent + forecast + critic), 2 for bare-ticker input.

### Tier 4 — Presentation (Streamlit)
- Single page with a **natural-language question box** ("Enter a ticker or ask a question…") + clickable example chips:
  - "Will Infosys increase its dividend next quarter?"
  - "Forecast ITC's dividend for next year"
  - "COALINDIA"
- Renders: direct-answer banner (for `dividend_qa`), forecast card (amount range, expected window, color-coded confidence badge, reasoning bullets, risks), key metrics table, Plotly dividend-history bar chart (last 10 years), "data as of <timestamp>" caption.
- Expandable "🔍 Agent trace" (intent parse + each node's output) — this sells the *agentic* story.
- Friendly handling of `clarify`, `out_of_scope`, invalid tickers, and LLM failures.
- Also expose a CLI: `python forecast.py "Will Infosys increase its dividend next quarter?"` or `python forecast.py ITC`.

---

## 4. End-to-End Workflows (sequences)

**A. Direct question**
```
"Will Infosys increase its dividend next quarter?"
  → Intent Agent    → {intent: dividend_qa, company: "Infosys"} → ticker_map → INFY
  → Data Node       (yfinance live fetch, cached 1h)
  → Ratio Node      (pandas metrics incl. dividend trajectory)
  → RAG Node        (Chroma snippets, may be empty)
  → Forecast Agent  (direct answer + likelihood + forecast; notes
                     interim/final cadence vs "quarterly" framing)
  → Critic Agent    (validates numbers & answer consistency; ≤1 retry)
  → Report Node     (answer banner + forecast + timestamp + disclaimer)
```
Latency: ~8–18 s (3 LLM calls, one fetch).

**B. Next-year forecast**
```
"ITC"  (bare ticker — regex shortcut, no intent LLM call)
  → Data → Ratio → RAG → Forecast Agent (next-FY DPS range, interim/
    final split, timing window) → Critic → Report
```
Latency: ~5–15 s (2 LLM calls).

---

## 5. Project Structure

```
divisense-ai/
├── ARCHITECTURE.md          # this file — source of truth
├── PROMPTS.md               # Claude Code build prompts
├── CLAUDE.md                # Claude Code project context
├── README.md
├── requirements.txt
├── .env.example             # GROQ_API_KEY=, GOOGLE_API_KEY=, TAVILY_API_KEY= (optional)
├── app.py                   # Streamlit UI (Tier 4)
├── forecast.py              # CLI entry point (free-text or ticker)
├── src/
│   ├── config.py            # models, TTLs, limits, paths, disclaimer
│   ├── data_agent.py        # Tier 1
│   ├── ticker_map.py        # Tier 1 (name → ticker resolution)
│   ├── corp_actions.py      # Tier 1 (interface + CSV impl)
│   ├── cache.py             # Tier 1
│   ├── pdf_ingest.py        # Tier 1 (one-time script)
│   ├── ratio_engine.py      # Tier 2
│   ├── rag.py               # Tier 2
│   ├── news.py              # Tier 2 (recent news via Tavily; fail-soft)
│   ├── llm_router.py        # Tier 2
│   ├── intent.py            # Tier 3 (Intent Agent prompt + parsing)
│   ├── graph.py             # Tier 3 (LangGraph)
│   └── report.py            # Tier 3 output formatting
├── data/
│   ├── ticker_aliases.csv   # ~100 company-name → ticker mappings
│   ├── corporate_actions.csv
│   ├── annual_reports/      # PDFs (gitignored)
│   └── chroma/              # vector store (gitignored)
└── tests/
    ├── test_ratio_engine.py # validate vs known ITC/COALINDIA numbers
    ├── test_intent.py       # intent parsing across query phrasings
    ├── test_news.py         # news fail-soft contract (network mocked)
    └── test_pipeline.py     # end-to-end smoke test (LLM mocked)
```

**Dependencies (`requirements.txt`):** `langgraph`, `langchain`, `langchain-groq`, `langchain-google-genai`, `yfinance`, `pandas`, `chromadb`, `pdfplumber`, `streamlit`, `diskcache`, `python-dotenv`, `plotly`. (`onnxruntime`/`tokenizers` come transitively via `chromadb` — no `sentence-transformers`/`torch`, see the memory note below.)

---

## 6. Design Principles (do not violate)

1. **Deterministic where money is involved; LLM where judgment is involved.**
2. **Fetch-on-demand freshness** — no stored market data beyond the cache TTL.
3. **Graceful degradation** — missing RAG docs, LLM 429s, or bad tickers must produce a useful message, never a stack trace.
4. **Swap-friendly interfaces** — data sources and LLM providers sit behind interfaces so future replacements touch one file.
5. **Transparency** — reasoning chain, timestamps, confidence, disclaimer on every output.
6. **Quota discipline** — ≤3 LLM calls per query; cache everything; bare tickers bypass the intent LLM.

**Agentic communication.** Agents (LangGraph nodes) communicate through the shared typed `DivisenseState` passed node-to-node — *not* via MCP. External services (yfinance, Chroma, Tavily, Groq/Gemini) are reached through direct client libraries / stdlib HTTP behind their own interfaces. This is a single-process local app, so MCP tool-servers would add infrastructure the "local laptop, no infra" constraint rules out. **MCP is a clean future swap** (Tavily and several data providers ship MCP servers) if DiviSense ever grows into a multi-client or hosted service — it would slot in behind the existing source interfaces without touching the agents.

**Deterministic retrieval, agentic interpretation.** New context sources like `news.py` follow principle #1: the *retrieval* is deterministic plumbing (fixed query, no LLM), while the *judgment* — deciding what a special-dividend rumor means for the forecast — is the Forecast Agent's job, checked by the Critic. Retrieval never spends an LLM call and never influences ticker resolution or the computed numbers.

---

## 7. Enhancement Roadmap (post-MVP — architecture already accommodates these)

| Phase | Enhancement | Where it plugs in |
|---|---|---|
| v1.1 | **Screener/ranking queries** ("Top PSU dividend payers") — universe metadata CSV + batch fetch + deterministic rank engine + LLM summary as a second LangGraph path | New `universe.py`, `rank_engine.py`, screener branch in `graph.py`; Intent Agent already classifies these (currently `out_of_scope`) |
| v1.1 | **NSE/BSE live corporate-actions scraper** | New `NSEScraperSource` class implementing `CorporateActionsSource` — zero changes elsewhere |
| v1.1 | **Screener.in fundamentals enrichment** | Additional fetcher inside `data_agent.py` merged into `raw_data` |
| v1.1 | **Multi-turn conversation memory** ("what about TCS?") | LangGraph checkpointer + Streamlit session state |
| v1.2 | **Scheduled watchlist monitoring + alerts** (APScheduler → email/Telegram) | New `scheduler.py` calling the existing pipeline |
| v1.2 | **Backtesting module** — hide latest dividend, measure forecast accuracy | *Already shipped* as `backtest.py`, reusing `graph.py` wholesale |
| v1.3 | **Peer/sector comparison agent** | New node in `graph.py` |
| v1.3 | **News & announcement sentiment agent** — _recent-news **context** already delivered in MVP (`news.py` + News Node, Tavily web search, deterministic/fail-soft). Remaining future work: LLM **sentiment scoring** of announcements and folding structured news facts into `corp_actions.py`._ | Extend the existing News Node; new sentiment step (weigh against the ≤3-call budget) |
| v1.3 | **Cleaner news context** — the `year` window can surface low-signal hits (social posts, generic pages). Two levers: a deterministic **relevance-score floor** (zero cost) and/or an **LLM news-summarizer node** that distills raw snippets into one "recent developments" note before the Forecast Agent (+1 LLM call → raises the budget ceiling to 4). | Tune filtering inside `news.py`; optional new summarizer node in `graph.py` (weigh against the ≤3-call budget) |
| v2.0 | **FastAPI backend + React frontend**; multi-user | Tier 4 swap; Tiers 1–3 untouched |
| v2.0 | **PostgreSQL + persistent forecast history** | Replace ad-hoc state persistence in Report Node |
| v2.0 | **Dockerize + deploy** (Railway) | Infra only. *A direct Streamlit→Render deploy (free tier, `render.yaml`, custom domain via GoDaddy DNS, RAG data committed to the repo, shared-password gate) already shipped ahead of this — see README's "Deploy to Render". This v2.0 item is Docker-izing the future FastAPI/React rewrite specifically, not a duplicate.* |
| v2.1 | **Local LLM option (Ollama)** for zero-quota operation | New provider in `llm_router.py` |
| v2.1 | **Portfolio mode** — holdings CSV → portfolio dividend income forecast | New Streamlit page looping the pipeline |

---

*Last updated: 2026-07-12. Treat §2–§6 as binding; §7 as the open runway.*
