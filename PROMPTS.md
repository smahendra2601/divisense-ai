# DiviSense AI — Claude Code Build Prompts

> **How to use:** Open this project folder in VS Code, start Claude Code, and feed these prompts
> **in order**. Each prompt is self-contained but assumes the previous ones completed.
>
> **Context line to prefix every session:**
> "Read ARCHITECTURE.md and CLAUDE.md in the project root before doing anything. Follow the tier boundaries and golden rules exactly."

---

## DAY 1 — Foundation (data + deterministic core)

### Prompt 1.1 — Project scaffold
```
Read ARCHITECTURE.md. Create the exact project structure from §5:
all folders, empty module files with docstrings describing their tier and
responsibility, requirements.txt with the listed dependencies,
.env.example with GROQ_API_KEY and GOOGLE_API_KEY placeholders,
a .gitignore (include .env, data/annual_reports/, data/chroma/,
__pycache__, .diskcache), and src/config.py containing: model names
(groq: llama-3.3-70b-versatile, gemini: gemini-2.5-flash), cache TTL of
3600 seconds, RAG chunk size 800 / overlap 100, the standard disclaimer
text, and all data paths.
Do not implement logic yet. Then create a venv setup command list in
README.md.
```

### Prompt 1.2 — Data agent + ticker map (Tier 1)
```
Implement src/data_agent.py and src/ticker_map.py per ARCHITECTURE.md
Tier 1.
data_agent.py: fetch_company_data(ticker: str) -> dict that appends
".NS", fetches via yfinance (full dividend history with dates/amounts,
current price, last 4 annual income statements, balance sheets, cash
flow statements), normalizes into a JSON-serializable dict with a
"data_timestamp" field (ISO, IST), and raises InvalidTickerError with a
helpful message for bad tickers. Integrate src/cache.py: a diskcache
decorator with 1-hour TTL. Write defensively — yfinance fields are
sometimes missing; return None for missing fields, never crash.
ticker_map.py: create data/ticker_aliases.csv with ~100 rows mapping
common Indian company names and variants to NSE tickers (Infosys/Infy
→INFY, Coal India→COALINDIA, SBI/State Bank→SBIN, TCS/Tata Consultancy
→TCS, ITC→ITC, HCL/HCL Tech→HCLTECH, ONGC→ONGC, L&T/Larsen→LT, etc. —
cover Nifty 50 + major PSU dividend payers with 2-3 aliases each).
resolve(name_or_ticker: str) -> str | None with case-insensitive,
whitespace-tolerant matching; exact ticker match wins first.
Add __main__ blocks so I can verify: python -m src.data_agent (prints
ITC data) and python -m src.ticker_map "coal india" (prints COALINDIA).
```

### Prompt 1.3 — Ratio engine (Tier 2, deterministic)
```
Implement src/ratio_engine.py per ARCHITECTURE.md. Pure pandas, NO LLM.
compute_metrics(raw_data: dict) -> dict returning:
- payout_ratio_5yr (list per year + average), dividend_cagr_5yr,
  fcf_dividend_coverage (FCF / total dividends paid, latest year),
  consecutive_increase_streak, current_yield_pct,
  debt_to_equity_trend (last 3 years), dividend_consistency_score
  (0-100, document the formula in the docstring),
  total_dividends_last_fy, years_of_history,
  recent_trajectory: classify the last 4 payouts as
  rising | flat | falling | mixed — this directly powers
  "will it increase?" answers
- every metric must handle missing inputs by returning None with a
  "warnings" list explaining what couldn't be computed
Then write tests/test_ratio_engine.py with sanity assertions using live
ITC and COALINDIA data (marked as integration tests). Show me computed
metrics for ITC, TCS, COALINDIA, HCLTECH, INFY in a table so I can
manually verify against Screener.in.
```

### Prompt 1.4 — Corporate actions interface (Tier 1)
```
Implement src/corp_actions.py per ARCHITECTURE.md: an abstract
CorporateActionsSource with get_actions(ticker) -> list[dict], a
CSVSource implementation reading data/corporate_actions.csv
(columns: ticker, action_type, amount, announcement_date, ex_date,
record_date, source_note), and a get_default_source() factory.
Create the CSV with 5 realistic sample rows (placeholder data clearly
marked as samples). Add a stub class NSEScraperSource that raises
NotImplementedError with a message pointing to ARCHITECTURE.md §7.
```

---

## DAY 2 — Agentic layer (LLMs + RAG + LangGraph)

### Prompt 2.1 — LLM router (Tier 2)
```
Implement src/llm_router.py per ARCHITECTURE.md. Requirements:
- invoke(prompt: str, task_type: Literal["reasoning","long_context"])
  -> str; reasoning → Groq via langchain-groq; long_context → Gemini
  Flash via langchain-google-genai; keys from .env via python-dotenv
- in-memory + disk cache keyed on prompt hash (skip LLM entirely on hit)
- quota tracker: count requests per provider per minute and per day;
  on 429 or predicted limit, automatically fall back to the other
  provider; log every routing decision
- invoke_json(prompt, schema_hint) that instructs the model to return
  only JSON, strips markdown fences, parses with one repair-retry
Add a __main__ smoke test that sends one tiny prompt to each provider.
```

### Prompt 2.2 — RAG pipeline (Tier 1 ingest + Tier 2 retrieval)
```
Implement src/pdf_ingest.py and src/rag.py per ARCHITECTURE.md.
pdf_ingest.py: CLI `python -m src.pdf_ingest <TICKER> <pdf_path>` that
extracts text with pdfplumber, chunks at 800 tokens / 100 overlap,
embeds with sentence-transformers all-MiniLM-L6-v2 (local, no API),
stores in a persistent Chroma collection "annual_reports" with metadata
{ticker, source_file, page}.
rag.py: retrieve(ticker, query="dividend policy capital allocation
payout", k=4) -> list of snippet dicts. MUST return [] gracefully when
the ticker has no documents — the pipeline runs without RAG context.
```

### Prompt 2.3 — Intent Agent (Tier 3 entry)
```
Implement src/intent.py per ARCHITECTURE.md §3 node 0.
parse_query(user_query) -> {intent, ticker, question, horizon}.
Rules:
- bare ticker-like input (regex; validate via ticker_map exact match or
  a yfinance existence check) short-circuits to forecast_single WITHOUT
  any LLM call
- otherwise one llm_router.invoke_json call that classifies intent as
  forecast_single | dividend_qa | out_of_scope (multi-company/screener/
  non-dividend questions) | clarify, and extracts the company mention,
  the question text, and horizon (e.g. "next quarter", "next year",
  "FY27")
- the LLM only proposes the company mention; ticker_map.resolve() makes
  the final ticker decision. Unresolvable company → clarify with the
  attempted name echoed back.
Write tests/test_intent.py covering: "Will Infosys increase its dividend
next quarter?", "Forecast ITC's dividend for next year", "COALINDIA",
"What dividend will TCS pay in FY27?", "Top public sector dividend
paying companies" (must be out_of_scope), and a nonsense query
(must be clarify). Mock the LLM in unit tests.
```

### Prompt 2.4 — LangGraph pipeline (Tier 3)
```
Implement src/graph.py per ARCHITECTURE.md §3, exactly:
- DivisenseState TypedDict with the listed fields
- entry intent_node (uses src/intent.py); conditional routing:
  forecast_single/dividend_qa → data_node; clarify/out_of_scope/error
  → report_node
- data_node → ratio_node → rag_node → forecast_node → critic_node;
  critic rejected AND retry_count==0 → back to forecast_node with the
  critique injected; else → report_node
- forecast_node prompt: metrics JSON + RAG snippets + (for dividend_qa)
  the user's exact question. Demand JSON {direct_answer?, likelihood?,
  amount_range_inr, expected_window, confidence, reasoning, risks}.
  For forecast_single: forecast next fiscal year's total dividend per
  share as a range, with likely interim/final split and timing window.
  For dividend_qa: answer the question FIRST and directly, then support
  it. Include the Indian cadence rule: if the question says "next
  quarter", map it to the company's actual interim/final payout pattern
  and state this explicitly. The model must ONLY use numbers present in
  the provided metrics.
- critic_node prompt: given metrics JSON + forecast JSON, verify every
  numeric claim traces to the metrics AND the direct answer is
  consistent with them (flag e.g. "likely increase" when trajectory is
  falling and payout ratio is stretched, unless the reasoning addresses
  it). Return {approved, issues}.
- report_node + src/report.py: markdown output — direct-answer banner
  (when present), forecast card, metrics table, corporate actions from
  CSVSource, data timestamp, agent trace, disclaimer from config. Also
  renders clarify / out_of_scope / error messages kindly.
Then implement forecast.py CLI accepting free text or a ticker:
  python forecast.py "Will Infosys increase its dividend next quarter?"
  python forecast.py ITC
Errors at any node route to report_node with a human-readable message.
```

### Prompt 2.5 — End-to-end verification
```
Run the full pipeline via forecast.py for:
1) "ITC"
2) "Will Infosys increase its dividend next quarter?"
3) "Forecast Coal India's dividend for next year"
4) "Top public sector dividend paying companies"  (expect a polite
   out-of-scope message)
5) "ZZZZZ"  (expect a friendly invalid-ticker message)
For each: show the intent parse, the agent trace, the critic verdict,
and confirm no number in the output is absent from computed metrics.
Confirm bare-ticker queries used only 2 LLM calls and question queries
used 3. Fix any failures. Then write tests/test_pipeline.py as a smoke
test with the LLM router mocked.
```

---

## DAY 3 — UI, resilience, submission polish

### Prompt 3.1 — Streamlit app (Tier 4)
```
Implement app.py per ARCHITECTURE.md Tier 4. Single page:
- header with DiviSense AI name + one-line description + disclaimer
  banner
- natural-language question box ("Enter a ticker or ask a question…")
  + clickable example chips: "Will Infosys increase its dividend next
  quarter?", "Forecast ITC's dividend for next year", "COALINDIA"
- spinner with per-node progress text while the graph runs
- results: direct-answer banner (for dividend_qa), forecast card
  (amount range, expected window, color-coded confidence badge,
  reasoning bullets, risks), key metrics table, Plotly dividend-history
  bar chart (last 10 years), "data as of <timestamp>" caption
- expandable "🔍 Agent trace" showing intent parse + each node's output
- friendly handling of clarify, out_of_scope, InvalidTickerError, and
  LLM failures
Keep it one file, clean, minimal custom CSS.
```

### Prompt 3.2 — Resilience hardening
```
Review the whole codebase against ARCHITECTURE.md §6. Verify and fix:
1) every external call (yfinance, Groq, Gemini, Chroma) is wrapped with
   timeout + clear error handling, 2) the cache actually prevents
   duplicate LLM calls for the same query within the TTL, 3) the app
   degrades gracefully when RAG has no docs, when one LLM provider is
   down, and when yfinance returns partial financials, 4) per-query LLM
   call count never exceeds 3. Simulate each failure and show me the
   resulting user-facing behavior.
```

### Prompt 3.3 — Mini backtest (credibility for submission)
```
Create backtest.py: for tickers [ITC, TCS, COALINDIA], remove the most
recent fiscal year's dividends from raw_data before the pipeline runs,
run the forecast, then compare the predicted amount_range vs the actual
withheld total. Print a results table (ticker, predicted range, actual,
hit/miss). Reuse graph.py — do not duplicate pipeline logic.
```

### Prompt 3.4 — README + submission package
```
Write a complete README.md: project description, what questions it can
answer (with the two example queries), architecture summary (reference
ARCHITECTURE.md), setup steps (venv, pip install, .env keys from
console.groq.com and aistudio.google.com), how to ingest an annual
report PDF, how to run CLI / Streamlit / backtest, screenshots
placeholder section, the enhancement roadmap summarized from
ARCHITECTURE.md §7 (screeners listed as v1.1), and the disclaimer.
Then give me a final pre-submission checklist.
```

---

## Utility prompts (use anytime)

```
Re-read ARCHITECTURE.md. Audit the current code for violations of the
golden rules in §3/§6 (LLM computing numbers, LLM deciding tickers,
missing timestamps, missing disclaimer, >3 LLM calls per query, tier
boundary leaks). List violations and fix them.
```

```
I'm hitting Groq 429s during testing. Tighten llm_router: lower the
local RPM budget, prefer cache, and add exponential backoff before
falling back to Gemini.
```

```
Add <ENHANCEMENT NAME> from ARCHITECTURE.md §7. Follow the "where it
plugs in" note exactly and do not modify unrelated tiers.
```
