"""Central configuration: model names, TTLs, RAG parameters, paths, disclaimer.

All other modules import constants from here — no magic numbers or
hard-coded paths elsewhere in the codebase.
"""

from pathlib import Path

# ── LLM models ────────────────────────────────────────────────────────
GROQ_MODEL = "llama-3.3-70b-versatile"   # primary: fast, short reasoning
# Fallback: large-context / 429s. gemini-2.5-flash was retired by Google
# (404 "no longer available", 2026-07); 3.x flash is 503-overloaded on the
# free tier, so pin the lite tier which reliably answers. If this one ever
# 404s, run `client.models.list()` and pick the current flash-class model.
GEMINI_MODEL = "gemini-3.1-flash-lite"

# ── Caching ───────────────────────────────────────────────────────────
CACHE_TTL_SECONDS = 3600  # 1-hour disk-cache TTL for fetched market data

# ── LLM quota limits (approximate free-tier caps — tune to your plan) ─
GROQ_RPM_LIMIT = 30
GROQ_RPD_LIMIT = 14400
GEMINI_RPM_LIMIT = 15
GEMINI_RPD_LIMIT = 1500

# ── External-call discipline (ARCHITECTURE.md §6) ─────────────────────
LLM_TIMEOUT_SECONDS = 60       # per-request timeout for Groq/Gemini calls
YFINANCE_TIMEOUT_SECONDS = 45  # wall-clock cap on one company's data fetch
MAX_LLM_CALLS_PER_QUERY = 3    # hard budget: intent + forecast + critic

# ── RAG (annual-report ingestion & retrieval) ────────────────────────
RAG_CHUNK_SIZE = 800      # tokens per chunk
RAG_CHUNK_OVERLAP = 100   # token overlap between chunks
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# ── Paths ─────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
TICKER_ALIASES_CSV = DATA_DIR / "ticker_aliases.csv"
CORPORATE_ACTIONS_CSV = DATA_DIR / "corporate_actions.csv"
ANNUAL_REPORTS_DIR = DATA_DIR / "annual_reports"
CHROMA_DIR = DATA_DIR / "chroma"
CACHE_DIR = PROJECT_ROOT / ".diskcache"
LLM_CACHE_DIR = CACHE_DIR / "llm"

# ── Disclaimer (must appear on every user-facing output) ─────────────
DISCLAIMER = (
    "⚠️ DiviSense AI is a research tool, not investment advice. "
    "Forecasts are model-generated estimates based on public data and "
    "may be wrong. Consult a SEBI-registered investment adviser before "
    "making investment decisions."
)
