"""Central configuration: model names, TTLs, RAG parameters, paths, disclaimer.

All other modules import constants from here — no magic numbers or
hard-coded paths elsewhere in the codebase.
"""

from pathlib import Path

# ── LLM models ────────────────────────────────────────────────────────
GROQ_MODEL = "llama-3.3-70b-versatile"   # primary: fast, short reasoning
GEMINI_MODEL = "gemini-2.5-flash"        # fallback: large-context / 429s

# ── Caching ───────────────────────────────────────────────────────────
CACHE_TTL_SECONDS = 3600  # 1-hour disk-cache TTL for fetched market data

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

# ── Disclaimer (must appear on every user-facing output) ─────────────
DISCLAIMER = (
    "⚠️ DiviSense AI is a research tool, not investment advice. "
    "Forecasts are model-generated estimates based on public data and "
    "may be wrong. Consult a SEBI-registered investment adviser before "
    "making investment decisions."
)
