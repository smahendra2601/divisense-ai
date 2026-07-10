"""Tier 1 — Data Acquisition: deterministic company-name → NSE-ticker resolution.

Three-stage resolution, all deterministic code (the LLM proposes company
*names*; it never has final say on a ticker):

1. **Alias CSV** (``data/ticker_aliases.csv``) — curated nicknames the
   official register can't know ("Infy", "SBI", "L&T").
2. **Exact symbol** in the live NSE symbol master — any of ~2,400 listed
   symbols typed directly (e.g. ``CANBK``).
3. **Fuzzy name search** against the master's official company names —
   ``search()`` returns scored candidates; ``resolve_with_suggestions()``
   auto-accepts a clear high-confidence winner and otherwise surfaces
   "did you mean?" options for the user to pick.

The symbol master is fetched from NSE (``config.NSE_SYMBOLS_URL``),
disk-cached for 7 days, snapshotted to ``data/nse_symbols.csv``, and
degrades gracefully: live fetch → snapshot file → aliases only.
"""

from __future__ import annotations

import csv
import io
import logging
import re
import urllib.request
from difflib import SequenceMatcher
from functools import lru_cache

from . import config
from .cache import disk_cached

logger = logging.getLogger(__name__)

_BROWSER_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
)

# Legal-form tokens dropped when comparing company names.
_SUFFIX_TOKENS = {"LIMITED", "LTD", "LTD.", "CO", "CO.", "COMPANY", "CORPORATION", "CORP"}


def _normalize(text: str) -> str:
    """Lowercase, strip, and collapse internal whitespace for matching."""
    return " ".join(text.strip().lower().split())


def _normalize_company(name: str) -> str:
    """Normalize a company name for fuzzy comparison.

    Uppercases, strips punctuation, and drops trailing legal-form tokens so
    "Canara Bank" and "CANARA BANK LIMITED" compare equal.
    """
    cleaned = re.sub(r"[^A-Za-z0-9& ]+", " ", name).upper()
    tokens = [t for t in cleaned.split() if t]
    while tokens and tokens[-1] in _SUFFIX_TOKENS:
        tokens.pop()
    return " ".join(tokens)


# ── alias CSV (stage 1) ──────────────────────────────────────────────
@lru_cache(maxsize=1)
def _load() -> tuple[dict[str, str], dict[str, str]]:
    """Read the alias CSV once into two normalized lookup maps.

    Returns ``(ticker_by_norm, alias_by_norm)`` where keys are normalized
    strings and values are the canonical NSE ticker as written in the CSV.
    """
    ticker_by_norm: dict[str, str] = {}
    alias_by_norm: dict[str, str] = {}

    with open(config.TICKER_ALIASES_CSV, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            ticker = (row.get("ticker") or "").strip()
            alias = (row.get("alias") or "").strip()
            if not ticker:
                continue
            ticker_by_norm[_normalize(ticker)] = ticker
            if alias:
                alias_by_norm[_normalize(alias)] = ticker

    return ticker_by_norm, alias_by_norm


# ── NSE symbol master (stages 2 & 3) ─────────────────────────────────
@disk_cached(ttl=config.SYMBOL_MASTER_TTL_SECONDS)
def _fetch_symbol_master_live() -> list[tuple[str, str]]:
    """Download the official NSE equity list. Raises on any failure.

    Failures are never cached (diskcache memoize only stores returns), so
    the next call retries. A successful fetch refreshes the on-disk
    snapshot used as the offline fallback.
    """
    request = urllib.request.Request(
        config.NSE_SYMBOLS_URL,
        headers={"User-Agent": _BROWSER_UA, "Referer": "https://www.nseindia.com/"},
    )
    with urllib.request.urlopen(request, timeout=30) as resp:
        text = resp.read().decode("utf-8", errors="replace")

    rows = _parse_master_csv(text)
    if not rows:
        raise ValueError("NSE symbol master parsed to zero rows")

    try:  # refresh the snapshot; best-effort
        with open(config.NSE_SYMBOLS_CSV, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["SYMBOL", "NAME OF COMPANY"])
            writer.writerows(rows)
    except OSError as exc:  # pragma: no cover - disk issues
        logger.warning("ticker_map: could not write symbol snapshot (%s)", exc)

    logger.info("ticker_map: fetched %d symbols from NSE", len(rows))
    return rows


def _parse_master_csv(text: str) -> list[tuple[str, str]]:
    """Parse EQUITY_L.csv-style content to [(symbol, company_name)]."""
    rows: list[tuple[str, str]] = []
    reader = csv.DictReader(io.StringIO(text))
    fields = {(f or "").strip().upper(): f for f in (reader.fieldnames or [])}
    sym_key, name_key = fields.get("SYMBOL"), fields.get("NAME OF COMPANY")
    if not sym_key or not name_key:
        return []
    for row in reader:
        symbol = (row.get(sym_key) or "").strip().upper()
        name = (row.get(name_key) or "").strip()
        if symbol and name:
            rows.append((symbol, name))
    return rows


@lru_cache(maxsize=1)
def _symbol_master() -> list[tuple[str, str]]:
    """The NSE symbol master: live fetch → snapshot file → empty list."""
    try:
        return _fetch_symbol_master_live()
    except Exception as exc:  # noqa: BLE001 - degrade, never crash resolution
        logger.warning("ticker_map: live symbol fetch failed (%s); trying snapshot", exc)
    try:
        with open(config.NSE_SYMBOLS_CSV, newline="", encoding="utf-8") as fh:
            rows = _parse_master_csv(fh.read())
        if rows:
            logger.info("ticker_map: using snapshot with %d symbols", len(rows))
            return rows
    except OSError:
        pass
    logger.warning("ticker_map: no symbol master available; alias-only resolution")
    return []


@lru_cache(maxsize=1)
def _master_symbols_set() -> frozenset[str]:
    return frozenset(sym for sym, _ in _symbol_master())


# ── public API ───────────────────────────────────────────────────────
def resolve(name_or_ticker: str) -> str | None:
    """Resolve a name or ticker to its canonical NSE symbol, or ``None``.

    Case-insensitive and whitespace-tolerant. Order: exact alias-CSV ticker,
    alias, then exact symbol in the NSE master. Fuzzy matching is *not*
    attempted here — use ``resolve_with_suggestions`` for that, so callers
    explicitly opt in to confidence-based behavior.
    """
    if not name_or_ticker or not name_or_ticker.strip():
        return None

    key = _normalize(name_or_ticker)
    ticker_by_norm, alias_by_norm = _load()

    if key in ticker_by_norm:  # exact ticker match wins first
        return ticker_by_norm[key]
    if key in alias_by_norm:
        return alias_by_norm[key]

    candidate = key.upper().replace(" ", "")
    if candidate in _master_symbols_set():
        return candidate
    return None


def master_available() -> bool:
    """True when the NSE symbol master (live or snapshot) is loaded.

    Callers use this to skip slower existence checks (yfinance) that are
    only worthwhile when the authoritative list couldn't be loaded.
    """
    return bool(_master_symbols_set())


def search(query: str, limit: int = config.TICKER_MAX_SUGGESTIONS) -> list[dict]:
    """Fuzzy-search official company names; return scored candidates.

    Each hit is ``{"ticker", "company_name", "score"}`` with score in
    [0, 1], best first, filtered at ``config.TICKER_SUGGEST_MIN_SCORE``.
    Deterministic code — difflib ratio over normalized names, with a
    containment boost so "Canara" surfaces every Canara-group company.
    """
    if not query or not query.strip():
        return []
    norm_query = _normalize_company(query)
    if not norm_query:
        return []

    scored: list[tuple[float, str, str]] = []
    for symbol, name in _symbol_master():
        norm_name = _normalize_company(name)
        if not norm_name:
            continue
        if norm_query == norm_name or norm_query == symbol:
            score = 1.0
        else:
            score = SequenceMatcher(None, norm_query, norm_name).ratio()
            # Containment boost: a query that is a prefix/word of the name
            # ("CANARA" in "CANARA BANK") should beat plain edit distance.
            if norm_name.startswith(norm_query + " ") or f" {norm_query} " in f" {norm_name} ":
                score = max(score, 0.75)
        if score >= config.TICKER_SUGGEST_MIN_SCORE:
            scored.append((score, symbol, name))

    scored.sort(key=lambda t: (-t[0], t[1]))
    return [
        {"ticker": sym, "company_name": name, "score": round(sc, 3)}
        for sc, sym, name in scored[:limit]
    ]


def resolve_with_suggestions(name_or_ticker: str) -> tuple[str | None, list[dict]]:
    """Resolve exactly, else fuzzy-match with confidence rules.

    Returns ``(ticker, suggestions)``:
    - exact resolution (aliases / symbol) → ``(ticker, [])``
    - a fuzzy top hit scoring >= ``TICKER_AUTO_ACCEPT_SCORE`` with a clear
      lead over the runner-up → ``(ticker, [])`` (auto-accepted, logged)
    - otherwise → ``(None, top-suggestions)`` for the user to pick from
    """
    ticker = resolve(name_or_ticker)
    if ticker:
        return ticker, []

    hits = search(name_or_ticker)
    if not hits:
        return None, []

    top = hits[0]
    runner_up = hits[1]["score"] if len(hits) > 1 else 0.0
    if top["score"] >= config.TICKER_AUTO_ACCEPT_SCORE and (
        len(hits) == 1 or top["score"] - runner_up >= 0.05
    ):
        logger.info(
            "ticker_map: auto-accepted %r -> %s (%s, score %.2f)",
            name_or_ticker, top["ticker"], top["company_name"], top["score"],
        )
        return top["ticker"], []

    return None, hits


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

    args = sys.argv[1:]
    if not args:
        print('Usage: python -m src.ticker_map "<company name or ticker>"')
        print('       python -m src.ticker_map --search "<partial name>"')
        sys.exit(1)

    if args[0] == "--search":
        query = " ".join(args[1:])
        hits = search(query)
        print(f"{len(hits)} candidate(s) for {query!r}:")
        for h in hits:
            print(f"  {h['score']:.2f}  {h['ticker']:<12} {h['company_name']}")
        sys.exit(0)

    query = " ".join(args)
    ticker, suggestions = resolve_with_suggestions(query)
    if ticker:
        print(ticker)
    elif suggestions:
        print(f"No exact match for '{query}'. Did you mean:")
        for s in suggestions:
            print(f"  {s['ticker']:<12} {s['company_name']}  (score {s['score']:.2f})")
        sys.exit(1)
    else:
        print(f"No NSE ticker found for '{query}'. Try a different name or spelling.")
        sys.exit(1)
