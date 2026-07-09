"""Tier 3 — Agentic Orchestration: Intent Agent (graph node 0).

``parse_query(user_query)`` turns a raw query into a structured dict:
``{intent, ticker, question, horizon, company_mention, message}`` where
``intent`` is one of ``forecast_single | dividend_qa | out_of_scope |
clarify``.

Routing rules (ARCHITECTURE.md §3, node 0):

1. **Bare ticker short-circuit** — a single ticker-shaped token that
   ``ticker_map`` resolves (or that yfinance confirms exists) becomes
   ``forecast_single`` *without any LLM call*, to save quota.
2. **Otherwise one LLM call** (`llm_router.invoke_json`) classifies the
   intent and extracts a company *mention*, the question text, and a
   horizon. Screener / multi-company / non-dividend queries →
   ``out_of_scope``.
3. The LLM only *proposes* the company mention; ``ticker_map.resolve()``
   (then a yfinance existence check) makes the final ticker decision — the
   LLM never has final say. An unresolvable company → ``clarify`` with the
   attempted name echoed back.

Everything degrades to ``clarify`` rather than raising, so a bad query or
an LLM failure still produces a friendly, actionable result.
"""

from __future__ import annotations

import logging
import re

from . import llm_router, ticker_map

logger = logging.getLogger(__name__)

VALID_INTENTS = {"forecast_single", "dividend_qa", "out_of_scope", "clarify"}

# A single ticker-shaped token (no whitespace): letters/digits plus the
# punctuation NSE symbols use (M&M, BAJAJ-AUTO, L&T).
_BARE_TICKER_RE = re.compile(r"^[A-Za-z][A-Za-z0-9&.\-]{0,14}$")
_TICKER_SHAPE_RE = re.compile(r"^[A-Z0-9&.\-]{1,15}$")

_OUT_OF_SCOPE_MSG = (
    "DiviSense AI analyses one NSE-listed company at a time. Screeners and "
    "multi-company rankings aren't supported yet (they're on the roadmap). "
    "Try a single company, e.g. 'ITC' or 'Will Infosys raise its dividend?'."
)
_GENERIC_CLARIFY_MSG = (
    "I couldn't tell which company you mean. Enter an NSE ticker (e.g. 'ITC') "
    "or ask about one company, e.g. 'Will Infosys increase its dividend next year?'."
)

_SCHEMA_HINT = (
    '{"intent": "forecast_single | dividend_qa | out_of_scope | clarify", '
    '"company_mention": "<company name as written, or null>", '
    '"question": "<the user\'s question in their words, or null>", '
    '"horizon": "<time horizon e.g. next quarter | next year | FY27, or null>"}'
)


def _result(
    intent: str,
    ticker: str | None = None,
    question: str | None = None,
    horizon: str | None = None,
    company_mention: str | None = None,
    message: str | None = None,
) -> dict:
    """Build a parse result with every contract key always present."""
    return {
        "intent": intent,
        "ticker": ticker,
        "question": question,
        "horizon": horizon,
        "company_mention": company_mention,
        "message": message,
    }


def _ticker_exists(candidate: str) -> bool:
    """True if yfinance returns usable data for ``candidate`` on NSE.

    Isolated so the network dependency can be mocked in tests and so a
    network error degrades to 'not confirmable' rather than an exception.
    """
    from .data_agent import InvalidTickerError, fetch_company_data

    try:
        fetch_company_data(candidate)
        return True
    except InvalidTickerError:
        return False
    except Exception as exc:  # noqa: BLE001 - network/backend issues → unconfirmable
        logger.warning("intent: yfinance existence check failed for %s (%s)", candidate, exc)
        return False


def _confirm_ticker(text: str | None) -> str | None:
    """Resolve a name/ticker to a canonical NSE ticker, or None.

    ``ticker_map.resolve`` is authoritative; if it misses, a ticker-shaped
    candidate (spaces stripped, upper-cased) is confirmed via yfinance so
    the thousands of NSE symbols not in the alias CSV still work. The LLM
    never decides this — code does.
    """
    if not text or not text.strip():
        return None

    resolved = ticker_map.resolve(text)
    if resolved:
        return resolved

    candidate = re.sub(r"[^A-Za-z0-9&.\-]", "", text).upper()
    if candidate and _TICKER_SHAPE_RE.match(candidate) and _ticker_exists(candidate):
        return candidate
    return None


def _classify_with_llm(user_query: str) -> dict:
    """One LLM call: classify intent + extract mention/question/horizon."""
    prompt = (
        "You are the intent parser for DiviSense AI, a dividend-forecasting tool "
        "for a SINGLE Indian (NSE-listed) company at a time.\n\n"
        "Classify the user's query into exactly one intent:\n"
        '- "forecast_single": wants a dividend forecast for ONE specific company '
        '(e.g. "Forecast ITC\'s dividend for next year", "Reliance dividend outlook").\n'
        '- "dividend_qa": asks a specific question about ONE company\'s dividend '
        '(e.g. "Will Infosys increase its dividend next quarter?", "Is HCL\'s dividend safe?").\n'
        '- "out_of_scope": about MULTIPLE companies, a screener/ranking '
        '("top dividend payers", "best PSU dividend stocks"), or not about dividends '
        "at all (price targets, buy/sell advice).\n"
        '- "clarify": ambiguous, names no company, or nonsensical.\n\n'
        "Also extract:\n"
        "- company_mention: the company name the user refers to (as written), or null if none/many.\n"
        "- question: the user's question in their own words, or null.\n"
        '- horizon: any time horizon mentioned (e.g. "next quarter", "next year", "FY27"), or null.\n\n'
        "Do NOT guess stock tickers. Report only the company NAME as mentioned.\n\n"
        f"User query: {user_query!r}"
    )
    return llm_router.invoke_json(prompt, _SCHEMA_HINT, task_type="reasoning")


def parse_query(user_query: str) -> dict:
    """Parse a raw user query into a structured intent dict (see module docs)."""
    if not user_query or not user_query.strip():
        return _result("clarify", message=_GENERIC_CLARIFY_MSG)

    query = user_query.strip()

    # 1. Bare ticker short-circuit — no LLM call.
    if _BARE_TICKER_RE.match(query):
        ticker = _confirm_ticker(query)
        if ticker:
            logger.info("intent: bare-ticker short-circuit %r -> %s", query, ticker)
            return _result("forecast_single", ticker=ticker, company_mention=query)
        # Not a valid ticker; fall through to the LLM.

    # 2. LLM classification.
    try:
        parsed = _classify_with_llm(query)
    except Exception as exc:  # noqa: BLE001 - LLM/JSON failure must not crash the pipeline
        logger.warning("intent: LLM classification failed (%s); asking to clarify", exc)
        return _result(
            "clarify",
            question=query,
            message="Sorry, I couldn't parse that. Try a ticker like 'ITC' or a "
            "question like 'Will Infosys raise its dividend next year?'.",
        )

    intent = parsed.get("intent")
    mention = parsed.get("company_mention") or None
    question = parsed.get("question") or query
    horizon = parsed.get("horizon") or None

    if intent not in VALID_INTENTS:
        logger.warning("intent: LLM returned unknown intent %r; asking to clarify", intent)
        return _result("clarify", question=question, horizon=horizon,
                       company_mention=mention, message=_GENERIC_CLARIFY_MSG)

    if intent == "out_of_scope":
        return _result("out_of_scope", question=question, horizon=horizon,
                       company_mention=mention, message=_OUT_OF_SCOPE_MSG)

    if intent == "clarify":
        msg = _unresolved_msg(mention) if mention else _GENERIC_CLARIFY_MSG
        return _result("clarify", question=question, horizon=horizon,
                       company_mention=mention, message=msg)

    # 3. forecast_single / dividend_qa — code makes the final ticker call.
    ticker = _confirm_ticker(mention)
    if not ticker:
        logger.info("intent: could not resolve company mention %r -> clarify", mention)
        return _result("clarify", question=question, horizon=horizon,
                       company_mention=mention, message=_unresolved_msg(mention))

    return _result(intent, ticker=ticker, question=question, horizon=horizon,
                   company_mention=mention)


def _unresolved_msg(mention: str | None) -> str:
    if mention:
        return (
            f"I couldn't match '{mention}' to an NSE-listed company. Try its ticker "
            "(e.g. 'ITC', 'COALINDIA', 'INFY') or check the spelling."
        )
    return _GENERIC_CLARIFY_MSG


if __name__ == "__main__":
    import json
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

    if len(sys.argv) < 2:
        print('Usage: python -m src.intent "<your query>"')
        print('Example: python -m src.intent "Will Infosys increase its dividend next quarter?"')
        sys.exit(1)

    result = parse_query(" ".join(sys.argv[1:]))
    print(json.dumps(result, indent=2, ensure_ascii=False))
