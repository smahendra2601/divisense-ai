"""Tier 2 — LLM Service: provider routing, quota tracking, response cache.

Single ``invoke(prompt, task_type)`` entry point plus
``invoke_json(prompt, schema_hint)``. Routes short reasoning to Groq
(openai/gpt-oss-120b, an open-weight reasoning model) and long-context
work to Gemini Flash.
Tracks per-provider RPM/RPD counters, auto-falls back to the other
provider on a 429 or a *predicted* quota breach, and caches identical
prompts in memory + on disk — free tiers exhaust fast, so cache
aggressively.

Every routing decision (cache hit, provider chosen, quota-based skip,
fallback, failure) is logged via the stdlib ``logging`` module at
``INFO``/``WARNING`` so the pipeline's behavior is auditable without a
debugger.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
from typing import Literal

from diskcache import Cache
from dotenv import load_dotenv

from . import config

logger = logging.getLogger(__name__)

# Populate os.environ from .env without clobbering vars already set
# (e.g. by CI secrets or the shell).
load_dotenv(dotenv_path=config.PROJECT_ROOT / ".env", override=False)

TaskType = Literal["reasoning", "long_context"]

# task_type -> (primary provider, fallback provider)
_TASK_ROUTES: dict[str, tuple[str, str]] = {
    "reasoning": ("groq", "gemini"),
    "long_context": ("gemini", "groq"),
}

_QUOTA_LIMITS: dict[str, dict[str, int]] = {
    "groq": {"rpm": config.GROQ_RPM_LIMIT, "rpd": config.GROQ_RPD_LIMIT},
    "gemini": {"rpm": config.GEMINI_RPM_LIMIT, "rpd": config.GEMINI_RPD_LIMIT},
}


class LLMConfigError(RuntimeError):
    """A provider is missing required configuration (e.g. no API key)."""


class LLMUnavailableError(RuntimeError):
    """Every provider for this call either failed or was quota-capped."""


class LLMJSONParseError(ValueError):
    """invoke_json() could not extract valid JSON, even after one repair retry."""


# ── quota tracking ────────────────────────────────────────────────────
class QuotaTracker:
    """Sliding-window request counter, per provider, for RPM + RPD limits.

    Keeps a timestamp per request and prunes anything older than a day on
    every read/write, so memory stays bounded. ``would_exceed`` lets
    ``invoke()`` skip a provider *before* spending a call on it — the
    "predicted limit" fallback path, as opposed to reacting to a live 429.
    """

    def __init__(self, limits: dict[str, dict[str, int]]):
        self._limits = limits
        self._events: dict[str, list[float]] = {p: [] for p in limits}

    def _prune(self, provider: str, now: float) -> None:
        cutoff = now - 86400
        self._events[provider] = [t for t in self._events.get(provider, []) if t > cutoff]

    def record(self, provider: str, now: float | None = None) -> None:
        now = time.time() if now is None else now
        self._events.setdefault(provider, [])
        self._prune(provider, now)
        self._events[provider].append(now)

    def counts(self, provider: str, now: float | None = None) -> tuple[int, int]:
        """Return (requests in the last 60s, requests in the last 24h)."""
        now = time.time() if now is None else now
        self._prune(provider, now)
        events = self._events.get(provider, [])
        minute_count = sum(1 for t in events if t > now - 60)
        return minute_count, len(events)

    def would_exceed(self, provider: str, now: float | None = None) -> bool:
        limits = self._limits.get(provider, {})
        minute_count, day_count = self.counts(provider, now)
        rpm_limit = limits.get("rpm")
        rpd_limit = limits.get("rpd")
        if rpm_limit is not None and minute_count >= rpm_limit:
            return True
        if rpd_limit is not None and day_count >= rpd_limit:
            return True
        return False


_quota = QuotaTracker(_QUOTA_LIMITS)


# ── response cache (in-memory + disk, keyed on a prompt hash) ─────────
_memory_cache: dict[str, str] = {}
_disk_cache = Cache(str(config.LLM_CACHE_DIR))


def _cache_key(prompt: str, task_type: str) -> str:
    return hashlib.sha256(f"{task_type}:{prompt}".encode("utf-8")).hexdigest()


def _cache_get(key: str) -> str | None:
    if key in _memory_cache:
        return _memory_cache[key]
    value = _disk_cache.get(key)
    if value is not None:
        _memory_cache[key] = value  # promote to the faster tier
    return value


def _cache_set(key: str, value: str) -> None:
    _memory_cache[key] = value
    _disk_cache.set(key, value, expire=config.CACHE_TTL_SECONDS)


def clear_cache() -> None:
    """Drop every cached LLM response, in memory and on disk."""
    _memory_cache.clear()
    _disk_cache.clear()


# ── provider clients (built lazily so import never requires API keys) ─
def _is_reasoning_model(model: str) -> bool:
    """True for Groq models that emit a chain-of-thought (gpt-oss, R1, Qwen3, QwQ).

    Reasoning models accept the ``reasoning_format`` / ``reasoning_effort``
    params; general models (llama-*) reject them, so only forward those
    knobs when the model actually supports them.
    """
    m = model.lower()
    return any(tag in m for tag in ("gpt-oss", "deepseek-r1", "qwen3", "qwq"))


def _build_groq_client():
    from langchain_groq import ChatGroq

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise LLMConfigError(
            "GROQ_API_KEY is not set. Copy .env.example to .env and fill it in."
        )
    extra: dict = {}
    if _is_reasoning_model(config.GROQ_MODEL):
        # Keep the chain-of-thought out of the response so invoke_json() sees
        # clean JSON, and cap reasoning depth for the free-tier TPM budget.
        # (langchain-groq exposes these as explicit ChatGroq params.)
        extra["reasoning_format"] = config.GROQ_REASONING_FORMAT
        extra["reasoning_effort"] = config.GROQ_REASONING_EFFORT
    return ChatGroq(
        model=config.GROQ_MODEL,
        api_key=api_key,
        temperature=0.2,
        timeout=config.LLM_TIMEOUT_SECONDS,
        **extra,
    )


def _build_gemini_client():
    from langchain_google_genai import ChatGoogleGenerativeAI

    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise LLMConfigError(
            "GOOGLE_API_KEY is not set. Copy .env.example to .env and fill it in."
        )
    return ChatGoogleGenerativeAI(
        model=config.GEMINI_MODEL,
        google_api_key=api_key,
        temperature=0.2,
        timeout=config.LLM_TIMEOUT_SECONDS,
    )


_CLIENT_BUILDERS = {"groq": _build_groq_client, "gemini": _build_gemini_client}
_clients: dict[str, object] = {}


def _get_client(provider: str):
    if provider not in _clients:
        builder = _CLIENT_BUILDERS.get(provider)
        if builder is None:
            raise LLMConfigError(f"Unknown provider {provider!r}")
        _clients[provider] = builder()
    return _clients[provider]


def _extract_text(content) -> str:
    """Normalize a chat response's content to plain text.

    Newer Gemini models return a LIST of content blocks
    (``[{"type": "text", "text": ...}, ...]``) instead of a string;
    ``str()``-ing that would feed Python repr to the JSON parser.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text") or "")
        return "".join(parts)
    return str(content)


def _call_provider(provider: str, prompt: str) -> str:
    """Invoke one provider's chat model and return its text content."""
    client = _get_client(provider)
    response = client.invoke(prompt)
    return _extract_text(getattr(response, "content", response))


# ── public entry points ────────────────────────────────────────────────
def invoke(prompt: str, task_type: TaskType) -> str:
    """Route ``prompt`` to the right provider and return its text response.

    ``task_type="reasoning"`` prefers Groq (fast, short reasoning);
    ``task_type="long_context"`` prefers Gemini Flash. Identical
    ``(task_type, prompt)`` pairs are served from cache without touching
    any provider. If the preferred provider is at or over its tracked
    RPM/RPD limit, it is skipped in favor of the other provider before a
    request is even sent; if a call still fails (e.g. a live 429), the
    other provider is tried next. Raises ``LLMUnavailableError`` only if
    both providers are capped or fail.
    """
    if task_type not in _TASK_ROUTES:
        raise ValueError(f"Unknown task_type {task_type!r}; expected 'reasoning' or 'long_context'")

    key = _cache_key(prompt, task_type)
    cached = _cache_get(key)
    if cached is not None:
        logger.info("llm_router: cache hit task_type=%s key=%s", task_type, key[:12])
        return cached

    primary, fallback = _TASK_ROUTES[task_type]
    last_error: Exception | None = None

    for provider in (primary, fallback):
        if _quota.would_exceed(provider):
            logger.warning(
                "llm_router: skipping %s (predicted quota breach) task_type=%s", provider, task_type
            )
            continue

        logger.info("llm_router: routing task_type=%s -> %s", task_type, provider)
        _quota.record(provider)
        try:
            result = _call_provider(provider, prompt)
        except Exception as exc:  # noqa: BLE001 - provider SDKs raise assorted types
            logger.warning("llm_router: %s failed (%s); falling back", provider, exc)
            last_error = exc
            continue

        logger.info("llm_router: %s succeeded task_type=%s", provider, task_type)
        _cache_set(key, result)
        return result

    raise LLMUnavailableError(
        f"All providers unavailable for task_type={task_type!r}. Last error: {last_error}"
    )


_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)


def _strip_markdown_fences(text: str) -> str:
    match = _FENCE_RE.search(text)
    return match.group(1).strip() if match else text.strip()


def _try_parse_json(text: str):
    try:
        return json.loads(_strip_markdown_fences(text))
    except (json.JSONDecodeError, TypeError):
        return None


def invoke_json(prompt: str, schema_hint: str, task_type: TaskType = "reasoning"):
    """Like ``invoke()``, but instructs the model to answer as pure JSON.

    Appends ``schema_hint`` to the prompt with an explicit "JSON only, no
    markdown fences" instruction. Markdown code fences are stripped
    defensively either way. If the response doesn't parse, one repair
    retry is made — the model is shown its own broken output and asked to
    fix it. Raises ``LLMJSONParseError`` if it still doesn't parse.
    """
    instructed = (
        f"{prompt}\n\n"
        "Respond with ONLY valid JSON — no markdown code fences, no commentary, "
        "no text before or after the JSON. The JSON must match this shape:\n"
        f"{schema_hint}"
    )
    raw = invoke(instructed, task_type)
    parsed = _try_parse_json(raw)
    if parsed is not None:
        return parsed

    logger.warning("llm_router: invoke_json got unparseable output; attempting one repair retry")
    repair_prompt = (
        "The following output was supposed to be valid JSON matching this shape:\n"
        f"{schema_hint}\n\n"
        "but it failed to parse. Return ONLY the corrected, valid JSON — no markdown "
        "fences, no commentary.\n\nBROKEN OUTPUT:\n"
        f"{raw}"
    )
    repaired_raw = invoke(repair_prompt, task_type)
    parsed = _try_parse_json(repaired_raw)
    if parsed is not None:
        return parsed

    raise LLMJSONParseError(
        f"Could not parse JSON after one repair attempt. Last raw output: {repaired_raw[:500]!r}"
    )


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

    tiny_prompt = "Reply with exactly one word: OK"

    print("=== Direct provider smoke test ===")
    for provider in ("groq", "gemini"):
        print(f"\n--- {provider} ---")
        try:
            print(f"response: {_call_provider(provider, tiny_prompt)!r}")
        except Exception as exc:  # noqa: BLE001
            print(f"FAILED: {exc}")

    print("\n=== invoke() routing ===")
    for task_type in ("reasoning", "long_context"):
        print(f"\n--- task_type={task_type} ---")
        try:
            print(f"response: {invoke(tiny_prompt, task_type)!r}")
        except LLMUnavailableError as exc:
            print(f"FAILED: {exc}")

    print("\n=== invoke_json() ===")
    try:
        result = invoke_json("What is 2 + 2?", schema_hint='{"answer": <integer>}')
        print(f"parsed: {result!r}")
    except (LLMJSONParseError, LLMUnavailableError) as exc:
        print(f"FAILED: {exc}")

    print("\n=== cache hit check (same reasoning prompt again) ===")
    start = time.time()
    try:
        invoke(tiny_prompt, "reasoning")
        print(f"second call took {time.time() - start:.4f}s (should be near-instant)")
    except LLMUnavailableError as exc:
        print(f"FAILED: {exc}")
