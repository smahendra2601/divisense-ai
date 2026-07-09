"""Tier 2 — LLM Service: provider routing, quota tracking, response cache.

Single ``invoke(prompt, task_type)`` entry point plus
``invoke_json(prompt, schema_hint)``. Routes short reasoning to Groq
(llama-3.3-70b-versatile) and long-context work to Gemini Flash.
Tracks per-provider RPM/RPD counters, auto-falls back on 429, and
caches identical prompts — free tiers exhaust fast, so cache
aggressively.
"""
