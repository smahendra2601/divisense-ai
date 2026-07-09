"""Tier 3 — Agentic Orchestration: LangGraph StateGraph.

Defines the typed ``DivisenseState`` (TypedDict) and wires the nodes:
Intent Agent → Data Node → Ratio Node → RAG Node → Forecast Agent
(question-aware) → Critic Agent → Report Node. The Critic can loop
back to Forecast once on failure (retry_count guard); clarify /
out_of_scope / errors route straight to the Report Node. At most 3
LLM calls per query (2 for bare-ticker input).
"""
