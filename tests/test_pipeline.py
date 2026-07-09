"""End-to-end smoke test for the LangGraph pipeline (LLM mocked).

Runs a query through Intent → Data → Ratio → RAG → Forecast → Critic →
Report with mocked LLM responses and verifies the final report carries
the timestamp, disclaimer, and reasoning chain.
"""
