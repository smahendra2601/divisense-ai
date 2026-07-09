"""Tier 2 — Knowledge: RAG retrieval over annual reports.

Queries Chroma for dividend-policy / capital-allocation snippets for a
ticker using the local embedding model
(``sentence-transformers/all-MiniLM-L6-v2``) — zero API cost. If no
documents exist for a ticker, returns empty context gracefully; the
pipeline must still work.
"""
