"""Tier 1 — Data Acquisition: one-time annual-report → Chroma pipeline.

Parses annual report PDFs from ``data/annual_reports/`` with
pdfplumber, chunks (~800 tokens, 100 overlap per config), embeds with
the local sentence-transformers model, and stores in Chroma. Annual
reports are static documents, so pre-embedding does not violate the
fetch-on-demand freshness principle.
"""
