"""Tier 1 — Data Acquisition: one-time annual-report → Chroma pipeline.

Parses an annual-report PDF with pdfplumber, chunks it at
``config.RAG_CHUNK_SIZE`` tokens with ``config.RAG_CHUNK_OVERLAP`` token
overlap (measured with the embedding model's own tokenizer), embeds each
chunk locally with sentence-transformers (no API), and upserts them into
the persistent Chroma ``annual_reports`` collection with metadata
``{ticker, source_file, page}``. Annual reports are static documents, so
pre-embedding them does not violate the fetch-on-demand freshness
principle.

Chunking is done **per page** so the ``page`` metadata is always exact;
most report pages fall under one chunk, dense pages split into several
(all tagged with that page number). Character offsets from the tokenizer
are used to slice the *original* text, so stored snippets keep their
formatting and case rather than being detokenized.

Usage::

    python -m src.pdf_ingest <TICKER> <pdf_path>

Note: all-MiniLM-L6-v2 embeds at most its first ~256 tokens per chunk,
so with the default 800-token chunk size the tail of a very dense page
contributes less to its embedding. Lower ``RAG_CHUNK_SIZE`` in config to
match the model exactly if you want every token to count.
"""

from __future__ import annotations

import logging
import os
import sys

from . import config
from .rag import embed_texts, get_collection, get_embedding_model

logger = logging.getLogger(__name__)


def extract_pages(pdf_path: str) -> list[tuple[int, str]]:
    """Return ``[(page_number, text)]`` (1-indexed) for pages with text."""
    import pdfplumber

    pages: list[tuple[int, str]] = []
    with pdfplumber.open(pdf_path) as pdf:
        for idx, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            text = text.strip()
            if text:
                pages.append((idx, text))
    return pages


def chunk_page_text(
    text: str,
    size: int = config.RAG_CHUNK_SIZE,
    overlap: int = config.RAG_CHUNK_OVERLAP,
) -> list[str]:
    """Split ``text`` into overlapping token windows, preserving original text.

    Uses the embedding model's fast tokenizer to find token boundaries,
    then slices the source string by character offsets so formatting is
    kept. Falls back to detokenized decoding if offset mapping is
    unavailable.
    """
    if not text:
        return []

    # Silence the expected "token indices sequence length is longer than 256"
    # notice — we tokenize only to find chunk boundaries here, not to embed,
    # and it would otherwise fire once per page on a real report.
    try:
        from transformers.utils import logging as _hf_logging

        _hf_logging.set_verbosity_error()
    except Exception:  # pragma: no cover - transformers internals may move
        pass

    tokenizer = get_embedding_model().tokenizer
    step = max(size - overlap, 1)

    try:
        encoding = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        offsets = encoding["offset_mapping"]
    except (TypeError, KeyError, NotImplementedError):
        offsets = None

    chunks: list[str] = []
    if offsets:
        n = len(offsets)
        for start in range(0, n, step):
            window = offsets[start : start + size]
            if not window:
                break
            chunk = text[window[0][0] : window[-1][1]].strip()
            if chunk:
                chunks.append(chunk)
            if start + size >= n:
                break
    else:
        # Fallback: decode token windows directly (loses original casing).
        ids = tokenizer.encode(text, add_special_tokens=False)
        n = len(ids)
        for start in range(0, n, step):
            window = ids[start : start + size]
            if not window:
                break
            chunk = tokenizer.decode(window).strip()
            if chunk:
                chunks.append(chunk)
            if start + size >= n:
                break

    return chunks


def ingest_pdf(ticker: str, pdf_path: str) -> int:
    """Ingest one PDF for ``ticker`` into Chroma. Returns the chunk count.

    Chunks are upserted with deterministic ids, so re-running on the same
    file refreshes rather than duplicates.
    """
    if not ticker or not ticker.strip():
        raise ValueError("A ticker is required, e.g. 'ITC'.")
    if not os.path.isfile(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    ticker = ticker.strip().upper()
    source_file = os.path.basename(pdf_path)

    logger.info("pdf_ingest: extracting text from %s", pdf_path)
    pages = extract_pages(pdf_path)
    if not pages:
        logger.warning("pdf_ingest: no extractable text in %s (scanned/image PDF?)", pdf_path)
        return 0

    ids: list[str] = []
    documents: list[str] = []
    metadatas: list[dict] = []
    for page_number, page_text in pages:
        for cidx, chunk in enumerate(chunk_page_text(page_text)):
            ids.append(f"{ticker}::{source_file}::p{page_number}::c{cidx}")
            documents.append(chunk)
            metadatas.append(
                {"ticker": ticker, "source_file": source_file, "page": page_number}
            )

    if not documents:
        logger.warning("pdf_ingest: text extracted but produced no chunks for %s", pdf_path)
        return 0

    logger.info("pdf_ingest: embedding %d chunk(s) from %d page(s)", len(documents), len(pages))
    embeddings = embed_texts(documents)

    collection = get_collection(create=True)
    collection.upsert(
        ids=ids, documents=documents, embeddings=embeddings, metadatas=metadatas
    )
    logger.info("pdf_ingest: upserted %d chunk(s) for %s from %s", len(ids), ticker, source_file)
    return len(ids)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

    if len(sys.argv) != 3:
        print("Usage: python -m src.pdf_ingest <TICKER> <pdf_path>")
        print("Example: python -m src.pdf_ingest ITC data/annual_reports/itc_ar2025.pdf")
        sys.exit(1)

    ticker_arg, pdf_arg = sys.argv[1], sys.argv[2]
    try:
        count = ingest_pdf(ticker_arg, pdf_arg)
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)

    if count:
        print(f"\n✅ Ingested {count} chunk(s) for {ticker_arg.upper()} from {os.path.basename(pdf_arg)}.")
        print(f"   Stored in Chroma collection '{config.CHROMA_DIR}' → 'annual_reports'.")
    else:
        print(f"\n⚠️  Nothing ingested from {pdf_arg} (no extractable text).")
