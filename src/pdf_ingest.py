"""Tier 1 — Data Acquisition: one-time annual-report → Chroma pipeline.

Parses an annual-report PDF with pdfplumber, chunks it at
``config.RAG_CHUNK_SIZE`` tokens with ``config.RAG_CHUNK_OVERLAP`` token
overlap (measured with the same ONNX tokenizer rag.py embeds with), and
upserts the chunks into the persistent Chroma ``annual_reports``
collection with metadata ``{ticker, source_file, page}`` — Chroma embeds
them itself via the collection's attached embedding function (see
``rag.get_embedding_function``), so ingestion and retrieval always embed
identically. Annual reports are static documents, so pre-embedding them
does not violate the fetch-on-demand freshness principle.

Chunking is done **per page** so the ``page`` metadata is always exact;
most report pages fall under one chunk, dense pages split into several
(all tagged with that page number). Character offsets from the tokenizer
are used to slice the *original* text, so stored snippets keep their
formatting and case rather than being detokenized.

Usage::

    python -m src.pdf_ingest <TICKER> <pdf_path>   # one file
    python -m src.pdf_ingest --all                 # every PDF in data/annual_reports/

Batch mode expects files named ``<TICKER>_<anything>.pdf`` (e.g.
``ITC_FY2025.pdf``) inside ``data/annual_reports/`` — the ticker is taken
from the part before the first underscore.

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
from .rag import get_collection, get_embedding_function

logger = logging.getLogger(__name__)

_tokenizer = None


def _get_chunking_tokenizer():
    """A raw (untruncated) tokenizer for finding chunk-boundary offsets.

    Loaded from the same ONNX model bundle rag.py's embedding function
    uses, so token counts here match what actually gets embedded.
    Deliberately not the embedding function's own ``.tokenizer`` — that
    one forces truncation/padding to 256 tokens for its own inference
    use, which would corrupt full-page chunk-boundary math.
    """
    global _tokenizer
    if _tokenizer is None:
        from tokenizers import Tokenizer

        embed_fn = get_embedding_function()
        embed_fn(["_warmup"])  # triggers the one-time model+tokenizer.json download
        tok_path = embed_fn.DOWNLOAD_PATH / embed_fn.EXTRACTED_FOLDER_NAME / "tokenizer.json"
        _tokenizer = Tokenizer.from_file(str(tok_path))
    return _tokenizer


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

    Uses the fast tokenizer's offset mapping to find token boundaries,
    then slices the source string by character offsets so formatting is
    kept (no detokenization, so casing/whitespace are exact).
    """
    if not text:
        return []

    tokenizer = _get_chunking_tokenizer()
    step = max(size - overlap, 1)

    encoding = tokenizer.encode(text, add_special_tokens=False)
    offsets = encoding.offsets

    chunks: list[str] = []
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
    collection = get_collection(create=True)
    collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
    logger.info("pdf_ingest: upserted %d chunk(s) for %s from %s", len(ids), ticker, source_file)
    return len(ids)


def ingest_all(reports_dir=None) -> dict[str, int]:
    """Ingest every ``<TICKER>_*.pdf`` in the annual-reports folder.

    Returns ``{filename: chunk_count}``; a failed file is logged and
    counted as 0 so one bad PDF never aborts the batch.
    """
    import glob

    reports_dir = str(reports_dir or config.ANNUAL_REPORTS_DIR)
    results: dict[str, int] = {}
    pdfs = sorted(glob.glob(os.path.join(reports_dir, "*.pdf")))
    if not pdfs:
        logger.warning("pdf_ingest: no PDFs found in %s", reports_dir)
        return results

    for path in pdfs:
        name = os.path.basename(path)
        ticker = name.split("_")[0].split(".")[0].upper()
        try:
            results[name] = ingest_pdf(ticker, path)
        except Exception as exc:  # noqa: BLE001 - keep the batch going
            logger.error("pdf_ingest: failed on %s (%s)", name, exc)
            results[name] = 0
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

    if len(sys.argv) == 2 and sys.argv[1] == "--all":
        summary = ingest_all()
        if not summary:
            print(f"⚠️  No PDFs found in {config.ANNUAL_REPORTS_DIR}.")
            sys.exit(1)
        print("\n=== Batch ingest summary ===")
        for name, count in summary.items():
            marker = "✅" if count else "⚠️ "
            print(f"{marker} {name}: {count} chunk(s)")
        sys.exit(0)

    if len(sys.argv) != 3:
        print("Usage: python -m src.pdf_ingest <TICKER> <pdf_path>")
        print("       python -m src.pdf_ingest --all")
        print("Example: python -m src.pdf_ingest ITC data/annual_reports/ITC_FY2025.pdf")
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
