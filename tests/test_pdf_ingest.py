"""Tests for src/pdf_ingest.py (Tier 1 — PDF → Chroma).

Chunking is tested with a stubbed tokenizer (offset-mapping based, like
the real fast tokenizer) so token-window math and overlap are verified
without loading the model. ingest_pdf's orchestration is tested with
extract_pages, the embedder, and the collection all faked — argument
validation, id/metadata construction, and upsert wiring, no PDF or model
required. A real PDF round-trip is covered separately by the module's
manual check.
"""

from __future__ import annotations

import pytest

from src import pdf_ingest


class _FakeTokenizer:
    """Mimics a HF fast tokenizer: 1 token per whitespace word, with offsets."""

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        offsets = []
        i = 0
        for word in text.split(" "):
            if word:
                start = text.index(word, i)
                offsets.append((start, start + len(word)))
                i = start + len(word)
        return {"offset_mapping": offsets}

    def encode(self, text, add_special_tokens=False):
        return text.split()


class _FakeModel:
    tokenizer = _FakeTokenizer()


@pytest.fixture
def fake_model(monkeypatch):
    monkeypatch.setattr(pdf_ingest, "get_embedding_model", lambda: _FakeModel())


# ── chunk_page_text ─────────────────────────────────────────────────
def test_empty_text_yields_no_chunks(fake_model):
    assert pdf_ingest.chunk_page_text("") == []


def test_short_text_is_single_chunk(fake_model):
    text = "one two three four five"
    assert pdf_ingest.chunk_page_text(text, size=800, overlap=100) == [text]


def test_long_text_splits_into_overlapping_windows(fake_model):
    words = [f"w{i}" for i in range(50)]
    text = " ".join(words)
    chunks = pdf_ingest.chunk_page_text(text, size=20, overlap=5)
    # step = 15 -> windows start at 0, 15, 30; the window at 30 reaches the
    # end (30+20 >= 50), so a 4th window would be fully redundant => 3 chunks.
    assert len(chunks) == 3
    assert chunks[0].startswith("w0 ")
    assert chunks[0].endswith(" w19")
    # overlap: chunk 1 starts 5 words before chunk 0's end (at w15)
    assert chunks[1].startswith("w15 ")
    assert chunks[1].endswith(" w34")
    # every word is covered, nothing dropped at the tail
    assert chunks[-1].endswith(" w49")


def test_chunks_preserve_original_casing(fake_model):
    text = "Dividend Policy And Capital Allocation Details"
    assert pdf_ingest.chunk_page_text(text, size=800, overlap=100) == [text]


# ── ingest_pdf orchestration ────────────────────────────────────────
class _FakeCollection:
    def __init__(self):
        self.upserts = []

    def upsert(self, ids, documents, embeddings, metadatas):
        self.upserts.append(
            {"ids": ids, "documents": documents, "embeddings": embeddings, "metadatas": metadatas}
        )


def test_ingest_pdf_rejects_blank_ticker(fake_model):
    with pytest.raises(ValueError):
        pdf_ingest.ingest_pdf("", "anything.pdf")


def test_ingest_pdf_rejects_missing_file(fake_model):
    with pytest.raises(FileNotFoundError):
        pdf_ingest.ingest_pdf("ITC", "does_not_exist_12345.pdf")


def test_ingest_pdf_builds_ids_metadata_and_upserts(monkeypatch, tmp_path, fake_model):
    pdf_path = tmp_path / "itc_ar2025.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 fake")  # just needs to exist

    monkeypatch.setattr(
        pdf_ingest,
        "extract_pages",
        lambda path: [(1, "dividend policy text"), (7, "capital allocation text")],
    )
    monkeypatch.setattr(pdf_ingest, "embed_texts", lambda docs: [[0.0] for _ in docs])
    fake_collection = _FakeCollection()
    monkeypatch.setattr(pdf_ingest, "get_collection", lambda create=True: fake_collection)

    count = pdf_ingest.ingest_pdf("itc", str(pdf_path))

    assert count == 2
    upsert = fake_collection.upserts[0]
    assert upsert["ids"] == ["ITC::itc_ar2025.pdf::p1::c0", "ITC::itc_ar2025.pdf::p7::c0"]
    assert upsert["metadatas"] == [
        {"ticker": "ITC", "source_file": "itc_ar2025.pdf", "page": 1},
        {"ticker": "ITC", "source_file": "itc_ar2025.pdf", "page": 7},
    ]


def test_ingest_pdf_returns_zero_when_no_text(monkeypatch, tmp_path, fake_model):
    pdf_path = tmp_path / "scanned.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 fake")
    monkeypatch.setattr(pdf_ingest, "extract_pages", lambda path: [])

    called = {"upsert": False}
    monkeypatch.setattr(
        pdf_ingest,
        "get_collection",
        lambda create=True: (_ for _ in ()).throw(AssertionError("should not be called")),
    )
    assert pdf_ingest.ingest_pdf("ITC", str(pdf_path)) == 0
    assert called["upsert"] is False
