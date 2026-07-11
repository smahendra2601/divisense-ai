"""Tests for src/rag.py (Tier 2 — RAG retrieval).

The graceful-degradation contract is the important part: retrieve() must
return [] for bad input, a missing collection, an unknown ticker, or any
backend error — never raise. These are exercised with a fake collection
and a stubbed embedder so no model download or Chroma store is needed.
The real end-to-end path is covered by the module's own round-trip check.
"""

from __future__ import annotations

import pytest

from src import rag


@pytest.fixture(autouse=True)
def _reset_singletons(monkeypatch):
    monkeypatch.setattr(rag, "_collection", None)
    monkeypatch.setattr(rag, "_embedding_function", None)
    yield


class _FakeCollection:
    def __init__(self, response):
        self._response = response
        self.last_query = None

    def query(self, query_texts, n_results, where):
        self.last_query = {"n_results": n_results, "where": where}
        return self._response


def test_empty_ticker_returns_empty_list():
    assert rag.retrieve("") == []
    assert rag.retrieve("   ") == []


def test_missing_collection_returns_empty_list(monkeypatch):
    monkeypatch.setattr(rag, "get_collection", lambda create=True: None)
    assert rag.retrieve("ITC") == []


def test_no_documents_for_ticker_returns_empty_list(monkeypatch):
    empty = _FakeCollection({"documents": [[]], "metadatas": [[]], "distances": [[]]})
    monkeypatch.setattr(rag, "get_collection", lambda create=True: empty)
    assert rag.retrieve("ITC") == []


def test_backend_error_returns_empty_list(monkeypatch):
    def boom(create=True):
        raise RuntimeError("chroma exploded")

    monkeypatch.setattr(rag, "get_collection", boom)
    assert rag.retrieve("ITC") == []


def test_snippets_shape_and_scoring(monkeypatch):
    response = {
        "documents": [["dividend policy text", "capital allocation text"]],
        "metadatas": [
            [
                {"ticker": "ITC", "source_file": "itc_ar2025.pdf", "page": 12},
                {"ticker": "ITC", "source_file": "itc_ar2025.pdf", "page": 40},
            ]
        ],
        "distances": [[0.2, 0.6]],
    }
    fake = _FakeCollection(response)
    monkeypatch.setattr(rag, "get_collection", lambda create=True: fake)

    hits = rag.retrieve("itc", query="dividend", k=2)
    assert len(hits) == 2
    assert hits[0]["text"] == "dividend policy text"
    assert hits[0]["page"] == 12
    assert hits[0]["source_file"] == "itc_ar2025.pdf"
    assert hits[0]["score"] == pytest.approx(0.8)  # 1 - 0.2
    assert hits[1]["score"] == pytest.approx(0.4)  # 1 - 0.6
    # ticker is normalized to upper-case before the where-filter
    assert fake.last_query["where"] == {"ticker": "ITC"}
    assert fake.last_query["n_results"] == 2


def test_default_query_constant_used(monkeypatch):
    fake = _FakeCollection({"documents": [["x"]], "metadatas": [[{}]], "distances": [[0.1]]})
    monkeypatch.setattr(rag, "get_collection", lambda create=True: fake)
    hits = rag.retrieve("ITC")
    assert hits and hits[0]["ticker"] == "ITC"  # falls back to arg ticker when meta empty
