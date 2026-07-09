"""Tier 2 — Knowledge: RAG retrieval over annual reports.

Queries a persistent Chroma collection for dividend-policy /
capital-allocation snippets for a ticker, using the local embedding
model (``sentence-transformers/all-MiniLM-L6-v2``) — zero API cost.

``retrieve(ticker, query, k)`` is the public entry point. It is written
to **fail soft**: if Chroma is missing, empty, or has no documents for
the ticker — or anything else goes wrong — it returns ``[]`` so the rest
of the pipeline runs without RAG context rather than crashing.

This module also owns the two resources shared with ``pdf_ingest`` (the
embedding model and the Chroma collection) so ingestion and retrieval
embed identically and read/write the same store.
"""

from __future__ import annotations

import logging

from . import config

logger = logging.getLogger(__name__)

# All annual-report chunks live in one collection, partitioned by the
# ``ticker`` metadata field. Cosine space matches our normalized embeddings.
COLLECTION_NAME = "annual_reports"

DEFAULT_QUERY = "dividend policy capital allocation payout"

# Lazily-initialized singletons — importing this module must stay cheap
# (the pipeline imports it always; the heavy model loads only on first use).
_model = None
_collection = None


def get_embedding_model():
    """Load and cache the local sentence-transformers model (no network at query time)."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        logger.info("rag: loading embedding model %s", config.EMBEDDING_MODEL)
        _model = SentenceTransformer(config.EMBEDDING_MODEL)
    return _model


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a list of texts into normalized vectors (cosine-ready)."""
    model = get_embedding_model()
    vectors = model.encode(list(texts), normalize_embeddings=True, show_progress_bar=False)
    return [v.tolist() for v in vectors]


def get_collection(create: bool = True):
    """Return the persistent Chroma ``annual_reports`` collection.

    Cached per process. ``create=False`` is honored best-effort: if the
    collection does not yet exist it returns ``None`` instead of creating
    it, so a pure query never materializes an empty store.
    """
    global _collection
    if _collection is not None:
        return _collection

    import chromadb

    try:
        from chromadb.config import Settings

        client = chromadb.PersistentClient(
            path=str(config.CHROMA_DIR),
            settings=Settings(anonymized_telemetry=False),
        )
    except Exception:  # older/newer chromadb without Settings kwarg
        client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))

    if not create:
        existing = {c.name for c in client.list_collections()}
        if COLLECTION_NAME not in existing:
            return None

    _collection = client.get_or_create_collection(
        name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
    )
    return _collection


def retrieve(ticker: str, query: str = DEFAULT_QUERY, k: int = 4) -> list[dict]:
    """Return up to ``k`` annual-report snippets for ``ticker``.

    Each snippet is a dict: ``{text, ticker, source_file, page, score,
    distance}`` where ``score = 1 - distance`` (cosine similarity).
    Results are ordered most-relevant first.

    Returns ``[]`` gracefully whenever there is nothing to return — no
    ticker match, no collection yet, or any backend error — so callers
    can treat RAG context as strictly optional.
    """
    if not ticker or not ticker.strip():
        return []
    ticker = ticker.strip().upper()

    try:
        collection = get_collection(create=False)
        if collection is None:
            logger.info("rag: no collection yet; returning empty context for %s", ticker)
            return []

        query_embedding = embed_texts([query])[0]
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            where={"ticker": ticker},
        )
    except Exception as exc:  # noqa: BLE001 - retrieval must never break the pipeline
        logger.warning("rag: retrieval failed for %s (%s); returning empty context", ticker, exc)
        return []

    documents = (results.get("documents") or [[]])[0]
    metadatas = (results.get("metadatas") or [[]])[0]
    distances = (results.get("distances") or [[]])[0]

    if not documents:
        logger.info("rag: no documents on file for %s; returning empty context", ticker)
        return []

    snippets: list[dict] = []
    for doc, meta, dist in zip(documents, metadatas, distances):
        meta = meta or {}
        snippets.append(
            {
                "text": doc,
                "ticker": meta.get("ticker", ticker),
                "source_file": meta.get("source_file"),
                "page": meta.get("page"),
                "distance": dist,
                "score": (1.0 - dist) if dist is not None else None,
            }
        )
    return snippets


if __name__ == "__main__":
    import json
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

    if len(sys.argv) < 2:
        print('Usage: python -m src.rag <TICKER> [query]')
        sys.exit(1)

    tk = sys.argv[1]
    q = " ".join(sys.argv[2:]) if len(sys.argv) > 2 else DEFAULT_QUERY
    hits = retrieve(tk, q)
    print(f"\n{len(hits)} snippet(s) for {tk.upper()} — query: {q!r}\n")
    print(json.dumps(hits, indent=2, ensure_ascii=False))
