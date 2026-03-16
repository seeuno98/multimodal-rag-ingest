from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from src.index.bm25 import build_bm25_index
from src.index.chunk import chunk_documents
from src.index.embed import FAILED_EMBEDDINGS_PATH, Embedder
from src.index.faiss_store import FaissStore
from src.ingest.normalize import read_jsonl, write_jsonl

LOGGER = logging.getLogger(__name__)


def build_index(
    docs_path: Path,
    chunks_path: Path,
    faiss_path: Path,
    metadata_path: Path,
    bm25_path: Path,
    openai_api_key: str,
    embed_model: str,
    chunk_max_chars: int,
    chunk_overlap_chars: int,
) -> tuple[int, int]:
    docs = read_jsonl(docs_path)
    chunks = chunk_documents(docs, max_chars=chunk_max_chars, overlap_chars=chunk_overlap_chars)
    write_jsonl(chunks_path, chunks)
    if not chunks:
        raise ValueError("No chunks produced. Run ingestion and normalization first.")
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY is required for embedding and indexing.")

    embedder = Embedder(api_key=openai_api_key, model=embed_model)
    vectors, successful_chunks, failed_chunks = embedder.embed_texts_with_failures(chunks)
    failed_embeddings_path = faiss_path.with_name(FAILED_EMBEDDINGS_PATH.name)
    write_jsonl(failed_embeddings_path, [failure.to_json() for failure in failed_chunks])
    if not successful_chunks:
        raise ValueError("All chunk embeddings failed; no index artifacts were produced.")
    store = FaissStore(dim=vectors.shape[1])
    store.add(vectors=vectors, metadata=successful_chunks)
    store.save(index_path=faiss_path, metadata_path=metadata_path)
    build_bm25_index(chunks=successful_chunks, out_path=bm25_path)
    LOGGER.info("Built BM25 index: chunks=%s path=%s", len(successful_chunks), bm25_path)
    LOGGER.info(
        "Indexed chunks attempted=%s embedded=%s failed=%s documents=%s failed_log=%s",
        len(chunks),
        len(successful_chunks),
        len(failed_chunks),
        len(docs),
        failed_embeddings_path,
    )
    return len(docs), len(successful_chunks)
