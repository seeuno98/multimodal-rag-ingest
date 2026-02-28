from __future__ import annotations

from pathlib import Path

from src.index.embed import Embedder
from src.index.faiss_store import FaissStore


def retrieve_chunks(
    question: str,
    index_path: Path,
    metadata_path: Path,
    openai_api_key: str,
    embed_model: str,
    top_k: int,
) -> list[dict]:
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY is required for retrieval.")
    embedder = Embedder(api_key=openai_api_key, model=embed_model)
    q_vec = embedder.embed_query_texts([question])[0]
    store = FaissStore.load(index_path=index_path, metadata_path=metadata_path)
    hits = store.search(query_vector=q_vec, top_k=top_k)
    for hit in hits:
        metadata = hit.get("metadata", {})
        hit["citation"] = hit.get("citation") or metadata.get("citation")
        hit["source_uri"] = hit.get("source_uri") or metadata.get("source_uri")
    return hits
