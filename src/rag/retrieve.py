from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Literal

import numpy as np

from src.index.bm25 import load_bm25_index, tokenize
from src.index.embed import Embedder
from src.index.faiss_store import FaissStore

LOGGER = logging.getLogger(__name__)
DEFAULT_RRF_K = 60
DEFAULT_CANDIDATE_MULTIPLIER = 5
DEFAULT_MIN_CANDIDATES = 50


def select_retrieval_mode(
    index_path: Path,
    metadata_path: Path,
    bm25_path: Path,
    requested_mode: Literal["dense", "bm25", "hybrid"] | None = None,
) -> Literal["dense", "bm25", "hybrid"]:
    if requested_mode is not None:
        return requested_mode

    dense_ready = index_path.exists() and metadata_path.exists()
    bm25_ready = bm25_path.exists()
    if dense_ready and bm25_ready:
        return "hybrid"
    return "dense"


def _candidate_limit(top_k: int) -> int:
    return max(DEFAULT_MIN_CANDIDATES, top_k * DEFAULT_CANDIDATE_MULTIPLIER)


def _dense_retrieve(
    question: str,
    index_path: Path,
    metadata_path: Path,
    openai_api_key: str,
    embed_model: str,
    top_k: int,
) -> list[dict[str, Any]]:
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY is required for dense retrieval.")
    embedder = Embedder(api_key=openai_api_key, model=embed_model)
    q_vec = embedder.embed_query_texts([question])[0]
    store = FaissStore.load(index_path=index_path, metadata_path=metadata_path)
    hits = store.search(query_vector=q_vec, top_k=top_k)
    for rank, hit in enumerate(hits, start=1):
        metadata = hit.get("metadata", {})
        hit["rank"] = rank
        hit["chunk_id"] = hit.get("chunk_id") or metadata.get("chunk_id")
        hit["citation"] = hit.get("citation") or metadata.get("citation")
        hit["source_uri"] = hit.get("source_uri") or metadata.get("source_uri")
        hit["retrieval"] = "dense"
    return hits


def _bm25_retrieve(
    question: str,
    bm25_path: Path,
    top_k: int,
) -> list[dict[str, Any]]:
    bm25, chunk_ids, chunk_meta = load_bm25_index(bm25_path)
    scores = np.asarray(bm25.get_scores(tokenize(question)), dtype=float)
    limit = min(len(scores), _candidate_limit(top_k))
    if limit == 0:
        return []
    ranked_indices = np.argsort(scores)[::-1][:limit]

    results: list[dict[str, Any]] = []
    for rank, idx in enumerate(ranked_indices, start=1):
        chunk_id = chunk_ids[int(idx)]
        meta = chunk_meta[chunk_id]
        results.append(
            {
                "rank": rank,
                "score": float(scores[int(idx)]),
                "chunk_id": chunk_id,
                "doc_id": meta["doc_id"],
                "citation": meta.get("citation"),
                "source_uri": meta.get("source_uri"),
                "text": meta.get("text", ""),
                "retrieval": "bm25",
            }
        )
    return results


def rrf_fusion(
    dense_results: list[dict[str, Any]],
    bm25_results: list[dict[str, Any]],
    k: int = DEFAULT_RRF_K,
) -> list[dict[str, Any]]:
    fused_scores: dict[str, float] = {}
    merged_meta: dict[str, dict[str, Any]] = {}
    components: dict[str, dict[str, int | None]] = {}

    for result_list, key in ((dense_results, "dense_rank"), (bm25_results, "bm25_rank")):
        for rank, result in enumerate(result_list, start=1):
            chunk_id = str(result["chunk_id"])
            fused_scores[chunk_id] = fused_scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
            components.setdefault(chunk_id, {"dense_rank": None, "bm25_rank": None})[key] = rank
            current = merged_meta.get(chunk_id)
            if current is None or result.get("retrieval") == "dense":
                merged_meta[chunk_id] = result

    ranked_chunk_ids = sorted(fused_scores, key=lambda cid: fused_scores[cid], reverse=True)
    fused_results: list[dict[str, Any]] = []
    for output_rank, chunk_id in enumerate(ranked_chunk_ids, start=1):
        result = dict(merged_meta[chunk_id])
        result["rank"] = output_rank
        result["score"] = fused_scores[chunk_id]
        result["retrieval"] = "hybrid"
        result["components"] = components[chunk_id]
        fused_results.append(result)
    return fused_results


def retrieve(
    question: str,
    index_path: Path,
    metadata_path: Path,
    openai_api_key: str,
    embed_model: str,
    k: int = 5,
    mode: Literal["dense", "bm25", "hybrid"] | None = None,
    bm25_path: Path | None = None,
) -> list[dict[str, Any]]:
    bm25_index_path = bm25_path or index_path.with_name("bm25.joblib")
    resolved_mode = select_retrieval_mode(index_path, metadata_path, bm25_index_path, requested_mode=mode)

    if resolved_mode == "dense":
        if not bm25_index_path.exists():
            LOGGER.info("Retrieval mode: dense (bm25 index missing)")
        return _dense_retrieve(
            question=question,
            index_path=index_path,
            metadata_path=metadata_path,
            openai_api_key=openai_api_key,
            embed_model=embed_model,
            top_k=k,
        )

    if resolved_mode == "bm25":
        LOGGER.info("Retrieval mode: bm25")
        return _bm25_retrieve(question=question, bm25_path=bm25_index_path, top_k=k)[:k]

    dense_candidates = _dense_retrieve(
        question=question,
        index_path=index_path,
        metadata_path=metadata_path,
        openai_api_key=openai_api_key,
        embed_model=embed_model,
        top_k=_candidate_limit(k),
    )
    bm25_candidates = _bm25_retrieve(
        question=question,
        bm25_path=bm25_index_path,
        top_k=_candidate_limit(k),
    )
    LOGGER.info(
        "Retrieval mode: hybrid (dense_candidates=%s, bm25_candidates=%s, rrf_k=%s)",
        len(dense_candidates),
        len(bm25_candidates),
        DEFAULT_RRF_K,
    )
    return rrf_fusion(
        dense_results=dense_candidates,
        bm25_results=bm25_candidates,
        k=DEFAULT_RRF_K,
    )[:k]


def retrieve_chunks(
    question: str,
    index_path: Path,
    metadata_path: Path,
    openai_api_key: str,
    embed_model: str,
    top_k: int,
    mode: Literal["dense", "bm25", "hybrid"] | None = None,
    bm25_path: Path | None = None,
) -> list[dict[str, Any]]:
    return retrieve(
        question=question,
        index_path=index_path,
        metadata_path=metadata_path,
        openai_api_key=openai_api_key,
        embed_model=embed_model,
        k=top_k,
        mode=mode,
        bm25_path=bm25_path,
    )
