from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from rank_bm25 import BM25Okapi

from src.index.bm25 import load_bm25_index, tokenize
from src.index.embed import Embedder
from src.index.faiss_store import FaissStore

LOGGER = logging.getLogger(__name__)
DEFAULT_RRF_K = 60
DEFAULT_CANDIDATE_MULTIPLIER = 5
DEFAULT_MIN_CANDIDATES = 50


@dataclass
class RetrievalTimings:
    total_ms: float = 0.0
    embedding_ms: float = 0.0
    dense_retrieval_ms: float = 0.0
    bm25_retrieval_ms: float = 0.0
    fusion_ms: float = 0.0


@dataclass
class RetrievalResources:
    embedder: Embedder | None = None
    faiss_store: FaissStore | None = None
    bm25: BM25Okapi | None = None
    bm25_chunk_ids: list[str] | None = None
    bm25_chunk_meta: dict[str, dict[str, Any]] | None = None


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
    resources: RetrievalResources | None = None,
) -> tuple[list[dict[str, Any]], float, float]:
    embedder = resources.embedder if resources is not None else None
    if embedder is None and not openai_api_key:
        raise ValueError("OPENAI_API_KEY is required for dense retrieval.")
    if embedder is None:
        embedder = Embedder(api_key=openai_api_key, model=embed_model)

    embedding_started = time.perf_counter()
    q_vec = embedder.embed_query_texts([question])[0]
    embedding_ms = (time.perf_counter() - embedding_started) * 1000

    store = resources.faiss_store if resources is not None else None
    if store is None:
        store = FaissStore.load(index_path=index_path, metadata_path=metadata_path)

    search_started = time.perf_counter()
    hits = store.search(query_vector=q_vec, top_k=top_k)
    dense_retrieval_ms = (time.perf_counter() - search_started) * 1000
    for rank, hit in enumerate(hits, start=1):
        metadata = hit.get("metadata", {})
        hit["rank"] = rank
        hit["chunk_id"] = hit.get("chunk_id") or metadata.get("chunk_id")
        hit["citation"] = hit.get("citation") or metadata.get("citation")
        hit["source_uri"] = hit.get("source_uri") or metadata.get("source_uri")
        hit["retrieval"] = "dense"
    return hits, embedding_ms, dense_retrieval_ms


def _bm25_retrieve(
    question: str,
    bm25_path: Path,
    top_k: int,
    resources: RetrievalResources | None = None,
) -> tuple[list[dict[str, Any]], float]:
    started = time.perf_counter()
    if (
        resources is not None
        and resources.bm25 is not None
        and resources.bm25_chunk_ids is not None
        and resources.bm25_chunk_meta is not None
    ):
        bm25 = resources.bm25
        chunk_ids = resources.bm25_chunk_ids
        chunk_meta = resources.bm25_chunk_meta
    else:
        bm25, chunk_ids, chunk_meta = load_bm25_index(bm25_path)
    scores = np.asarray(bm25.get_scores(tokenize(question)), dtype=float)
    limit = min(len(scores), _candidate_limit(top_k))
    if limit == 0:
        return [], (time.perf_counter() - started) * 1000
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
    return results, (time.perf_counter() - started) * 1000


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


def fuse_ranked_results(
    dense_results: list[dict[str, Any]],
    bm25_results: list[dict[str, Any]],
    top_k: int,
    rrf_k: int = DEFAULT_RRF_K,
) -> list[dict[str, Any]]:
    return rrf_fusion(dense_results=dense_results, bm25_results=bm25_results, k=rrf_k)[:top_k]


def retrieve(
    question: str,
    index_path: Path,
    metadata_path: Path,
    openai_api_key: str,
    embed_model: str,
    k: int = 5,
    mode: Literal["dense", "bm25", "hybrid"] | None = None,
    bm25_path: Path | None = None,
    resources: RetrievalResources | None = None,
) -> list[dict[str, Any]]:
    results, _ = retrieve_with_timings(
        question=question,
        index_path=index_path,
        metadata_path=metadata_path,
        openai_api_key=openai_api_key,
        embed_model=embed_model,
        k=k,
        mode=mode,
        bm25_path=bm25_path,
        resources=resources,
    )
    return results


def retrieve_with_timings(
    question: str,
    index_path: Path,
    metadata_path: Path,
    openai_api_key: str,
    embed_model: str,
    k: int = 5,
    mode: Literal["dense", "bm25", "hybrid"] | None = None,
    bm25_path: Path | None = None,
    resources: RetrievalResources | None = None,
) -> tuple[list[dict[str, Any]], RetrievalTimings]:
    bm25_index_path = bm25_path or index_path.with_name("bm25.joblib")
    resolved_mode = select_retrieval_mode(index_path, metadata_path, bm25_index_path, requested_mode=mode)
    timings = RetrievalTimings()
    started = time.perf_counter()

    if resolved_mode == "dense":
        if not bm25_index_path.exists():
            LOGGER.info("Retrieval mode: dense (bm25 index missing)")
        dense_results, embedding_ms, dense_retrieval_ms = _dense_retrieve(
            question=question,
            index_path=index_path,
            metadata_path=metadata_path,
            openai_api_key=openai_api_key,
            embed_model=embed_model,
            top_k=k,
            resources=resources,
        )
        timings.embedding_ms = embedding_ms
        timings.dense_retrieval_ms = dense_retrieval_ms
        timings.total_ms = (time.perf_counter() - started) * 1000
        return dense_results, timings

    if resolved_mode == "bm25":
        LOGGER.info("Retrieval mode: bm25")
        bm25_results, bm25_retrieval_ms = _bm25_retrieve(
            question=question,
            bm25_path=bm25_index_path,
            top_k=k,
            resources=resources,
        )
        timings.bm25_retrieval_ms = bm25_retrieval_ms
        timings.total_ms = (time.perf_counter() - started) * 1000
        return bm25_results[:k], timings

    dense_candidates, embedding_ms, dense_retrieval_ms = _dense_retrieve(
        question=question,
        index_path=index_path,
        metadata_path=metadata_path,
        openai_api_key=openai_api_key,
        embed_model=embed_model,
        top_k=_candidate_limit(k),
        resources=resources,
    )
    bm25_candidates, bm25_retrieval_ms = _bm25_retrieve(
        question=question,
        bm25_path=bm25_index_path,
        top_k=_candidate_limit(k),
        resources=resources,
    )
    LOGGER.info(
        "Retrieval mode: hybrid (dense_candidates=%s, bm25_candidates=%s, rrf_k=%s)",
        len(dense_candidates),
        len(bm25_candidates),
        DEFAULT_RRF_K,
    )
    fusion_started = time.perf_counter()
    fused_results = rrf_fusion(
        dense_results=dense_candidates,
        bm25_results=bm25_candidates,
        k=DEFAULT_RRF_K,
    )[:k]
    timings.embedding_ms = embedding_ms
    timings.dense_retrieval_ms = dense_retrieval_ms
    timings.bm25_retrieval_ms = bm25_retrieval_ms
    timings.fusion_ms = (time.perf_counter() - fusion_started) * 1000
    timings.total_ms = (time.perf_counter() - started) * 1000
    return fused_results, timings


def retrieve_chunks(
    question: str,
    index_path: Path,
    metadata_path: Path,
    openai_api_key: str,
    embed_model: str,
    top_k: int,
    mode: Literal["dense", "bm25", "hybrid"] | None = None,
    bm25_path: Path | None = None,
    resources: RetrievalResources | None = None,
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
        resources=resources,
    )


def retrieve_chunks_with_timings(
    question: str,
    index_path: Path,
    metadata_path: Path,
    openai_api_key: str,
    embed_model: str,
    top_k: int,
    mode: Literal["dense", "bm25", "hybrid"] | None = None,
    bm25_path: Path | None = None,
    resources: RetrievalResources | None = None,
) -> tuple[list[dict[str, Any]], RetrievalTimings]:
    return retrieve_with_timings(
        question=question,
        index_path=index_path,
        metadata_path=metadata_path,
        openai_api_key=openai_api_key,
        embed_model=embed_model,
        k=top_k,
        mode=mode,
        bm25_path=bm25_path,
        resources=resources,
    )
