from __future__ import annotations

import logging
import re
import time
from pathlib import Path
from typing import Any, Literal

from src.config import AppConfig
from src.index.bm25 import load_bm25_index
from src.index.embed import Embedder
from src.index.faiss_store import FaissStore
from src.rag.answer import generate_grounded_answer
from src.rag.citations import build_citation_url_map
from src.rag.retrieve import RetrievalResources, RetrievalTimings, retrieve_chunks, retrieve_chunks_with_timings

LOGGER = logging.getLogger(__name__)
RETRIEVAL_MODES = ("dense", "bm25", "hybrid")
WHITESPACE_PATTERN = re.compile(r"\s+")


def _normalize_whitespace(text: str) -> str:
    return WHITESPACE_PATTERN.sub(" ", text).strip()


def _make_snippet(text: str, max_chars: int = 240) -> str:
    normalized = _normalize_whitespace(text)
    if len(normalized) <= max_chars:
        return normalized
    return normalized[: max_chars - 3].rstrip() + "..."


def _unique_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for value in values:
        if not value or value in seen:
            continue
        seen.add(value)
        unique.append(value)
    return unique


class RetrievalService:
    def __init__(self, cfg: AppConfig) -> None:
        self.cfg = cfg
        self.index_path = cfg.index_dir / "faiss.index"
        self.metadata_path = cfg.index_dir / "metadata.jsonl"
        self.bm25_path = cfg.index_dir / "bm25.joblib"
        self.resources = RetrievalResources()
        self._load_resources()

    def _load_resources(self) -> None:
        started = time.perf_counter()
        if self.index_path.exists() and self.metadata_path.exists():
            faiss_started = time.perf_counter()
            self.resources.faiss_store = FaissStore.load(
                index_path=self.index_path,
                metadata_path=self.metadata_path,
            )
            LOGGER.info(
                "Loaded FAISS store index_path=%s metadata_path=%s items=%s load_ms=%.2f",
                self.index_path,
                self.metadata_path,
                len(self.resources.faiss_store.metadata),
                (time.perf_counter() - faiss_started) * 1000,
            )

        if self.cfg.openai_api_key:
            self.resources.embedder = Embedder(
                api_key=self.cfg.openai_api_key,
                model=self.cfg.openai_embed_model,
            )
            LOGGER.info("Initialized embedder model=%s", self.cfg.openai_embed_model)

        if self.bm25_path.exists():
            bm25_started = time.perf_counter()
            bm25, chunk_ids, chunk_meta = load_bm25_index(self.bm25_path)
            self.resources.bm25 = bm25
            self.resources.bm25_chunk_ids = chunk_ids
            self.resources.bm25_chunk_meta = chunk_meta
            LOGGER.info(
                "Loaded BM25 artifacts path=%s chunks=%s load_ms=%.2f",
                self.bm25_path,
                len(chunk_ids),
                (time.perf_counter() - bm25_started) * 1000,
            )
        LOGGER.info("Retrieval resources ready total_load_ms=%.2f", (time.perf_counter() - started) * 1000)

    def status(self) -> dict[str, Any]:
        return {
            "ok": True,
            "faiss_loaded": self.resources.faiss_store is not None,
            "bm25_loaded": self.resources.bm25 is not None,
            "embedder_ready": self.resources.embedder is not None,
        }

    def _validate_mode(self, mode: Literal["dense", "bm25", "hybrid"]) -> None:
        if mode == "dense":
            if self.resources.faiss_store is None:
                raise RuntimeError("FAISS index is not loaded.")
            if self.resources.embedder is None:
                raise RuntimeError("OPENAI_API_KEY is required for dense retrieval.")
            return
        if mode == "bm25":
            if self.resources.bm25 is None:
                raise RuntimeError("BM25 artifacts are not loaded.")
            return
        if self.resources.faiss_store is None or self.resources.bm25 is None:
            raise RuntimeError("Hybrid retrieval requires both FAISS and BM25 artifacts.")
        if self.resources.embedder is None:
            raise RuntimeError("OPENAI_API_KEY is required for hybrid retrieval.")

    def retrieve(
        self,
        query: str,
        k: int = 5,
        mode: Literal["dense", "bm25", "hybrid"] = "dense",
    ) -> list[dict[str, Any]]:
        self._validate_mode(mode)
        return retrieve_chunks(
            question=query,
            index_path=self.index_path,
            metadata_path=self.metadata_path,
            openai_api_key=self.cfg.openai_api_key,
            embed_model=self.cfg.openai_embed_model,
            top_k=k,
            mode=mode,
            bm25_path=self.bm25_path,
            resources=self.resources,
        )

    def retrieve_with_timings(
        self,
        query: str,
        k: int = 5,
        mode: Literal["dense", "bm25", "hybrid"] = "dense",
    ) -> tuple[list[dict[str, Any]], RetrievalTimings]:
        self._validate_mode(mode)
        results, timings = retrieve_chunks_with_timings(
            question=query,
            index_path=self.index_path,
            metadata_path=self.metadata_path,
            openai_api_key=self.cfg.openai_api_key,
            embed_model=self.cfg.openai_embed_model,
            top_k=k,
            mode=mode,
            bm25_path=self.bm25_path,
            resources=self.resources,
        )
        LOGGER.info(
            (
                "retrieve_breakdown mode=%s k=%s query_chars=%s total_ms=%.2f "
                "embedding_ms=%.2f dense_ms=%.2f bm25_ms=%.2f fusion_ms=%.2f"
            ),
            mode,
            k,
            len(query),
            timings.total_ms,
            timings.embedding_ms,
            timings.dense_retrieval_ms,
            timings.bm25_retrieval_ms,
            timings.fusion_ms,
        )
        return results, timings

    def answer(
        self,
        query: str,
        k: int = 5,
        mode: Literal["dense", "bm25", "hybrid"] = "dense",
    ) -> dict[str, Any]:
        results = self.retrieve(query=query, k=k, mode=mode)
        answer = generate_grounded_answer(
            question=query,
            retrieved_chunks=results,
            openai_api_key=self.cfg.openai_api_key,
            chat_model=self.cfg.openai_chat_model,
        )
        citations = _unique_preserve_order(
            [
                result.get("citation") or result.get("metadata", {}).get("citation", "")
                for result in results
            ]
        )
        citation_urls = build_citation_url_map(results)
        return {
            "answer": answer,
            "citations": citations,
            "citation_urls": citation_urls,
            "results": results,
        }


def format_retrieval_result(result: dict[str, Any]) -> dict[str, Any]:
    text = str(result.get("text", ""))
    return {
        "rank": int(result.get("rank", 0)),
        "doc_id": str(result.get("doc_id", "")),
        "chunk_id": str(result.get("chunk_id", "")) or None,
        "retrieval": result.get("retrieval"),
        "citation": result.get("citation") or result.get("metadata", {}).get("citation"),
        "source_uri": result.get("source_uri") or result.get("metadata", {}).get("source_uri"),
        "score": float(result["score"]) if "score" in result else None,
        "snippet": _make_snippet(text),
        "components": result.get("components"),
    }
