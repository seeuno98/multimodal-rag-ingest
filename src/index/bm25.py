from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import joblib
from rank_bm25 import BM25Okapi

TOKEN_PATTERN = re.compile(r"[^a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return [token for token in TOKEN_PATTERN.split(text.lower()) if token]


def build_bm25_index(chunks: list[dict[str, Any]], out_path: Path) -> dict[str, Any]:
    corpus_tokens = [tokenize(chunk.get("text", "")) for chunk in chunks]
    bm25 = BM25Okapi(corpus_tokens)
    chunk_ids = [str(chunk["chunk_id"]) for chunk in chunks]
    chunk_meta = {
        str(chunk["chunk_id"]): {
            "chunk_id": chunk["chunk_id"],
            "doc_id": chunk["doc_id"],
            "citation": chunk.get("metadata", {}).get("citation"),
            "source_uri": chunk.get("metadata", {}).get("source_uri"),
            "text": chunk.get("text", ""),
        }
        for chunk in chunks
    }
    payload = {"bm25": bm25, "chunk_ids": chunk_ids, "chunk_meta": chunk_meta}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, out_path)
    return payload


def load_bm25_index(path: Path) -> tuple[BM25Okapi, list[str], dict[str, dict[str, Any]]]:
    payload = joblib.load(path)
    return payload["bm25"], payload["chunk_ids"], payload["chunk_meta"]
