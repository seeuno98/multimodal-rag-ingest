from __future__ import annotations

import hashlib
import re
from typing import Any


def _split_paragraphs(text: str) -> list[str]:
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    return paragraphs if paragraphs else [text.strip()]


def _make_chunks(text: str, max_chars: int, overlap_chars: int) -> list[str]:
    paragraphs = _split_paragraphs(text)
    chunks: list[str] = []
    current = ""
    for para in paragraphs:
        if len(current) + len(para) + 2 <= max_chars:
            current = (current + "\n\n" + para).strip()
            continue
        if current:
            chunks.append(current)
        if len(para) <= max_chars:
            current = para
            continue
        start = 0
        while start < len(para):
            end = min(start + max_chars, len(para))
            chunks.append(para[start:end].strip())
            if end >= len(para):
                break
            start = max(0, end - overlap_chars)
        current = ""
    if current:
        chunks.append(current)
    return [c for c in chunks if c]


def _timestamp_to_hhmmss(seconds: float) -> str:
    total = int(max(0, round(seconds)))
    h = total // 3600
    m = (total % 3600) // 60
    s = total % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def _build_citation(source_type: str, doc_id: str, metadata: dict[str, Any]) -> str:
    if source_type == "arxiv_pdf":
        page = metadata.get("page")
        if page is not None:
            return f"[arxiv:{doc_id} p.{page}]"
        return f"[arxiv:{doc_id}]"
    if source_type == "youtube":
        ts = _timestamp_to_hhmmss(float(metadata.get("timestamp_start", 0.0)))
        return f"[youtube:{doc_id} {ts}]"
    return f"[rss:{doc_id}]"


def chunk_documents(
    docs: list[dict[str, Any]], max_chars: int, overlap_chars: int
) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    for doc in docs:
        doc_id = doc["doc_id"]
        source_type = doc["source_type"]
        for seg in doc.get("segments", []):
            seg_meta = seg.get("metadata", {})
            for idx, text_chunk in enumerate(
                _make_chunks(seg.get("text", ""), max_chars=max_chars, overlap_chars=overlap_chars)
            ):
                chunk_id_seed = f"{doc_id}:{seg['segment_id']}:{idx}:{text_chunk}"
                chunk_id = hashlib.sha1(chunk_id_seed.encode("utf-8")).hexdigest()[:20]
                metadata = {
                    **seg_meta,
                    "source_type": source_type,
                    "title": doc.get("title", ""),
                    "source_uri": doc.get("source_uri", ""),
                }
                metadata["citation"] = _build_citation(source_type, doc_id, metadata)
                chunks.append(
                    {
                        "chunk_id": chunk_id,
                        "doc_id": doc_id,
                        "text": text_chunk,
                        "metadata": metadata,
                    }
                )
    return chunks
