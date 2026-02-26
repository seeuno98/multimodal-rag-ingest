from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger(__name__)


def stable_doc_id(source_uri: str) -> str:
    return hashlib.sha1(source_uri.encode("utf-8")).hexdigest()[:16]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_documents(raw_dir: Path, output_path: Path) -> int:
    in_files = [
        raw_dir / "arxiv_docs.jsonl",
        raw_dir / "youtube_docs.jsonl",
        raw_dir / "rss_docs.jsonl",
    ]
    seen_ids: set[str] = set()
    all_docs: list[dict[str, Any]] = []
    for in_file in in_files:
        docs = read_jsonl(in_file)
        for doc in docs:
            doc_id = doc["doc_id"]
            if doc_id in seen_ids:
                continue
            seen_ids.add(doc_id)
            all_docs.append(doc)

    write_jsonl(output_path, all_docs)
    LOGGER.info("Normalized %s documents into %s", len(all_docs), output_path)
    return len(all_docs)
