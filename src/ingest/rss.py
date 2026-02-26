from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import feedparser
import trafilatura

from src.ingest.normalize import append_jsonl, read_jsonl, stable_doc_id, utc_now_iso

LOGGER = logging.getLogger(__name__)


def _load_feed_urls(path: Path) -> list[str]:
    urls: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            urls.append(line)
    return urls


def _extract_entry_doc(entry: dict[str, Any]) -> dict[str, Any] | None:
    url = entry.get("link", "")
    if not url:
        return None
    downloaded = trafilatura.fetch_url(url)
    if not downloaded:
        return None
    text = trafilatura.extract(downloaded, include_comments=False, include_tables=False)
    if not text:
        return None
    published = entry.get("published", "") or entry.get("updated", "")
    doc_id = stable_doc_id(url)
    return {
        "doc_id": doc_id,
        "source_type": "rss_blog",
        "title": entry.get("title", url),
        "source_uri": url,
        "created_at": utc_now_iso(),
        "segments": [
            {
                "segment_id": "s0",
                "text": text.strip(),
                "metadata": {"url": url, "published_at": published},
            }
        ],
    }


def ingest_rss(feeds_file: Path, raw_dir: Path) -> int:
    out_path = raw_dir / "rss_docs.jsonl"
    existing_docs = read_jsonl(out_path)
    existing_ids = {row["doc_id"] for row in existing_docs}
    feed_urls = _load_feed_urls(feeds_file)

    new_docs: list[dict[str, Any]] = []
    for feed_url in feed_urls:
        parsed = feedparser.parse(feed_url)
        for entry in parsed.entries:
            doc = _extract_entry_doc(entry)
            if not doc:
                continue
            if doc["doc_id"] in existing_ids:
                continue
            new_docs.append(doc)
            existing_ids.add(doc["doc_id"])

    append_jsonl(out_path, new_docs)
    LOGGER.info("Ingested %s new RSS documents", len(new_docs))
    return len(new_docs)
