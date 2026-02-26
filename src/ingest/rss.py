from __future__ import annotations

import logging
import random
import time
from pathlib import Path
from typing import Any

import feedparser
import requests
import trafilatura

from src.ingest.normalize import append_jsonl, read_jsonl, stable_doc_id, utc_now_iso

LOGGER = logging.getLogger(__name__)
REQUEST_TIMEOUT_S = 15
MAX_RETRIES = 2
MAX_PER_FEED = 20
JITTER_MIN_S = 0.3
JITTER_MAX_S = 0.8
USER_AGENT = "multimodal-rag-ingest/0.1 (+https://github.com)"


def _load_feed_urls(path: Path) -> list[str]:
    urls: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            urls.append(line)
    return urls


def _fetch_html(url: str, session: requests.Session) -> tuple[str | None, bool]:
    rate_limited = False
    for attempt in range(MAX_RETRIES):
        try:
            response = session.get(
                url,
                timeout=REQUEST_TIMEOUT_S,
                headers={"User-Agent": USER_AGENT},
                allow_redirects=True,
            )
            if response.status_code == 429:
                rate_limited = True
                if attempt == 0:
                    time.sleep(0.5)
                    continue
                LOGGER.warning("Skipping %s after HTTP 429 retry", url)
                return None, True
            response.raise_for_status()
            return response.text, False
        except requests.HTTPError as exc:
            if attempt == MAX_RETRIES - 1:
                LOGGER.warning("HTTP error fetching %s: %s", url, exc)
                return None, rate_limited
        except requests.RequestException as exc:
            if attempt == MAX_RETRIES - 1:
                LOGGER.warning("Request failed for %s: %s", url, exc)
                return None, rate_limited
            time.sleep(0.4 * (attempt + 1))
    return None, rate_limited


def _extract_entry_doc(entry: dict[str, Any], session: requests.Session) -> tuple[dict[str, Any] | None, bool]:
    url = entry.get("link", "")
    if not url:
        return None, False
    html, was_rate_limited = _fetch_html(url, session)
    if not html:
        return None, was_rate_limited
    text = trafilatura.extract(html, include_comments=False, include_tables=False)
    if not text:
        return None, was_rate_limited
    published = entry.get("published", "") or entry.get("updated", "")
    doc_id = stable_doc_id(url)
    return (
        {
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
        },
        was_rate_limited,
    )


def ingest_rss(feeds_file: Path, raw_dir: Path) -> int:
    out_path = raw_dir / "rss_docs.jsonl"
    existing_docs = read_jsonl(out_path)
    existing_ids = {row["doc_id"] for row in existing_docs}
    feed_urls = _load_feed_urls(feeds_file)

    new_docs: list[dict[str, Any]] = []
    total_fetched = 0
    total_extracted = 0
    total_skipped = 0
    total_rate_limited = 0

    with requests.Session() as session:
        for feed_url in feed_urls:
            parsed = feedparser.parse(feed_url)
            for entry in parsed.entries[:MAX_PER_FEED]:
                total_fetched += 1
                doc, was_rate_limited = _extract_entry_doc(entry, session)
                if was_rate_limited:
                    total_rate_limited += 1
                if not doc:
                    total_skipped += 1
                    time.sleep(random.uniform(JITTER_MIN_S, JITTER_MAX_S))
                    continue
                if doc["doc_id"] in existing_ids:
                    total_skipped += 1
                    time.sleep(random.uniform(JITTER_MIN_S, JITTER_MAX_S))
                    continue
                new_docs.append(doc)
                existing_ids.add(doc["doc_id"])
                total_extracted += 1
                time.sleep(random.uniform(JITTER_MIN_S, JITTER_MAX_S))

    append_jsonl(out_path, new_docs)
    LOGGER.info(
        "RSS ingest summary: fetched=%s extracted=%s skipped=%s rate_limited=%s new_docs=%s",
        total_fetched,
        total_extracted,
        total_skipped,
        total_rate_limited,
        len(new_docs),
    )
    return len(new_docs)
