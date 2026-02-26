from __future__ import annotations

import logging
import urllib.parse
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import fitz
import requests

from src.ingest.normalize import append_jsonl, read_jsonl, stable_doc_id, utc_now_iso

LOGGER = logging.getLogger(__name__)
ARXIV_API = "http://export.arxiv.org/api/query"
ATOM_NS = {"atom": "http://www.w3.org/2005/Atom"}


def _fetch_arxiv_entries(query: str, max_results: int) -> list[dict[str, Any]]:
    params = {
        "search_query": f"all:{query}",
        "start": 0,
        "max_results": max_results,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    }
    response = requests.get(ARXIV_API, params=params, timeout=30)
    response.raise_for_status()
    root = ET.fromstring(response.text)
    entries: list[dict[str, Any]] = []

    for entry in root.findall("atom:entry", ATOM_NS):
        entry_id = entry.findtext("atom:id", default="", namespaces=ATOM_NS).strip()
        title = " ".join(
            entry.findtext("atom:title", default="", namespaces=ATOM_NS).strip().split()
        )
        published = entry.findtext("atom:published", default="", namespaces=ATOM_NS).strip()
        pdf_link = ""
        for link in entry.findall("atom:link", ATOM_NS):
            href = link.attrib.get("href", "")
            link_type = link.attrib.get("type", "")
            title_attr = link.attrib.get("title", "")
            if title_attr == "pdf" or link_type == "application/pdf":
                pdf_link = href
                break
        if not pdf_link and entry_id:
            pdf_link = entry_id.replace("/abs/", "/pdf/") + ".pdf"
        if entry_id and title and pdf_link:
            entries.append(
                {"source_uri": entry_id, "title": title, "published_at": published, "pdf_url": pdf_link}
            )
    return entries


def _download_pdf(pdf_url: str, target_path: Path) -> None:
    if target_path.exists():
        return
    target_path.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(pdf_url, timeout=60, stream=True) as response:
        response.raise_for_status()
        with target_path.open("wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)


def _extract_pdf_segments(pdf_path: Path, source_url: str, published_at: str) -> list[dict[str, Any]]:
    segments: list[dict[str, Any]] = []
    with fitz.open(pdf_path) as doc:
        for i, page in enumerate(doc):
            text = page.get_text("text").strip()
            if not text:
                continue
            segments.append(
                {
                    "segment_id": f"p{i + 1}",
                    "text": text,
                    "metadata": {"page": i + 1, "url": source_url, "published_at": published_at},
                }
            )
    return segments


def ingest_arxiv(query: str, max_results: int, raw_dir: Path) -> int:
    out_path = raw_dir / "arxiv_docs.jsonl"
    pdf_dir = raw_dir / "arxiv_pdfs"
    existing_docs = read_jsonl(out_path)
    existing_ids = {row["doc_id"] for row in existing_docs}
    entries = _fetch_arxiv_entries(query, max_results=max_results)

    new_docs: list[dict[str, Any]] = []
    for entry in entries:
        doc_id = stable_doc_id(entry["source_uri"])
        if doc_id in existing_ids:
            continue
        safe_filename = urllib.parse.quote(doc_id, safe="") + ".pdf"
        pdf_path = pdf_dir / safe_filename
        try:
            _download_pdf(entry["pdf_url"], pdf_path)
            segments = _extract_pdf_segments(
                pdf_path, source_url=entry["source_uri"], published_at=entry["published_at"]
            )
            if not segments:
                LOGGER.warning("No text extracted from %s", entry["pdf_url"])
                continue
            new_docs.append(
                {
                    "doc_id": doc_id,
                    "source_type": "arxiv_pdf",
                    "title": entry["title"],
                    "source_uri": entry["source_uri"],
                    "created_at": utc_now_iso(),
                    "segments": segments,
                }
            )
        except requests.RequestException as exc:
            LOGGER.error("Failed downloading %s: %s", entry["pdf_url"], exc)
        except Exception as exc:
            LOGGER.error("Failed processing arXiv entry %s: %s", entry["source_uri"], exc)

    append_jsonl(out_path, new_docs)
    LOGGER.info("Ingested %s new arXiv documents", len(new_docs))
    return len(new_docs)
