from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path
from typing import Any

from openai import OpenAI

from src.ingest.normalize import append_jsonl, read_jsonl, stable_doc_id, utc_now_iso

LOGGER = logging.getLogger(__name__)


def _run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, check=True, text=True, capture_output=True)


def _load_urls(urls_file: Path) -> list[str]:
    urls: list[str] = []
    for line in urls_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            urls.append(line)
    return urls


def _fetch_video_metadata(url: str) -> dict[str, Any]:
    result = _run(["yt-dlp", "--dump-single-json", "--skip-download", url])
    return json.loads(result.stdout)


def _parse_vtt_to_segments(vtt_path: Path) -> list[dict[str, Any]]:
    segments: list[dict[str, Any]] = []
    if not vtt_path.exists():
        return segments

    def parse_ts(ts: str) -> float:
        hh, mm, ss = ts.split(":")
        return int(hh) * 3600 + int(mm) * 60 + float(ss)

    lines = vtt_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    for idx, line in enumerate(lines):
        if "-->" not in line:
            continue
        start_raw, end_raw = [t.strip().replace(",", ".") for t in line.split("-->")]
        text_lines: list[str] = []
        j = idx + 1
        while j < len(lines) and lines[j].strip():
            text_lines.append(lines[j].strip())
            j += 1
        text = " ".join(text_lines).strip()
        if not text:
            continue
        segments.append(
            {
                "segment_id": f"t{len(segments)}",
                "text": text,
                "metadata": {
                    "timestamp_start": parse_ts(start_raw),
                    "timestamp_end": parse_ts(end_raw),
                },
            }
        )
    return segments


def _download_captions(url: str, out_prefix: Path) -> Path | None:
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "yt-dlp",
        "--skip-download",
        "--write-auto-subs",
        "--write-subs",
        "--sub-langs",
        "en.*",
        "--sub-format",
        "vtt",
        "-o",
        str(out_prefix),
        url,
    ]
    subprocess.run(cmd, check=False, text=True, capture_output=True)
    candidates = sorted(out_prefix.parent.glob(out_prefix.name + "*.vtt"))
    return candidates[0] if candidates else None


def _download_audio(url: str, out_prefix: Path) -> Path:
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "yt-dlp",
        "-f",
        "bestaudio/best",
        "--extract-audio",
        "--audio-format",
        "mp3",
        "-o",
        str(out_prefix.with_suffix(".%(ext)s")),
        url,
    ]
    _run(cmd)
    mp3_path = out_prefix.with_suffix(".mp3")
    if not mp3_path.exists():
        matches = list(out_prefix.parent.glob(out_prefix.name + "*.mp3"))
        if not matches:
            raise FileNotFoundError(f"Audio file not found for {url}")
        return matches[0]
    return mp3_path


def _transcribe_audio(client: OpenAI, audio_path: Path) -> list[dict[str, Any]]:
    with audio_path.open("rb") as f:
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            file=f,
            response_format="verbose_json",
        )
    segments_raw = getattr(transcript, "segments", None) or []
    segments: list[dict[str, Any]] = []
    for i, seg in enumerate(segments_raw):
        seg_dict = seg.model_dump() if hasattr(seg, "model_dump") else dict(seg)
        text = str(seg_dict.get("text", "")).strip()
        if not text:
            continue
        segments.append(
            {
                "segment_id": f"w{i}",
                "text": text,
                "metadata": {
                    "timestamp_start": float(seg_dict.get("start", 0.0)),
                    "timestamp_end": float(seg_dict.get("end", 0.0)),
                },
            }
        )
    if not segments:
        text = getattr(transcript, "text", "")
        if text:
            segments = [
                {"segment_id": "w0", "text": text, "metadata": {"timestamp_start": 0.0, "timestamp_end": 0.0}}
            ]
    return segments


def ingest_youtube(urls_file: Path, raw_dir: Path, openai_api_key: str) -> int:
    out_path = raw_dir / "youtube_docs.jsonl"
    existing_docs = read_jsonl(out_path)
    existing_ids = {row["doc_id"] for row in existing_docs}

    urls = _load_urls(urls_file)
    client = OpenAI(api_key=openai_api_key) if openai_api_key else None
    new_docs: list[dict[str, Any]] = []
    for url in urls:
        doc_id = stable_doc_id(url)
        if doc_id in existing_ids:
            continue
        try:
            info = _fetch_video_metadata(url)
            title = info.get("title", url)
            source_uri = info.get("webpage_url", url)
            out_prefix = raw_dir / "youtube_assets" / doc_id
            caption_path = _download_captions(url, out_prefix)
            if caption_path:
                segments = _parse_vtt_to_segments(caption_path)
            else:
                if client is None:
                    LOGGER.warning("Skipping %s: no captions and OPENAI_API_KEY not set", url)
                    continue
                audio_path = _download_audio(url, out_prefix)
                segments = _transcribe_audio(client, audio_path)
            if not segments:
                LOGGER.warning("No transcript extracted for %s", url)
                continue
            for segment in segments:
                segment["metadata"]["url"] = source_uri
            new_docs.append(
                {
                    "doc_id": doc_id,
                    "source_type": "youtube",
                    "title": title,
                    "source_uri": source_uri,
                    "created_at": utc_now_iso(),
                    "segments": segments,
                }
            )
        except subprocess.CalledProcessError as exc:
            LOGGER.error("yt-dlp failed for %s: %s", url, exc.stderr or exc)
        except Exception as exc:
            LOGGER.error("Failed processing YouTube URL %s: %s", url, exc)

    append_jsonl(out_path, new_docs)
    LOGGER.info("Ingested %s new YouTube documents", len(new_docs))
    return len(new_docs)
