from __future__ import annotations

import json
import logging
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Sequence

import httpx
import numpy as np
import openai
from openai import OpenAI

LOGGER = logging.getLogger(__name__)

BATCH_SIZE = 32
MAX_ATTEMPTS = 6
BASE_BACKOFF_S = 0.5
BACKOFF_CAP_S = 20.0
SUCCESS_JITTER_MIN_S = 0.05
SUCCESS_JITTER_MAX_S = 0.15
RETRY_JITTER_MAX_S = 0.25
RETRY_MS_PATTERN = re.compile(r"Please try again in (\d+)ms", re.IGNORECASE)
CACHE_PATH = Path("data/index/embeddings_cache.jsonl")
CACHE_META_PATH = Path("data/index/embeddings_cache_meta.json")
MODEL_DIMENSIONS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}


@dataclass(frozen=True)
class RetryDecision:
    should_retry: bool
    sleep_s: float
    reason: str


@dataclass(frozen=True)
class EmbeddingRequest:
    chunk_id: str | None
    text: str


def parse_retry_ms(message: str) -> int | None:
    match = RETRY_MS_PATTERN.search(message)
    if not match:
        return None
    return int(match.group(1))


def compute_backoff_s(attempt: int, cap_s: float = BACKOFF_CAP_S) -> float:
    if attempt < 1:
        raise ValueError("attempt must be >= 1")
    return min(BASE_BACKOFF_S * (2 ** (attempt - 1)), cap_s)


def load_embedding_cache(path: Path) -> dict[str, list[float]]:
    if not path.exists():
        return {}

    cache: dict[str, list[float]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                LOGGER.warning("Skipping malformed embedding cache line %s in %s", line_number, path)
                continue

            chunk_id = row.get("chunk_id")
            embedding = row.get("embedding")
            if not isinstance(chunk_id, str) or not isinstance(embedding, list):
                LOGGER.warning("Skipping invalid embedding cache record on line %s in %s", line_number, path)
                continue
            cache[chunk_id] = embedding
    return cache


def append_embeddings_to_cache(path: Path, items: list[tuple[str, list[float]]]) -> None:
    if not items:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for chunk_id, embedding in items:
            f.write(json.dumps({"chunk_id": chunk_id, "embedding": embedding}) + "\n")


def load_cache_meta(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        LOGGER.warning("Failed to load embedding cache metadata from %s", path)
        return None
    return data if isinstance(data, dict) else None


def write_cache_meta(path: Path, model: str, dimension: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump({"model": model, "dimension": dimension}, f)


def validate_cache_meta(meta: dict[str, Any] | None, model: str, dimension: int) -> bool:
    if not meta:
        return False
    return meta.get("model") == model and meta.get("dimension") == dimension


def _expected_dimension_for_model(model: str) -> int | None:
    return MODEL_DIMENSIONS.get(model)


def _error_message(exc: Exception) -> str:
    message = str(exc)
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        error = body.get("error")
        if isinstance(error, dict) and isinstance(error.get("message"), str):
            message = f"{message} {error['message']}"
    return message


def _status_code(exc: Exception) -> int | None:
    status_code = getattr(exc, "status_code", None)
    if isinstance(status_code, int):
        return status_code
    response = getattr(exc, "response", None)
    response_status = getattr(response, "status_code", None)
    return response_status if isinstance(response_status, int) else None


def is_retryable_error(exc: Exception) -> bool:
    timeout_types: tuple[type[BaseException], ...] = (
        openai.APITimeoutError,
        httpx.ConnectTimeout,
        httpx.ReadTimeout,
        httpx.TimeoutException,
    )
    api_connection_error = getattr(openai, "APIConnectionError", None)
    if api_connection_error is not None:
        timeout_types = timeout_types + (api_connection_error,)

    if isinstance(exc, timeout_types):
        return True

    status_code = _status_code(exc)
    message = _error_message(exc).lower()
    if status_code == 429 or "rate_limit_exceeded" in message:
        return True
    if status_code is not None and status_code >= 500:
        return True
    return False


def get_retry_decision(exc: Exception, attempt: int) -> RetryDecision:
    if not is_retryable_error(exc):
        return RetryDecision(should_retry=False, sleep_s=0.0, reason="non_retryable")

    retry_ms = parse_retry_ms(_error_message(exc))
    jitter_s = random.uniform(0.0, RETRY_JITTER_MAX_S)
    if retry_ms is not None:
        return RetryDecision(
            should_retry=True,
            sleep_s=retry_ms / 1000.0 + jitter_s,
            reason="server_retry_hint",
        )

    return RetryDecision(
        should_retry=True,
        sleep_s=compute_backoff_s(attempt) + jitter_s,
        reason=type(exc).__name__,
    )


class Embedder:
    def __init__(
        self,
        api_key: str,
        model: str,
        batch_size: int = BATCH_SIZE,
        cache_path: Path = CACHE_PATH,
        cache_meta_path: Path = CACHE_META_PATH,
    ) -> None:
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.batch_size = batch_size
        self.cache_path = cache_path
        self.cache_meta_path = cache_meta_path
        self.retry_count = 0
        self.total_batches = 0

    def _normalize_requests(self, texts: Sequence[dict[str, Any] | str]) -> list[EmbeddingRequest]:
        requests: list[EmbeddingRequest] = []
        for item in texts:
            if isinstance(item, dict):
                requests.append(
                    EmbeddingRequest(
                        chunk_id=item.get("chunk_id"),
                        text=str(item.get("text", "")),
                    )
                )
            else:
                requests.append(EmbeddingRequest(chunk_id=None, text=str(item)))
        return requests

    def _embed_batch(self, batch: list[str], batch_index: int) -> list[list[float]]:
        self.total_batches += 1
        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                response = self.client.embeddings.create(model=self.model, input=batch)
                time.sleep(random.uniform(SUCCESS_JITTER_MIN_S, SUCCESS_JITTER_MAX_S))
                return [item.embedding for item in response.data]
            except Exception as exc:
                decision = get_retry_decision(exc, attempt=attempt)
                if not decision.should_retry or attempt == MAX_ATTEMPTS:
                    raise RuntimeError(
                        f"Embedding batch {batch_index} failed after {attempt} attempts "
                        f"for {len(batch)} texts: {type(exc).__name__}: {_error_message(exc)}"
                    ) from exc
                self.retry_count += 1
                LOGGER.warning(
                    "Embedding retry batch=%s attempt=%s/%s error=%s sleep=%.3fs reason=%s",
                    batch_index,
                    attempt,
                    MAX_ATTEMPTS,
                    type(exc).__name__,
                    decision.sleep_s,
                    decision.reason,
                )
                time.sleep(decision.sleep_s)

        raise RuntimeError(f"Embedding batch {batch_index} exhausted retry loop unexpectedly.")

    def embed_texts(self, texts: Sequence[dict[str, Any] | str]) -> np.ndarray:
        return self._embed_texts(texts, context="index")

    def embed_query_texts(self, texts: Sequence[str]) -> np.ndarray:
        return self._embed_texts(texts, context="query")

    def _embed_texts(
        self,
        texts: Sequence[dict[str, Any] | str],
        context: Literal["index", "query"] = "index",
    ) -> np.ndarray:
        if not texts:
            return np.empty((0, 0), dtype=np.float32)

        requests = self._normalize_requests(texts)
        use_cache = context == "index" and all(request.chunk_id for request in requests)
        expected_dimension = _expected_dimension_for_model(self.model)
        cache: dict[str, list[float]] = {}
        cache_hits = 0
        cache_misses = 0
        new_cache_items: list[tuple[str, list[float]]] = []
        self.retry_count = 0
        self.total_batches = 0

        if use_cache and expected_dimension is not None:
            meta = load_cache_meta(self.cache_meta_path)
            if meta is not None and not validate_cache_meta(meta, self.model, expected_dimension):
                LOGGER.warning(
                    "Embedding cache metadata mismatch for %s; ignoring old cache and resetting metadata",
                    self.model,
                )
                cache = {}
                self.cache_path.parent.mkdir(parents=True, exist_ok=True)
                self.cache_path.write_text("", encoding="utf-8")
                write_cache_meta(self.cache_meta_path, self.model, expected_dimension)
            else:
                cache = load_embedding_cache(self.cache_path)
                if meta is None:
                    write_cache_meta(self.cache_meta_path, self.model, expected_dimension)

        ordered_embeddings: list[list[float] | None] = [None] * len(requests)
        misses: list[tuple[int, str, str]] = []
        for idx, request in enumerate(requests):
            if use_cache and request.chunk_id and request.chunk_id in cache:
                ordered_embeddings[idx] = cache[request.chunk_id]
                cache_hits += 1
            else:
                if use_cache:
                    cache_misses += 1
                misses.append((idx, request.chunk_id or "", request.text))

        for batch_number, start in enumerate(range(0, len(misses), self.batch_size), start=1):
            batch = misses[start : start + self.batch_size]
            batch_texts = [text for _, _, text in batch]
            batch_vectors = self._embed_batch(batch_texts, batch_index=batch_number)
            if expected_dimension is None and batch_vectors:
                expected_dimension = len(batch_vectors[0])
                write_cache_meta(self.cache_meta_path, self.model, expected_dimension)
            for (idx, chunk_id, _), embedding in zip(batch, batch_vectors):
                ordered_embeddings[idx] = embedding
                if use_cache and chunk_id:
                    new_cache_items.append((chunk_id, embedding))

        if use_cache and new_cache_items:
            append_embeddings_to_cache(self.cache_path, new_cache_items)

        final_vectors = [embedding for embedding in ordered_embeddings if embedding is not None]
        if context == "index":
            LOGGER.info(
                "Index embedding cache summary: total_chunks=%s cache_hits=%s cache_misses=%s "
                "new_embeddings_written=%s batches=%s retries=%s batch_size=%s",
                len(requests),
                cache_hits,
                cache_misses,
                len(new_cache_items),
                self.total_batches,
                self.retry_count,
                self.batch_size,
            )
        else:
            LOGGER.info(
                "Query embedding summary: texts=%s batches=%s retries=%s model=%s",
                len(requests),
                self.total_batches,
                self.retry_count,
                self.model,
            )
        return np.asarray(final_vectors, dtype=np.float32)
