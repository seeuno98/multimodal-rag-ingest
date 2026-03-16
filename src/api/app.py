from __future__ import annotations

import logging
import time
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from src.api.service import RETRIEVAL_MODES, RetrievalService, format_retrieval_result
from src.config import get_config

LOGGER = logging.getLogger(__name__)


class HealthResponse(BaseModel):
    status: str
    resources: dict[str, Any]


class RetrievalRequest(BaseModel):
    query: str = Field(..., min_length=1)
    k: int = Field(default=5, ge=1, le=20)
    mode: Literal["dense", "bm25", "hybrid"] = "dense"


class RetrievalResult(BaseModel):
    rank: int
    doc_id: str
    chunk_id: str | None = None
    retrieval: str | None = None
    citation: str | None = None
    source_uri: str | None = None
    score: float | None = None
    snippet: str
    components: dict[str, int | None] | None = None


class RetrievalResponse(BaseModel):
    query: str
    mode: Literal["dense", "bm25", "hybrid"]
    top_k: int
    results: list[RetrievalResult]


class AnswerResponse(BaseModel):
    query: str
    mode: Literal["dense", "bm25", "hybrid"]
    top_k: int
    answer: str
    citations: list[str]
    citation_urls: dict[str, str]
    results: list[RetrievalResult]


def create_app(
    service_factory: Callable[[], RetrievalService] | None = None,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        factory = service_factory or (lambda: RetrievalService(get_config()))
        app.state.retrieval_service = factory()
        LOGGER.info(
            "API startup complete modes=%s status=%s",
            ",".join(RETRIEVAL_MODES),
            app.state.retrieval_service.status(),
        )
        yield

    app = FastAPI(title="multimodal-rag-ingest", version="0.1.0", lifespan=lifespan)

    @app.middleware("http")
    async def log_requests(request: Request, call_next):  # type: ignore[no-untyped-def]
        started = time.perf_counter()
        response = await call_next(request)
        elapsed_ms = (time.perf_counter() - started) * 1000
        LOGGER.info(
            "request method=%s path=%s status_code=%s latency_ms=%.2f",
            request.method,
            request.url.path,
            response.status_code,
            elapsed_ms,
        )
        response.headers["X-Process-Time-Ms"] = f"{elapsed_ms:.2f}"
        return response

    @app.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        service: RetrievalService = app.state.retrieval_service
        return HealthResponse(status="ok", resources=service.status())

    @app.post("/retrieve", response_model=RetrievalResponse)
    async def retrieve(request: RetrievalRequest) -> RetrievalResponse:
        service: RetrievalService = app.state.retrieval_service
        try:
            results = service.retrieve(query=request.query, k=request.k, mode=request.mode)
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        payload = [RetrievalResult(**format_retrieval_result(result)) for result in results]
        return RetrievalResponse(
            query=request.query,
            mode=request.mode,
            top_k=request.k,
            results=payload,
        )

    @app.post("/answer", response_model=AnswerResponse)
    async def answer(request: RetrievalRequest) -> AnswerResponse:
        service: RetrievalService = app.state.retrieval_service
        try:
            response = service.answer(query=request.query, k=request.k, mode=request.mode)
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        payload = [RetrievalResult(**format_retrieval_result(result)) for result in response["results"]]
        return AnswerResponse(
            query=request.query,
            mode=request.mode,
            top_k=request.k,
            answer=response["answer"],
            citations=response["citations"],
            citation_urls=response["citation_urls"],
            results=payload,
        )

    return app


app = create_app()
