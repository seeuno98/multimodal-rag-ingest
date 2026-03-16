from __future__ import annotations

from fastapi.testclient import TestClient

from src.api.app import create_app


class FakeRetrievalService:
    def status(self) -> dict[str, bool]:
        return {"ok": True, "faiss_loaded": True, "bm25_loaded": True, "embedder_ready": True}

    def retrieve(self, query: str, k: int = 5, mode: str = "dense") -> list[dict]:
        return [
            {
                "rank": 1,
                "doc_id": "doc-1",
                "chunk_id": "chunk-1",
                "retrieval": mode,
                "citation": "[rss:doc-1]",
                "source_uri": "https://example.com/doc-1",
                "score": 0.9,
                "text": "retrieval augmented generation is a pattern for grounding answers",
            }
        ][:k]

    def answer(self, query: str, k: int = 5, mode: str = "dense") -> dict:
        return {
            "answer": "RAG combines retrieval and generation.",
            "citations": ["[rss:doc-1]"],
            "citation_urls": {"[rss:doc-1]": "https://example.com/doc-1"},
            "results": self.retrieve(query=query, k=k, mode=mode),
        }


def test_health_endpoint() -> None:
    app = create_app(service_factory=FakeRetrievalService)
    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["resources"]["faiss_loaded"] is True


def test_retrieve_and_answer_endpoints() -> None:
    app = create_app(service_factory=FakeRetrievalService)
    with TestClient(app) as client:
        retrieve_response = client.post("/retrieve", json={"query": "What is RAG?", "k": 1, "mode": "hybrid"})
        answer_response = client.post("/answer", json={"query": "What is RAG?", "k": 1, "mode": "hybrid"})

    assert retrieve_response.status_code == 200
    assert retrieve_response.json()["mode"] == "hybrid"
    assert retrieve_response.json()["results"][0]["doc_id"] == "doc-1"

    assert answer_response.status_code == 200
    assert answer_response.json()["citations"] == ["[rss:doc-1]"]
    assert "RAG combines retrieval" in answer_response.json()["answer"]
