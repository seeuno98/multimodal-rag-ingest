from __future__ import annotations

import json
from pathlib import Path

from src.eval.run_eval import run_eval


def test_run_eval_uses_dense_only_when_bm25_missing(tmp_path, monkeypatch) -> None:
    eval_file = tmp_path / "questions.json"
    eval_file.write_text(
        json.dumps([{"query": "q1", "relevant_doc_ids": ["doc-1"]}]),
        encoding="utf-8",
    )
    index_path = tmp_path / "faiss.index"
    metadata_path = tmp_path / "metadata.jsonl"
    bm25_path = tmp_path / "bm25.joblib"
    results_path = tmp_path / "results.json"

    calls: list[str] = []

    def fake_retrieve(
        question: str,
        index_path: Path,
        metadata_path: Path,
        openai_api_key: str,
        embed_model: str,
        k: int = 5,
        mode: str | None = None,
        bm25_path: Path | None = None,
    ) -> list[dict[str, str]]:
        calls.append(str(mode))
        return [{"doc_id": "doc-1", "citation": "[1]"}]

    monkeypatch.setattr("src.eval.run_eval.retrieve", fake_retrieve)

    report = run_eval(
        eval_file=eval_file,
        index_path=index_path,
        metadata_path=metadata_path,
        openai_api_key="test-key",
        embed_model="test-model",
        k=5,
        bm25_path=bm25_path,
        results_path=results_path,
    )

    assert calls == ["dense"]
    assert list(report["summary"]) == ["dense"]
    assert "bm25" not in report["table"]
    assert json.loads(results_path.read_text(encoding="utf-8")) == report["summary"]


def test_run_eval_uses_all_modes_when_bm25_exists(tmp_path, monkeypatch) -> None:
    eval_file = tmp_path / "questions.json"
    eval_file.write_text(
        json.dumps([{"query": "q1", "relevant_doc_ids": ["doc-1"]}]),
        encoding="utf-8",
    )
    index_path = tmp_path / "faiss.index"
    metadata_path = tmp_path / "metadata.jsonl"
    bm25_path = tmp_path / "bm25.joblib"
    bm25_path.write_text("ready", encoding="utf-8")
    results_path = tmp_path / "results.json"

    calls: list[str] = []
    mode_results = {
        "dense": [{"doc_id": "doc-1", "citation": "[1]"}],
        "bm25": [{"doc_id": "doc-x", "citation": "[x]"}, {"doc_id": "doc-1", "citation": "[1]"}],
        "hybrid": [{"doc_id": "doc-1", "citation": "[1]"}],
    }

    def fake_retrieve(
        question: str,
        index_path: Path,
        metadata_path: Path,
        openai_api_key: str,
        embed_model: str,
        k: int = 5,
        mode: str | None = None,
        bm25_path: Path | None = None,
    ) -> list[dict[str, str]]:
        assert mode is not None
        calls.append(mode)
        return mode_results[mode]

    monkeypatch.setattr("src.eval.run_eval.retrieve", fake_retrieve)

    report = run_eval(
        eval_file=eval_file,
        index_path=index_path,
        metadata_path=metadata_path,
        openai_api_key="test-key",
        embed_model="test-model",
        k=5,
        bm25_path=bm25_path,
        results_path=results_path,
    )

    assert calls == ["dense", "bm25", "hybrid"]
    assert list(report["summary"]) == ["dense", "bm25", "hybrid"]
    assert report["summary"]["dense"] == {
        "recall@1": 1.0,
        "recall@5": 1.0,
        "recall@10": 1.0,
        "mrr": 1.0,
    }
    assert report["summary"]["bm25"] == {
        "recall@1": 0.0,
        "recall@5": 1.0,
        "recall@10": 1.0,
        "mrr": 0.5,
    }
    assert report["summary"]["hybrid"] == {
        "recall@1": 1.0,
        "recall@5": 1.0,
        "recall@10": 1.0,
        "mrr": 1.0,
    }
    assert "dense | 1.0000 | 1.0000 | 1.0000 | 1.0000" in report["table"]
    assert "bm25 | 0.0000 | 1.0000 | 1.0000 | 0.5000" in report["table"]
    assert "hybrid | 1.0000 | 1.0000 | 1.0000 | 1.0000" in report["table"]
    assert json.loads(results_path.read_text(encoding="utf-8")) == report["summary"]


def test_run_eval_deduplicates_doc_ids_for_metrics(tmp_path, monkeypatch) -> None:
    eval_file = tmp_path / "questions.json"
    eval_file.write_text(
        json.dumps([{"query": "q1", "relevant_doc_ids": ["doc-2"]}]),
        encoding="utf-8",
    )
    index_path = tmp_path / "faiss.index"
    metadata_path = tmp_path / "metadata.jsonl"
    results_path = tmp_path / "results.json"

    def fake_retrieve(
        question: str,
        index_path: Path,
        metadata_path: Path,
        openai_api_key: str,
        embed_model: str,
        k: int = 5,
        mode: str | None = None,
        bm25_path: Path | None = None,
    ) -> list[dict[str, str]]:
        return [
            {"doc_id": "doc-1", "citation": "[1]"},
            {"doc_id": "doc-1", "citation": "[1]"},
            {"doc_id": "doc-2", "citation": "[2]"},
        ]

    monkeypatch.setattr("src.eval.run_eval.retrieve", fake_retrieve)

    report = run_eval(
        eval_file=eval_file,
        index_path=index_path,
        metadata_path=metadata_path,
        openai_api_key="test-key",
        embed_model="test-model",
        k=5,
        bm25_path=tmp_path / "missing-bm25.joblib",
        results_path=results_path,
    )

    assert report["summary"]["dense"]["mrr"] == 0.5
