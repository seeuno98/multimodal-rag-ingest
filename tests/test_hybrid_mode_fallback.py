from __future__ import annotations

from src.rag.retrieve import select_retrieval_mode


def test_hybrid_mode_defaults_to_dense_when_bm25_missing(tmp_path) -> None:
    index_path = tmp_path / "faiss.index"
    metadata_path = tmp_path / "metadata.jsonl"
    bm25_path = tmp_path / "bm25.joblib"
    index_path.write_text("", encoding="utf-8")
    metadata_path.write_text("", encoding="utf-8")

    mode = select_retrieval_mode(index_path, metadata_path, bm25_path, requested_mode=None)

    assert mode == "dense"
