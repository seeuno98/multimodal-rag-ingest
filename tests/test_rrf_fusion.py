from __future__ import annotations

from src.rag.retrieve import fuse_ranked_results


def test_rrf_fusion_orders_expected_chunks() -> None:
    dense = [
        {"chunk_id": "a", "doc_id": "a", "citation": "[rss:a]", "source_uri": "u1", "text": "a", "retrieval": "dense"},
        {"chunk_id": "b", "doc_id": "b", "citation": "[rss:b]", "source_uri": "u2", "text": "b", "retrieval": "dense"},
        {"chunk_id": "c", "doc_id": "c", "citation": "[rss:c]", "source_uri": "u3", "text": "c", "retrieval": "dense"},
    ]
    bm25 = [
        {"chunk_id": "b", "doc_id": "b", "citation": "[rss:b]", "source_uri": "u2", "text": "b", "retrieval": "bm25"},
        {"chunk_id": "c", "doc_id": "c", "citation": "[rss:c]", "source_uri": "u3", "text": "c", "retrieval": "bm25"},
        {"chunk_id": "a", "doc_id": "a", "citation": "[rss:a]", "source_uri": "u1", "text": "a", "retrieval": "bm25"},
    ]

    fused = fuse_ranked_results(dense, bm25, top_k=3, rrf_k=60)

    assert [item["chunk_id"] for item in fused] == ["b", "a", "c"]
    assert fused[0]["score"] >= fused[1]["score"] >= fused[2]["score"]
