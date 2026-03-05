from __future__ import annotations

from src.eval.metrics import recall_at_k, reciprocal_rank


def test_recall_at_k_simple() -> None:
    relevant = {"doc-a", "doc-b"}
    retrieved = ["doc-x", "doc-a", "doc-y", "doc-b"]

    assert recall_at_k(relevant, retrieved, 1) == 0.0
    assert recall_at_k(relevant, retrieved, 5) == 1.0


def test_reciprocal_rank_simple() -> None:
    relevant = {"doc-b"}
    retrieved = ["doc-x", "doc-b", "doc-y"]

    assert reciprocal_rank(relevant, retrieved) == 0.5
