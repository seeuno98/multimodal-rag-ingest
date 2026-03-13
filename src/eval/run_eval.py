from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from src.eval.metrics import mean, recall_at_k, reciprocal_rank
from src.rag.retrieve import retrieve

LOGGER = logging.getLogger(__name__)
MODE_ORDER = ["dense", "bm25", "hybrid"]


def _metric_targets(row: dict[str, Any]) -> set[str]:
    if row.get("relevant_citations"):
        return set(row["relevant_citations"])
    if row.get("relevant_doc_ids"):
        return set(row["relevant_doc_ids"])
    return set()


def _dedupe_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        deduped.append(value)
    return deduped


def _result_targets(row: dict[str, Any], results: list[dict[str, Any]]) -> list[str]:
    if row.get("relevant_citations"):
        values = [str(result.get("citation")) for result in results]
    else:
        values = [str(result.get("doc_id")) for result in results]
    return _dedupe_preserve_order(values)


def _format_table(mode_metrics: dict[str, dict[str, float]]) -> str:
    header = "Mode | Recall@1 | Recall@5 | Recall@10 | MRR"
    lines = [header, "-" * len(header)]
    for mode in MODE_ORDER:
        if mode not in mode_metrics:
            continue
        metrics = mode_metrics[mode]
        lines.append(
            f"{mode} | {metrics['recall@1']:.4f} | {metrics['recall@5']:.4f} | "
            f"{metrics['recall@10']:.4f} | {metrics['mrr']:.4f}"
        )
    return "\n".join(lines)


def _evaluate_mode(
    mode: str,
    questions: list[dict[str, Any]],
    index_path: Path,
    metadata_path: Path,
    openai_api_key: str,
    embed_model: str,
    k: int,
    bm25_path: Path,
) -> dict[str, float]:
    LOGGER.info("Running eval mode: %s", mode)
    scores = {"recall@1": [], "recall@5": [], "recall@10": [], "mrr": []}

    for row in questions:
        question = row.get("query") or row.get("question")
        if not question:
            continue

        results = retrieve(
            question=question,
            index_path=index_path,
            metadata_path=metadata_path,
            openai_api_key=openai_api_key,
            embed_model=embed_model,
            k=max(10, k),
            mode=mode,  # type: ignore[arg-type]
            bm25_path=bm25_path,
        )
        targets = _metric_targets(row)
        retrieved_targets = _result_targets(row, results)
        scores["recall@1"].append(recall_at_k(targets, retrieved_targets, 1))
        scores["recall@5"].append(recall_at_k(targets, retrieved_targets, 5))
        scores["recall@10"].append(recall_at_k(targets, retrieved_targets, 10))
        scores["mrr"].append(reciprocal_rank(targets, retrieved_targets))

    return {metric: mean(values) for metric, values in scores.items()}


def run_eval(
    eval_file: Path,
    index_path: Path,
    metadata_path: Path,
    openai_api_key: str,
    embed_model: str,
    k: int,
    bm25_path: Path,
    results_path: Path,
) -> dict[str, Any]:
    with eval_file.open("r", encoding="utf-8") as f:
        questions = json.load(f)

    bm25_exists = bm25_path.exists()
    modes = ["dense", "bm25", "hybrid"] if bm25_exists else ["dense"]

    mode_metrics: dict[str, dict[str, float]] = {}
    for mode in modes:
        mode_metrics[mode] = _evaluate_mode(
            mode=mode,
            questions=questions,
            index_path=index_path,
            metadata_path=metadata_path,
            openai_api_key=openai_api_key,
            embed_model=embed_model,
            k=k,
            bm25_path=bm25_path,
        )

    table = _format_table(mode_metrics)

    results_path.parent.mkdir(parents=True, exist_ok=True)
    with results_path.open("w", encoding="utf-8") as f:
        json.dump(mode_metrics, f, indent=2)

    LOGGER.info("Wrote evaluation report to %s", results_path)
    return {"table": table, "summary": mode_metrics, "results_path": str(results_path)}

