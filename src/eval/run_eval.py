from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from src.eval.metrics import mean, recall_at_k, reciprocal_rank
from src.rag.retrieve import retrieve_chunks

LOGGER = logging.getLogger(__name__)


def _metric_targets(row: dict[str, Any], results: list[dict[str, Any]]) -> set[str]:
    if row.get("relevant_citations"):
        return set(row["relevant_citations"])
    if row.get("relevant_doc_ids"):
        return set(row["relevant_doc_ids"])
    return set()


def _result_targets(row: dict[str, Any], results: list[dict[str, Any]]) -> list[str]:
    if row.get("relevant_citations"):
        return [str(result.get("citation")) for result in results]
    return [str(result.get("doc_id")) for result in results]


def _format_table(mode_metrics: dict[str, dict[str, float]]) -> str:
    header = "Mode | Recall@1 | Recall@5 | Recall@10 | MRR"
    lines = [header, "-" * len(header)]
    for mode, metrics in mode_metrics.items():
        lines.append(
            f"{mode} | {metrics['Recall@1']:.4f} | {metrics['Recall@5']:.4f} | "
            f"{metrics['Recall@10']:.4f} | {metrics['MRR']:.4f}"
        )
    return "\n".join(lines)


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

    modes = ["dense"]
    if bm25_path.exists():
        modes.extend(["bm25", "hybrid"])

    per_mode_scores: dict[str, dict[str, list[float]]] = {
        mode: {"Recall@1": [], "Recall@5": [], "Recall@10": [], "MRR": []} for mode in modes
    }
    per_query_details: list[dict[str, Any]] = []

    for row in questions:
        question = row.get("query") or row.get("question")
        if not question:
            continue

        query_detail: dict[str, Any] = {"query": question, "modes": {}}
        for mode in modes:
            results = retrieve_chunks(
                question=question,
                index_path=index_path,
                metadata_path=metadata_path,
                openai_api_key=openai_api_key,
                embed_model=embed_model,
                top_k=max(10, k),
                mode=mode,  # type: ignore[arg-type]
                bm25_path=bm25_path,
            )
            targets = _metric_targets(row, results)
            retrieved_targets = _result_targets(row, results)
            mode_scores = {
                "Recall@1": recall_at_k(targets, retrieved_targets, 1),
                "Recall@5": recall_at_k(targets, retrieved_targets, 5),
                "Recall@10": recall_at_k(targets, retrieved_targets, 10),
                "MRR": reciprocal_rank(targets, retrieved_targets),
            }
            for metric_name, value in mode_scores.items():
                per_mode_scores[mode][metric_name].append(value)
            query_detail["modes"][mode] = {
                "results": [
                    {
                        "doc_id": item.get("doc_id"),
                        "chunk_id": item.get("chunk_id"),
                        "citation": item.get("citation"),
                        "score": item.get("score"),
                        "retrieval": item.get("retrieval"),
                    }
                    for item in results[: max(10, k)]
                ],
                "metrics": mode_scores,
            }
        per_query_details.append(query_detail)

    mode_metrics = {
        mode: {metric: mean(values) for metric, values in metrics.items()}
        for mode, metrics in per_mode_scores.items()
    }
    table = _format_table(mode_metrics)

    results_path.parent.mkdir(parents=True, exist_ok=True)
    with results_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "summary": mode_metrics,
                "per_query": per_query_details,
            },
            f,
            indent=2,
        )

    LOGGER.info("Wrote evaluation report to %s", results_path)
    return {"table": table, "summary": mode_metrics, "results_path": str(results_path)}
