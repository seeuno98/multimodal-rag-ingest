from __future__ import annotations

import json
import logging
from pathlib import Path

from src.eval.metrics import mean, recall_at_k, reciprocal_rank
from src.rag.retrieve import retrieve_chunks

LOGGER = logging.getLogger(__name__)


def run_eval(
    eval_file: Path,
    index_path: Path,
    metadata_path: Path,
    openai_api_key: str,
    embed_model: str,
    k: int,
) -> dict[str, float]:
    with eval_file.open("r", encoding="utf-8") as f:
        questions = json.load(f)

    recalls: list[float] = []
    mrrs: list[float] = []
    for row in questions:
        question = row["question"]
        relevant = set(row.get("relevant_doc_ids", []))
        results = retrieve_chunks(
            question=question,
            index_path=index_path,
            metadata_path=metadata_path,
            openai_api_key=openai_api_key,
            embed_model=embed_model,
            top_k=k,
        )
        retrieved_doc_ids = [r["doc_id"] for r in results]
        recalls.append(recall_at_k(relevant_ids=relevant, retrieved_ids=retrieved_doc_ids, k=k))
        mrrs.append(reciprocal_rank(relevant_ids=relevant, retrieved_ids=retrieved_doc_ids))

    report = {"Recall@K": mean(recalls), "MRR": mean(mrrs)}
    LOGGER.info("Evaluation on %s questions: Recall@%s=%.4f MRR=%.4f", len(questions), k, report["Recall@K"], report["MRR"])
    return report
