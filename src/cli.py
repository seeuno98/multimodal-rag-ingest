from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from src.config import get_config
from src.eval.run_eval import run_eval
from src.index.build_index import build_index
from src.ingest.arxiv import ingest_arxiv
from src.ingest.normalize import normalize_documents
from src.ingest.rss import ingest_rss
from src.ingest.youtube import ingest_youtube
from src.rag.answer import generate_grounded_answer
from src.rag.retrieve import retrieve_chunks


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="multimodal-rag-ingest")
    sub = parser.add_subparsers(dest="command", required=True)

    ingest_parser = sub.add_parser("ingest")
    ingest_sub = ingest_parser.add_subparsers(dest="ingest_type", required=True)

    arxiv_parser = ingest_sub.add_parser("arxiv")
    arxiv_parser.add_argument("--query", required=True)
    arxiv_parser.add_argument("--max", type=int, default=10)

    rss_parser = ingest_sub.add_parser("rss")
    rss_parser.add_argument("--feeds", type=Path, required=True)

    yt_parser = ingest_sub.add_parser("youtube")
    yt_parser.add_argument("--urls", type=Path, required=True)

    sub.add_parser("normalize")
    sub.add_parser("index")

    query_parser = sub.add_parser("query")
    query_parser.add_argument("--q", required=True)
    query_parser.add_argument("--k", type=int, default=None)

    eval_parser = sub.add_parser("eval")
    eval_parser.add_argument("--file", type=Path, required=True)
    eval_parser.add_argument("--k", type=int, default=None)

    return parser


def main() -> None:
    configure_logging()
    args = build_parser().parse_args()
    cfg = get_config()

    docs_path = cfg.processed_dir / "docs.jsonl"
    chunks_path = cfg.processed_dir / "chunks.jsonl"
    index_path = cfg.index_dir / "faiss.index"
    metadata_path = cfg.index_dir / "metadata.jsonl"

    if args.command == "ingest":
        if args.ingest_type == "arxiv":
            count = ingest_arxiv(query=args.query, max_results=args.max, raw_dir=cfg.raw_dir)
            print(f"Ingested arXiv docs: {count}")
            return
        if args.ingest_type == "rss":
            count = ingest_rss(feeds_file=args.feeds, raw_dir=cfg.raw_dir)
            print(f"Ingested RSS docs: {count}")
            return
        if args.ingest_type == "youtube":
            count = ingest_youtube(
                urls_file=args.urls,
                raw_dir=cfg.raw_dir,
                openai_api_key=cfg.openai_api_key,
            )
            print(f"Ingested YouTube docs: {count}")
            return

    if args.command == "normalize":
        count = normalize_documents(raw_dir=cfg.raw_dir, output_path=docs_path)
        print(f"Normalized docs: {count}")
        return

    if args.command == "index":
        doc_count, chunk_count = build_index(
            docs_path=docs_path,
            chunks_path=chunks_path,
            faiss_path=index_path,
            metadata_path=metadata_path,
            openai_api_key=cfg.openai_api_key,
            embed_model=cfg.openai_embed_model,
            chunk_max_chars=cfg.chunk_max_chars,
            chunk_overlap_chars=cfg.chunk_overlap_chars,
        )
        print(f"Indexed documents={doc_count}, chunks={chunk_count}")
        return

    if args.command == "query":
        top_k = args.k or cfg.top_k
        hits = retrieve_chunks(
            question=args.q,
            index_path=index_path,
            metadata_path=metadata_path,
            openai_api_key=cfg.openai_api_key,
            embed_model=cfg.openai_embed_model,
            top_k=top_k,
        )
        answer = generate_grounded_answer(
            question=args.q,
            retrieved_chunks=hits,
            openai_api_key=cfg.openai_api_key,
            chat_model=cfg.openai_chat_model,
        )
        print(answer)
        print("\nTop Retrieved Chunks:")
        for i, hit in enumerate(hits, start=1):
            print(
                json.dumps(
                    {
                        "rank": i,
                        "score": round(hit["score"], 4),
                        "doc_id": hit["doc_id"],
                        "citation": hit.get("citation") or hit.get("metadata", {}).get("citation"),
                        "source_uri": hit.get("source_uri") or hit.get("metadata", {}).get("source_uri"),
                    },
                    ensure_ascii=False,
                )
            )
        return

    if args.command == "eval":
        top_k = args.k or cfg.top_k
        report = run_eval(
            eval_file=args.file,
            index_path=index_path,
            metadata_path=metadata_path,
            openai_api_key=cfg.openai_api_key,
            embed_model=cfg.openai_embed_model,
            k=top_k,
        )
        print(json.dumps(report, indent=2))
        return


if __name__ == "__main__":
    main()
