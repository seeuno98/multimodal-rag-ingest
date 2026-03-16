from __future__ import annotations

import argparse
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import requests


def percentile(sorted_values: list[float], pct: float) -> float:
    if not sorted_values:
        return 0.0
    if pct <= 0:
        return sorted_values[0]
    if pct >= 100:
        return sorted_values[-1]
    index = round((pct / 100.0) * (len(sorted_values) - 1))
    return sorted_values[index]


def run_request(session: requests.Session, url: str, payload: dict[str, Any], timeout_s: float) -> float:
    started = time.perf_counter()
    response = session.post(url, json=payload, timeout=timeout_s)
    response.raise_for_status()
    response.json()
    return (time.perf_counter() - started) * 1000


def main() -> None:
    parser = argparse.ArgumentParser(description="Simple latency benchmark for the retrieval API.")
    parser.add_argument("--url", default="http://127.0.0.1:8000/retrieve")
    parser.add_argument("--query", required=True)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--mode", default="dense", choices=["dense", "bm25", "hybrid"])
    parser.add_argument("--requests", type=int, default=25)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=30.0)
    args = parser.parse_args()

    payload = {"query": args.query, "k": args.k, "mode": args.mode}
    with requests.Session() as session:
        for _ in range(args.warmup):
            run_request(session=session, url=args.url, payload=payload, timeout_s=args.timeout)

        latencies_ms: list[float] = []
        started = time.perf_counter()
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            futures = [
                pool.submit(run_request, session, args.url, payload, args.timeout)
                for _ in range(args.requests)
            ]
            for future in as_completed(futures):
                latencies_ms.append(future.result())
        elapsed_s = time.perf_counter() - started

    latencies_ms.sort()
    throughput = len(latencies_ms) / elapsed_s if elapsed_s > 0 else 0.0
    print(f"requests={len(latencies_ms)} concurrency={args.concurrency} mode={args.mode}")
    print(f"avg_ms={statistics.mean(latencies_ms):.2f}")
    print(f"p50_ms={percentile(latencies_ms, 50):.2f}")
    print(f"p95_ms={percentile(latencies_ms, 95):.2f}")
    print(f"max_ms={latencies_ms[-1]:.2f}")
    print(f"throughput_rps={throughput:.2f}")


if __name__ == "__main__":
    main()
