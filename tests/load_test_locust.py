from __future__ import annotations

import os

from locust import HttpUser, between, task

QUERY = os.getenv("RETRIEVAL_QUERY", "What is retrieval augmented generation?")
TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "5"))
MODE = os.getenv("RETRIEVAL_MODE", "hybrid")


class RetrievalUser(HttpUser):
    wait_time = between(0.1, 0.5)

    @task
    def retrieve(self) -> None:
        self.client.post(
            "/retrieve",
            json={
                "query": QUERY,
                "k": TOP_K,
                "mode": MODE,
            },
            name=f"/retrieve {MODE}",
        )
