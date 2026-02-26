from __future__ import annotations

import logging
import time
from typing import Sequence

import numpy as np
from openai import OpenAI

LOGGER = logging.getLogger(__name__)


class Embedder:
    def __init__(self, api_key: str, model: str, batch_size: int = 100) -> None:
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.batch_size = batch_size

    def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, 0), dtype=np.float32)

        vectors: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = list(texts[i : i + self.batch_size])
            retries = 3
            for attempt in range(retries):
                try:
                    response = self.client.embeddings.create(model=self.model, input=batch)
                    vectors.extend([item.embedding for item in response.data])
                    break
                except Exception as exc:
                    if attempt == retries - 1:
                        raise
                    sleep_s = 2**attempt
                    LOGGER.warning("Embedding batch failed (%s). Retrying in %ss", exc, sleep_s)
                    time.sleep(sleep_s)
        return np.asarray(vectors, dtype=np.float32)
