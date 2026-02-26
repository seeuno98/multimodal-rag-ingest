from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import faiss
import numpy as np


class FaissStore:
    def __init__(self, dim: int) -> None:
        self.index = faiss.IndexFlatIP(dim)
        self.metadata: list[dict[str, Any]] = []

    @staticmethod
    def _normalize(vectors: np.ndarray) -> np.ndarray:
        if vectors.size == 0:
            return vectors
        norms = np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-12
        return vectors / norms

    def add(self, vectors: np.ndarray, metadata: list[dict[str, Any]]) -> None:
        vectors = self._normalize(vectors.astype(np.float32))
        self.index.add(vectors)
        self.metadata.extend(metadata)

    def search(self, query_vector: np.ndarray, top_k: int) -> list[dict[str, Any]]:
        query = self._normalize(query_vector.reshape(1, -1).astype(np.float32))
        scores, indices = self.index.search(query, top_k)
        results: list[dict[str, Any]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.metadata):
                continue
            result = dict(self.metadata[idx])
            result["score"] = float(score)
            results.append(result)
        return results

    def save(self, index_path: Path, metadata_path: Path) -> None:
        index_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(index_path))
        with metadata_path.open("w", encoding="utf-8") as f:
            for item in self.metadata:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

    @classmethod
    def load(cls, index_path: Path, metadata_path: Path) -> "FaissStore":
        index = faiss.read_index(str(index_path))
        store = cls(dim=index.d)
        store.index = index
        with metadata_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    store.metadata.append(json.loads(line))
        return store
