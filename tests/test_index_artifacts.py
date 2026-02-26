from __future__ import annotations

from pathlib import Path

import pytest


def test_faiss_index_artifact() -> None:
    index_path = Path("data/index/faiss.index")
    if not index_path.exists():
        pytest.skip(f"Missing artifact: {index_path}")

    assert index_path.stat().st_size > 0, "faiss.index must be non-empty"
