from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def run_step(args: list[str], root: Path) -> None:
    print(f"[smoke] Running: {' '.join(args)}")
    subprocess.run(args, cwd=root, check=True)


def assert_exists(path: Path, message: str) -> None:
    if not path.exists():
        raise AssertionError(message)


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    docs_path = root / "data/processed/docs.jsonl"
    faiss_path = root / "data/index/faiss.index"
    smoke_embed = os.getenv("SMOKE_EMBED", "0")
    smoke_query = os.getenv("SMOKE_QUERY", "1")

    try:
        print("[smoke] Step 1/2: normalize")
        run_step([sys.executable, "-m", "src.cli", "normalize"], root)

        print("[smoke] Step 2/2: validate docs artifact")
        assert_exists(docs_path, f"Missing required artifact: {docs_path}")

        if smoke_embed == "1":
            print("[smoke] Step 3/5: index")
            run_step([sys.executable, "-m", "src.cli", "index"], root)

            print("[smoke] Step 4/5: validate index artifact")
            assert_exists(faiss_path, f"Missing required artifact: {faiss_path}")

            print("[smoke] Step 5/5: query")
            run_step(
                [
                    sys.executable,
                    "-m",
                    "src.cli",
                    "query",
                    "--q",
                    "What is retrieval augmented generation?",
                    "--k",
                    "5",
                ],
                root,
            )
            print("Smoke test passed")
            return 0

        if smoke_query != "1":
            print("Skipping query (SMOKE_QUERY=0).")
            print("Smoke test passed")
            return 0

        if not faiss_path.exists():
            print(
                "Index not found; skipping query to avoid OpenAI costs. "
                "Run make smoke_paid to build index."
            )
            print("Smoke test passed")
            return 0

        print("[smoke] Step 3/3: query")
        run_step(
            [
                sys.executable,
                "-m",
                "src.cli",
                "query",
                "--q",
                "What is retrieval augmented generation?",
                "--k",
                "5",
            ],
            root,
        )
    except (subprocess.CalledProcessError, AssertionError) as exc:
        print(f"[smoke] FAILED: {exc}")
        return 1

    print("Smoke test passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
