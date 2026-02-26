from __future__ import annotations

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

    try:
        print("[smoke] Step 1/5: normalize")
        run_step([sys.executable, "-m", "src.cli", "normalize"], root)

        print("[smoke] Step 2/5: validate docs artifact")
        assert_exists(docs_path, f"Missing required artifact: {docs_path}")

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
    except (subprocess.CalledProcessError, AssertionError) as exc:
        print(f"[smoke] FAILED: {exc}")
        return 1

    print("Smoke test passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
