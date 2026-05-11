"""Fetch the FastAPI docs (English) into corpus/fastapi/.

Uses a shallow git clone, copies the docs, then deletes the clone.
Saves ~MB by only keeping markdown files we care about.
"""
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


REPO = "https://github.com/fastapi/fastapi.git"
DOCS_SUBPATH = "docs/en/docs"
OUTPUT = Path(__file__).resolve().parents[1] / "corpus" / "fastapi"


def main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    print(f"Cloning {REPO} (shallow) into temp dir…")
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        subprocess.run(
            ["git", "clone", "--depth", "1", "--filter=blob:none", REPO, str(tmp_path / "fastapi")],
            check=True,
        )
        src = tmp_path / "fastapi" / DOCS_SUBPATH
        if not src.exists():
            print(f"!! docs path not found: {src}", file=sys.stderr)
            return 1
        copied = 0
        for md in src.rglob("*.md"):
            rel = md.relative_to(src)
            dest = OUTPUT / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(md, dest)
            copied += 1
        print(f"Copied {copied} markdown files into {OUTPUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
