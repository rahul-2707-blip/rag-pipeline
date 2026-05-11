"""Multi-format document loaders. Normalize everything to plaintext + metadata."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

from bs4 import BeautifulSoup
from PyPDF2 import PdfReader


@dataclass
class RawDocument:
    """A normalized document ready for chunking."""
    source: str            # e.g. "fastapi/tutorial/first-steps.md"
    title: str             # extracted from first heading or filename
    text: str              # cleaned plaintext
    metadata: dict = field(default_factory=dict)


_HEADING_PATTERNS = [
    re.compile(r"^#\s+(.+)$", re.MULTILINE),          # markdown H1
    re.compile(r"^##\s+(.+)$", re.MULTILINE),         # markdown H2
]


def _strip_markdown(md: str) -> str:
    # Remove code-fence markers but keep code content (useful for technical docs)
    md = re.sub(r"```[\w]*\n", "\n", md)
    md = re.sub(r"```", "", md)
    # Inline-code → keep content
    md = re.sub(r"`([^`]+)`", r"\1", md)
    # Bold/italic
    md = re.sub(r"\*\*([^*]+)\*\*", r"\1", md)
    md = re.sub(r"\*([^*]+)\*", r"\1", md)
    # Links: [text](url) → text
    md = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", md)
    # Images
    md = re.sub(r"!\[([^\]]*)\]\([^)]+\)", "", md)
    return md


def _extract_title(text: str, fallback: str) -> str:
    for pattern in _HEADING_PATTERNS:
        m = pattern.search(text)
        if m:
            return m.group(1).strip()
    return fallback


def load_markdown(path: Path, base: Path) -> RawDocument:
    raw = path.read_text(encoding="utf-8", errors="ignore")
    title = _extract_title(raw, path.stem)
    text = _strip_markdown(raw)
    return RawDocument(
        source=str(path.relative_to(base)),
        title=title,
        text=text,
        metadata={"format": "markdown", "char_count": len(text)},
    )


def load_text(path: Path, base: Path) -> RawDocument:
    text = path.read_text(encoding="utf-8", errors="ignore")
    return RawDocument(
        source=str(path.relative_to(base)),
        title=path.stem,
        text=text,
        metadata={"format": "text", "char_count": len(text)},
    )


def load_html(path: Path, base: Path) -> RawDocument:
    raw = path.read_text(encoding="utf-8", errors="ignore")
    soup = BeautifulSoup(raw, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()
    title_tag = soup.find("h1") or soup.find("title")
    title = title_tag.get_text(strip=True) if title_tag else path.stem
    text = soup.get_text("\n", strip=True)
    return RawDocument(
        source=str(path.relative_to(base)),
        title=title,
        text=text,
        metadata={"format": "html", "char_count": len(text)},
    )


def load_pdf(path: Path, base: Path) -> RawDocument:
    reader = PdfReader(str(path))
    pages: list[str] = []
    for i, page in enumerate(reader.pages):
        try:
            pages.append(page.extract_text() or "")
        except Exception:
            pages.append("")
    text = "\n\n".join(pages)
    return RawDocument(
        source=str(path.relative_to(base)),
        title=path.stem,
        text=text,
        metadata={"format": "pdf", "char_count": len(text), "page_count": len(reader.pages)},
    )


_LOADERS = {
    ".md": load_markdown,
    ".markdown": load_markdown,
    ".txt": load_text,
    ".html": load_html,
    ".htm": load_html,
    ".pdf": load_pdf,
}


def load_directory(root: Path) -> Iterator[RawDocument]:
    """Walk a directory and yield RawDocuments for every supported file."""
    root = Path(root)
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in _LOADERS:
            loader = _LOADERS[path.suffix.lower()]
            try:
                doc = loader(path, root)
                if doc.text.strip():
                    yield doc
            except Exception as e:
                print(f"  ! skip {path.relative_to(root)}: {e}")
