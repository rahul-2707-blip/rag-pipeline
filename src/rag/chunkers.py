"""Three chunking strategies. Each chunk records WHICH strategy produced it
so we can compare strategies in the eval phase.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from .loaders import RawDocument


@dataclass
class Chunk:
    source: str
    title: str
    section: Optional[str]
    text: str
    chunk_index: int
    strategy: str
    metadata: dict = field(default_factory=dict)


# ---------- Strategy 1: fixed-size with overlap (baseline) ----------

def chunk_fixed(doc: RawDocument, size: int = 800, overlap: int = 120) -> list[Chunk]:
    """Slide a window of `size` chars with `overlap` over the text."""
    text = doc.text
    if not text:
        return []
    step = max(1, size - overlap)
    chunks: list[Chunk] = []
    i = 0
    idx = 0
    while i < len(text):
        piece = text[i : i + size].strip()
        if piece:
            chunks.append(Chunk(
                source=doc.source,
                title=doc.title,
                section=None,
                text=piece,
                chunk_index=idx,
                strategy="fixed",
                metadata={"size": size, "overlap": overlap},
            ))
            idx += 1
        i += step
    return chunks


# ---------- Strategy 2: recursive split by section headers ----------

_HEADER_RE = re.compile(r"(^|\n)(#{1,6})\s+(.+)\n", re.MULTILINE)


def chunk_recursive(doc: RawDocument, max_chars: int = 1200, min_chars: int = 200) -> list[Chunk]:
    """Split on markdown headers (or two-newline breaks if no headers) and pack sections
    up to max_chars, falling back to fixed-size split for any section that's too long.
    """
    text = doc.text
    if not text:
        return []

    # Find sections by header positions
    matches = list(_HEADER_RE.finditer(text))
    sections: list[tuple[str, str]] = []  # (section_title, body)
    if matches:
        for i, m in enumerate(matches):
            section_title = m.group(3).strip()
            body_start = m.end()
            body_end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            body = text[body_start:body_end].strip()
            if body:
                sections.append((section_title, body))
        # Preamble before first header
        if matches[0].start() > 0:
            preamble = text[: matches[0].start()].strip()
            if preamble:
                sections.insert(0, (None, preamble))
    else:
        # No headers — split on paragraph breaks
        for para in re.split(r"\n\s*\n", text):
            para = para.strip()
            if para:
                sections.append((None, para))

    # Pack sections; fall back to fixed split for oversize ones
    chunks: list[Chunk] = []
    idx = 0
    buf = ""
    buf_section = None
    for section_title, body in sections:
        if len(body) > max_chars:
            # Flush buffer first
            if buf:
                chunks.append(_mk(doc, buf_section, buf, idx, "recursive"))
                idx += 1
                buf = ""
            # Split body with fixed window
            for j in range(0, len(body), max_chars - 100):
                piece = body[j : j + max_chars].strip()
                if piece:
                    chunks.append(_mk(doc, section_title, piece, idx, "recursive"))
                    idx += 1
            continue
        if len(buf) + len(body) + 2 <= max_chars:
            buf = (buf + "\n\n" + body) if buf else body
            buf_section = buf_section or section_title
        else:
            if buf:
                chunks.append(_mk(doc, buf_section, buf, idx, "recursive"))
                idx += 1
            buf = body
            buf_section = section_title

    if buf and len(buf) >= min_chars:
        chunks.append(_mk(doc, buf_section, buf, idx, "recursive"))
    elif buf and chunks:
        # Append small trailing piece to last chunk
        chunks[-1].text += "\n\n" + buf

    return chunks


def _mk(doc: RawDocument, section: Optional[str], text: str, idx: int, strategy: str) -> Chunk:
    return Chunk(
        source=doc.source,
        title=doc.title,
        section=section,
        text=text,
        chunk_index=idx,
        strategy=strategy,
        metadata={"char_count": len(text)},
    )


# ---------- Strategy 3: semantic chunking on topic boundaries ----------

def chunk_semantic(
    doc: RawDocument,
    embed_fn: Callable[[list[str]], np.ndarray],
    similarity_threshold: float = 0.55,
    min_chars: int = 200,
    max_chars: int = 1500,
) -> list[Chunk]:
    """Split text into sentences, embed each, then merge adjacent sentences while
    similarity stays high. Start a new chunk when adjacent-sentence similarity
    drops below threshold (topic boundary detected).
    """
    text = doc.text
    if not text:
        return []

    sentences = _split_sentences(text)
    if not sentences:
        return []
    if len(sentences) == 1:
        return [_mk(doc, None, sentences[0], 0, "semantic")]

    embeddings = embed_fn(sentences)
    # Normalize for cosine = dot product
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / np.where(norms == 0, 1, norms)

    chunks: list[Chunk] = []
    buf = sentences[0]
    idx = 0
    for i in range(1, len(sentences)):
        sim = float(embeddings[i - 1] @ embeddings[i])
        candidate = (buf + " " + sentences[i]) if buf else sentences[i]
        if sim < similarity_threshold or len(candidate) > max_chars:
            if buf and len(buf) >= min_chars:
                chunks.append(_mk(doc, None, buf, idx, "semantic"))
                idx += 1
                buf = sentences[i]
            else:
                buf = candidate  # too short to flush yet
        else:
            buf = candidate

    if buf:
        if chunks and len(buf) < min_chars:
            chunks[-1].text += " " + buf
        else:
            chunks.append(_mk(doc, None, buf, idx, "semantic"))

    return chunks


_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")


def _split_sentences(text: str) -> list[str]:
    sentences = []
    for paragraph in re.split(r"\n\s*\n", text):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        for sentence in _SENTENCE_RE.split(paragraph):
            sentence = sentence.strip()
            if sentence:
                sentences.append(sentence)
    return sentences


# ---------- Dispatcher ----------

STRATEGIES = {"fixed", "recursive", "semantic"}


def chunk_doc(doc: RawDocument, strategy: str, embed_fn: Optional[Callable] = None) -> list[Chunk]:
    if strategy == "fixed":
        return chunk_fixed(doc)
    if strategy == "recursive":
        return chunk_recursive(doc)
    if strategy == "semantic":
        if embed_fn is None:
            raise ValueError("semantic chunking requires embed_fn")
        return chunk_semantic(doc, embed_fn)
    raise ValueError(f"unknown strategy: {strategy}")
