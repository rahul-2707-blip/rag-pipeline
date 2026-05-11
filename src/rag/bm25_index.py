"""BM25 sparse retrieval index. Built from the same chunks as the dense index,
persisted as a pickle alongside the database so both indexes stay in sync.
"""
from __future__ import annotations

import pickle
import re
from pathlib import Path
from typing import Optional

from rank_bm25 import BM25Okapi


INDEX_DIR = Path(__file__).resolve().parents[2] / "data"
_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


def _tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text)]


class BM25Index:
    """Wraps rank_bm25 with chunk-id mapping and persistence."""

    def __init__(self, chunk_ids: list[int], tokenized_docs: list[list[str]]):
        self.chunk_ids = chunk_ids
        self.tokenized = tokenized_docs
        self.bm25 = BM25Okapi(tokenized_docs) if tokenized_docs else None

    def search(self, query: str, k: int = 10) -> list[tuple[int, float]]:
        if self.bm25 is None:
            return []
        tokens = _tokenize(query)
        if not tokens:
            return []
        scores = self.bm25.get_scores(tokens)
        # argsort descending
        top = sorted(range(len(scores)), key=lambda i: -scores[i])[:k]
        return [(self.chunk_ids[i], float(scores[i])) for i in top if scores[i] > 0]

    def save(self, strategy: str) -> Path:
        INDEX_DIR.mkdir(parents=True, exist_ok=True)
        path = INDEX_DIR / f"bm25_{strategy}.pkl"
        with open(path, "wb") as f:
            pickle.dump({"chunk_ids": self.chunk_ids, "tokenized": self.tokenized}, f)
        return path

    @classmethod
    def load(cls, strategy: str) -> Optional["BM25Index"]:
        path = INDEX_DIR / f"bm25_{strategy}.pkl"
        if not path.exists():
            return None
        with open(path, "rb") as f:
            data = pickle.load(f)
        return cls(data["chunk_ids"], data["tokenized"])


def build_bm25(chunks: list[dict]) -> BM25Index:
    chunk_ids = [c["chunk_id"] for c in chunks]
    tokenized = [_tokenize(c["text"]) for c in chunks]
    return BM25Index(chunk_ids, tokenized)
