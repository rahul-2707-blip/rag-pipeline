"""Local sentence-transformers embedder. Free, fast, runs on CPU.

Default model: BAAI/bge-small-en-v1.5 (384-dim, ~33MB, MTEB top-tier for size).
"""
from __future__ import annotations

import os
from functools import lru_cache
from typing import Optional

import numpy as np
from sentence_transformers import SentenceTransformer


DEFAULT_MODEL = os.environ.get("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")


@lru_cache(maxsize=2)
def _model(name: str) -> SentenceTransformer:
    return SentenceTransformer(name)


def embed_texts(texts: list[str], model_name: Optional[str] = None, batch_size: int = 32) -> np.ndarray:
    """Return an (N, D) array of normalized embeddings for the given texts."""
    if not texts:
        return np.zeros((0, embedding_dim(model_name)))
    model = _model(model_name or DEFAULT_MODEL)
    vecs = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        normalize_embeddings=True,
    )
    return np.asarray(vecs, dtype=np.float32)


def embed_text(text: str, model_name: Optional[str] = None) -> np.ndarray:
    return embed_texts([text], model_name=model_name)[0]


def embedding_dim(model_name: Optional[str] = None) -> int:
    return _model(model_name or DEFAULT_MODEL).get_sentence_embedding_dimension()
