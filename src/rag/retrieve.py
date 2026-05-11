"""Hybrid retrieval: dense + sparse + Reciprocal Rank Fusion + optional reranker.

The interview talking points live here. Read this file top-to-bottom before
demoing.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Optional

import psycopg

from .bm25_index import BM25Index
from .embed import embed_text
from .store import _connect, dense_search


@dataclass
class RetrievedChunk:
    chunk_id: int
    source: str
    title: str
    section: Optional[str]
    text: str
    chunk_index: int
    dense_score: float = 0.0
    sparse_score: float = 0.0
    rrf_score: float = 0.0
    rerank_score: float = 0.0


# ---------- Reciprocal Rank Fusion ----------

def reciprocal_rank_fusion(
    dense_hits: list[dict],
    sparse_hits: list[tuple[int, float]],
    dense_weight: float = 0.7,
    sparse_weight: float = 0.3,
    rrf_k: int = 60,
) -> list[tuple[int, float]]:
    """Combine two ranked lists into one.

    RRF score for chunk c = sum over each list L of weight_L * 1 / (k + rank_L(c)).
    The +k smooths the denominator and prevents the #1 result from dominating.
    """
    scores: dict[int, float] = {}

    for rank, hit in enumerate(dense_hits, start=1):
        scores[hit["chunk_id"]] = scores.get(hit["chunk_id"], 0.0) + dense_weight / (rrf_k + rank)

    for rank, (chunk_id, _) in enumerate(sparse_hits, start=1):
        scores[chunk_id] = scores.get(chunk_id, 0.0) + sparse_weight / (rrf_k + rank)

    fused = sorted(scores.items(), key=lambda x: -x[1])
    return fused


# ---------- Cross-encoder reranker ----------

@lru_cache(maxsize=1)
def _reranker():
    """Lazy-load the cross-encoder. ~80MB, runs on CPU."""
    from sentence_transformers import CrossEncoder

    model_name = os.environ.get("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    return CrossEncoder(model_name)


def rerank(query: str, candidates: list[RetrievedChunk], top_n: int) -> list[RetrievedChunk]:
    """Score each candidate's relevance to the query with a cross-encoder.

    Cross-encoders see the query+passage together (not as separate embeddings),
    so they're far more accurate than dense retrieval — but too slow to run
    over the full corpus. Used here as a precision step on the top ~20.
    """
    if not candidates:
        return []
    pairs = [(query, c.text) for c in candidates]
    scores = _reranker().predict(pairs)
    for c, s in zip(candidates, scores):
        c.rerank_score = float(s)
    candidates.sort(key=lambda c: -c.rerank_score)
    return candidates[:top_n]


# ---------- Fetch helper ----------

def _fetch_chunks(chunk_ids: list[int]) -> dict[int, dict]:
    if not chunk_ids:
        return {}
    with _connect() as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT c.id, d.source, d.title, c.section, c.text, c.chunk_index
            FROM chunks c JOIN documents d ON d.id = c.document_id
            WHERE c.id = ANY(%s)
            """,
            (chunk_ids,),
        )
        rows = cur.fetchall()
    return {
        r[0]: {
            "source": r[1],
            "title": r[2],
            "section": r[3],
            "text": r[4],
            "chunk_index": r[5],
        }
        for r in rows
    }


# ---------- Public API ----------

@dataclass
class RetrievalConfig:
    strategy: str = "recursive"
    dense_k: int = 10
    sparse_k: int = 10
    fusion_pool: int = 20
    final_k: int = 5
    dense_weight: float = 0.7
    sparse_weight: float = 0.3
    use_rerank: bool = True
    mode: str = "hybrid"  # "hybrid" | "dense" | "sparse"


def retrieve(query: str, config: RetrievalConfig) -> list[RetrievedChunk]:
    """Run the full retrieval pipeline and return the final top-k chunks."""
    # Dense
    query_emb = embed_text(query)
    dense_hits = dense_search(query_emb, strategy=config.strategy, k=config.dense_k) if config.mode != "sparse" else []
    dense_score_map = {h["chunk_id"]: h["score"] for h in dense_hits}

    # Sparse
    sparse_hits: list[tuple[int, float]] = []
    if config.mode != "dense":
        bm25 = BM25Index.load(config.strategy)
        if bm25 is not None:
            sparse_hits = bm25.search(query, k=config.sparse_k)
    sparse_score_map = {cid: s for cid, s in sparse_hits}

    # Fusion (or single-mode passthrough)
    if config.mode == "hybrid":
        fused = reciprocal_rank_fusion(
            dense_hits, sparse_hits, config.dense_weight, config.sparse_weight
        )
        candidate_ids = [cid for cid, _ in fused[: config.fusion_pool]]
        rrf_score_map = dict(fused)
    elif config.mode == "dense":
        candidate_ids = [h["chunk_id"] for h in dense_hits[: config.fusion_pool]]
        rrf_score_map = {}
    else:  # sparse
        candidate_ids = [cid for cid, _ in sparse_hits[: config.fusion_pool]]
        rrf_score_map = {}

    chunks_data = _fetch_chunks(candidate_ids)
    candidates: list[RetrievedChunk] = []
    for cid in candidate_ids:
        if cid not in chunks_data:
            continue
        d = chunks_data[cid]
        candidates.append(RetrievedChunk(
            chunk_id=cid,
            source=d["source"],
            title=d["title"],
            section=d["section"],
            text=d["text"],
            chunk_index=d["chunk_index"],
            dense_score=dense_score_map.get(cid, 0.0),
            sparse_score=sparse_score_map.get(cid, 0.0),
            rrf_score=rrf_score_map.get(cid, 0.0),
        ))

    if config.use_rerank and candidates:
        candidates = rerank(query, candidates, top_n=config.final_k)
    else:
        candidates = candidates[: config.final_k]

    return candidates
