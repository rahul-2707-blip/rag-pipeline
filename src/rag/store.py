"""Postgres + pgvector storage for chunks and embeddings.

Schema:
  documents(id, source, title, ingested_at)
  chunks(id, document_id, chunk_index, strategy, section, text, char_count, embedding vector(D))

Near-duplicate detection: cosine similarity > 0.95 to ANY existing chunk skips insertion.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict
from typing import Iterable, Optional

import numpy as np
import psycopg
from pgvector.psycopg import register_vector

from .chunkers import Chunk
from .embed import DEFAULT_MODEL, embedding_dim


DUPLICATE_THRESHOLD = 0.95


def _connect(register: bool = True) -> psycopg.Connection:
    url = os.environ.get("DATABASE_URL")
    if not url:
        raise RuntimeError("DATABASE_URL is not set")
    conn = psycopg.connect(url, autocommit=False)
    if register:
        register_vector(conn)
    return conn


def init_schema() -> None:
    dim = embedding_dim()
    # First create the extension on a raw connection (vector type doesn't exist yet)
    with _connect(register=False) as conn, conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        conn.commit()
    # Now safe to register and create the rest of the schema
    with _connect() as conn, conn.cursor() as cur:
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS documents (
                id SERIAL PRIMARY KEY,
                source TEXT NOT NULL UNIQUE,
                title TEXT NOT NULL,
                ingested_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            );
            """
        )
        cur.execute(
            f"""
            CREATE TABLE IF NOT EXISTS chunks (
                id SERIAL PRIMARY KEY,
                document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                chunk_index INTEGER NOT NULL,
                strategy TEXT NOT NULL,
                section TEXT,
                text TEXT NOT NULL,
                char_count INTEGER NOT NULL,
                embedding vector({dim})
            );
            """
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS chunks_strategy_idx ON chunks(strategy);"
        )
        # Cosine HNSW index for fast similarity search
        cur.execute(
            "CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw "
            "ON chunks USING hnsw (embedding vector_cosine_ops);"
        )
        conn.commit()


def reset_strategy(strategy: str) -> int:
    """Delete all chunks for a given strategy (used when re-indexing)."""
    with _connect() as conn, conn.cursor() as cur:
        cur.execute("DELETE FROM chunks WHERE strategy = %s", (strategy,))
        deleted = cur.rowcount
        conn.commit()
    return deleted


def upsert_document(source: str, title: str) -> int:
    with _connect() as conn, conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO documents (source, title)
            VALUES (%s, %s)
            ON CONFLICT (source) DO UPDATE SET title = EXCLUDED.title, ingested_at = NOW()
            RETURNING id
            """,
            (source, title),
        )
        doc_id = cur.fetchone()[0]
        conn.commit()
    return doc_id


def insert_chunks(
    document_id: int,
    chunks: list[Chunk],
    embeddings: np.ndarray,
    skip_duplicates: bool = True,
) -> dict:
    """Insert chunks + their embeddings. Returns stats incl. dedup count."""
    stats = {"inserted": 0, "skipped_duplicates": 0}
    if not chunks:
        return stats

    with _connect() as conn, conn.cursor() as cur:
        for chunk, vec in zip(chunks, embeddings):
            if skip_duplicates:
                cur.execute(
                    "SELECT 1 FROM chunks "
                    "WHERE strategy = %s "
                    "  AND 1 - (embedding <=> %s::vector) > %s "
                    "LIMIT 1",
                    (chunk.strategy, vec.tolist(), DUPLICATE_THRESHOLD),
                )
                if cur.fetchone() is not None:
                    stats["skipped_duplicates"] += 1
                    continue
            cur.execute(
                """
                INSERT INTO chunks
                    (document_id, chunk_index, strategy, section, text, char_count, embedding)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    document_id,
                    chunk.chunk_index,
                    chunk.strategy,
                    chunk.section,
                    chunk.text,
                    len(chunk.text),
                    vec.tolist(),
                ),
            )
            stats["inserted"] += 1
        conn.commit()
    return stats


def dense_search(query_embedding: np.ndarray, strategy: str, k: int = 10) -> list[dict]:
    """Return top-k chunks by cosine similarity for the given strategy."""
    with _connect() as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT c.id, c.document_id, d.source, d.title, c.section, c.text,
                   c.chunk_index, 1 - (c.embedding <=> %s::vector) AS score
            FROM chunks c
            JOIN documents d ON d.id = c.document_id
            WHERE c.strategy = %s
            ORDER BY c.embedding <=> %s::vector
            LIMIT %s
            """,
            (query_embedding.tolist(), strategy, query_embedding.tolist(), k),
        )
        rows = cur.fetchall()
    return [
        {
            "chunk_id": r[0],
            "document_id": r[1],
            "source": r[2],
            "title": r[3],
            "section": r[4],
            "text": r[5],
            "chunk_index": r[6],
            "score": float(r[7]),
        }
        for r in rows
    ]


def all_chunks(strategy: str) -> list[dict]:
    """Fetch every chunk for a strategy (used to build the BM25 index)."""
    with _connect() as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT c.id, d.source, d.title, c.section, c.text, c.chunk_index
            FROM chunks c
            JOIN documents d ON d.id = c.document_id
            WHERE c.strategy = %s
            ORDER BY c.id
            """,
            (strategy,),
        )
        rows = cur.fetchall()
    return [
        {
            "chunk_id": r[0],
            "source": r[1],
            "title": r[2],
            "section": r[3],
            "text": r[4],
            "chunk_index": r[5],
        }
        for r in rows
    ]


def list_documents() -> list[dict]:
    with _connect() as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT d.id, d.source, d.title, d.ingested_at,
                   (SELECT COUNT(*) FROM chunks c WHERE c.document_id = d.id)
            FROM documents d
            ORDER BY d.source
            """
        )
        rows = cur.fetchall()
    return [
        {"id": r[0], "source": r[1], "title": r[2], "ingested_at": r[3].isoformat(), "chunk_count": r[4]}
        for r in rows
    ]
