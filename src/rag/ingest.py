"""End-to-end ingestion: load → chunk → embed → store → build BM25."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from rich.console import Console

from .bm25_index import build_bm25
from .chunkers import chunk_doc, STRATEGIES
from .embed import embed_texts
from .loaders import load_directory
from .store import (
    all_chunks,
    init_schema,
    insert_chunks,
    reset_strategy,
    upsert_document,
)


console = Console()


def ingest_directory(
    root: Path,
    strategy: str = "recursive",
    reset: bool = False,
    skip_duplicates: bool = True,
) -> dict:
    if strategy not in STRATEGIES:
        raise ValueError(f"unknown strategy: {strategy}")

    init_schema()
    if reset:
        deleted = reset_strategy(strategy)
        console.print(f"[yellow]reset[/]: deleted {deleted} existing chunks for strategy={strategy}")

    docs = list(load_directory(root))
    console.print(f"loaded [bold]{len(docs)}[/] documents from {root}")

    total = {"docs": 0, "inserted": 0, "skipped_duplicates": 0}
    embed_fn = embed_texts if strategy == "semantic" else None

    for doc in docs:
        chunks = chunk_doc(doc, strategy, embed_fn=embed_fn)
        if not chunks:
            continue
        embeddings = embed_texts([c.text for c in chunks])
        doc_id = upsert_document(doc.source, doc.title)
        stats = insert_chunks(doc_id, chunks, embeddings, skip_duplicates=skip_duplicates)
        total["docs"] += 1
        total["inserted"] += stats["inserted"]
        total["skipped_duplicates"] += stats["skipped_duplicates"]
        console.print(
            f"  · [cyan]{doc.source}[/] → {stats['inserted']} new, "
            f"{stats['skipped_duplicates']} duped"
        )

    # Rebuild BM25 from all chunks of this strategy
    chunks_for_bm25 = all_chunks(strategy)
    bm25 = build_bm25(chunks_for_bm25)
    bm25.save(strategy)
    console.print(f"[green]✓[/] BM25 index rebuilt: {len(chunks_for_bm25)} chunks")
    total["bm25_chunks"] = len(chunks_for_bm25)
    return total
