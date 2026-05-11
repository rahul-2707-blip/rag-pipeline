"""FastAPI service exposing the RAG pipeline.

Endpoints:
  POST /v1/ask         — answer a question with citations + confidence
  GET  /v1/documents   — list ingested documents
  POST /v1/ingest      — re-ingest a directory (sync, blocking)
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

load_dotenv()

from rag.ask import ask as ask_question
from rag.ingest import ingest_directory
from rag.retrieve import RetrievalConfig
from rag.store import list_documents


app = FastAPI(
    title="RAG Pipeline",
    description="Hybrid retrieval RAG with citation verification and confidence scoring.",
    version="0.1.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1)
    strategy: str = Field("recursive", pattern="^(fixed|recursive|semantic)$")
    mode: str = Field("hybrid", pattern="^(hybrid|dense|sparse)$")
    use_rerank: bool = True
    final_k: int = Field(5, ge=1, le=20)


class CitedChunk(BaseModel):
    chunk_id: int
    source: str
    title: str
    section: Optional[str]
    text: str
    dense_score: float
    sparse_score: float
    rrf_score: float
    rerank_score: float


class CitationVerdictOut(BaseModel):
    sentence: str
    citation_num: int
    supported: bool
    reason: str


class ConfidenceOut(BaseModel):
    retrieval: float
    citation_coverage: float
    completeness: float
    composite: float


class AskResponse(BaseModel):
    question: str
    answer: str
    refused: bool
    citations: list[int]
    confidence: ConfidenceOut
    citation_verdicts: list[CitationVerdictOut]
    chunks: list[CitedChunk]


@app.post("/v1/ask", response_model=AskResponse)
def post_ask(req: AskRequest):
    config = RetrievalConfig(
        strategy=req.strategy,
        mode=req.mode,
        use_rerank=req.use_rerank,
        final_k=req.final_k,
    )
    bundle = ask_question(req.question, config=config, verify=True)
    return AskResponse(
        question=bundle.question,
        answer=bundle.answer,
        refused=bundle.refused,
        citations=bundle.citations,
        confidence=ConfidenceOut(**bundle.confidence.__dict__),
        citation_verdicts=[CitationVerdictOut(**v.__dict__) for v in bundle.citation_verdicts],
        chunks=[CitedChunk(**c.__dict__) for c in bundle.chunks],
    )


@app.get("/v1/documents")
def get_documents():
    return {"documents": list_documents()}


class IngestRequest(BaseModel):
    root: str = "corpus/fastapi"
    strategy: str = Field("recursive", pattern="^(fixed|recursive|semantic)$")
    reset: bool = False


@app.post("/v1/ingest")
def post_ingest(req: IngestRequest):
    root = Path(req.root)
    if not root.exists():
        raise HTTPException(status_code=400, detail=f"path not found: {req.root}")
    stats = ingest_directory(root, strategy=req.strategy, reset=req.reset)
    return stats


@app.get("/health")
def health():
    return {"ok": True}
