"""End-to-end question answering: retrieve → generate → verify citations → score."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .confidence import (
    Confidence,
    refusal_response,
    score as score_confidence,
    should_refuse,
    _retrieval_score,
)
from .generate import CitationVerdict, Generation, generate, verify_citations
from .retrieve import RetrievalConfig, RetrievedChunk, retrieve


@dataclass
class AnswerBundle:
    question: str
    answer: str
    refused: bool
    chunks: list[RetrievedChunk]
    citations: list[int]
    citation_verdicts: list[CitationVerdict]
    confidence: Confidence
    config: RetrievalConfig


def ask(question: str, config: Optional[RetrievalConfig] = None, verify: bool = True) -> AnswerBundle:
    config = config or RetrievalConfig()
    chunks = retrieve(question, config)

    # Gate on retrieval confidence BEFORE we even spend tokens generating
    r_score = _retrieval_score(chunks)
    if should_refuse(r_score):
        answer = refusal_response(chunks)
        empty_gen = Generation(answer=answer, citations=[], used_chunks=chunks, raw_prompt="")
        verdicts: list[CitationVerdict] = []
        confidence = score_confidence(empty_gen, chunks, verdicts)
        return AnswerBundle(
            question=question,
            answer=answer,
            refused=True,
            chunks=chunks,
            citations=[],
            citation_verdicts=verdicts,
            confidence=confidence,
            config=config,
        )

    generation = generate(question, chunks)
    verdicts = verify_citations(generation) if verify else []
    confidence = score_confidence(generation, chunks, verdicts)

    return AnswerBundle(
        question=question,
        answer=generation.answer,
        refused=False,
        chunks=chunks,
        citations=generation.citations,
        citation_verdicts=verdicts,
        confidence=confidence,
        config=config,
    )
