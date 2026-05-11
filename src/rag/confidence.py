"""Composite confidence score for an answer.

Three dimensions:
  - retrieval_confidence: how relevant are the top chunks?
    (mean rerank score if reranker used, else mean dense score)
  - citation_coverage: fraction of factual sentences that have at least one verified citation
  - answer_completeness: 1.0 if the model didn't bail with "I don't have enough information", else 0.0

Composite = weighted sum, configurable.

If retrieval_confidence < threshold → caller should return the structured
"I don't know" response instead of the generated answer.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from .generate import CitationVerdict, Generation
from .retrieve import RetrievedChunk


RETRIEVAL_FLOOR = 0.25         # below this → refuse to answer
COMPLETENESS_BAIL_RE = re.compile(r"I don't have enough information", re.IGNORECASE)


@dataclass
class Confidence:
    retrieval: float
    citation_coverage: float
    completeness: float
    composite: float


def _retrieval_score(chunks: list[RetrievedChunk]) -> float:
    if not chunks:
        return 0.0
    # Prefer rerank scores when available; rerank score is unbounded so squash via sigmoid
    has_rerank = any(c.rerank_score != 0.0 for c in chunks)
    if has_rerank:
        mean = sum(c.rerank_score for c in chunks) / len(chunks)
        # bge-reranker outputs roughly in [-10, 10]; map to (0, 1)
        return 1.0 / (1.0 + pow(2.718281828, -mean))
    return sum(c.dense_score for c in chunks) / len(chunks)


def _citation_coverage(generation: Generation, verdicts: list[CitationVerdict]) -> float:
    # Count factual sentences (non-empty, not the bail message)
    sentences = re.split(r"(?<=[.!?])\s+", generation.answer.strip())
    factual = [s for s in sentences if s.strip() and not COMPLETENESS_BAIL_RE.search(s)]
    if not factual:
        return 0.0
    cite_re = re.compile(r"\[(\d+)\]")
    supported_sentences = 0
    verdicts_by_sentence: dict[str, list[CitationVerdict]] = {}
    for v in verdicts:
        verdicts_by_sentence.setdefault(v.sentence, []).append(v)
    for s in factual:
        if not cite_re.search(s):
            continue  # cited zero times → not supported
        if any(v.supported for v in verdicts_by_sentence.get(s.strip(), [])):
            supported_sentences += 1
    return supported_sentences / len(factual)


def _completeness(generation: Generation) -> float:
    return 0.0 if COMPLETENESS_BAIL_RE.search(generation.answer) else 1.0


def score(
    generation: Generation,
    chunks: list[RetrievedChunk],
    verdicts: list[CitationVerdict],
    w_retrieval: float = 0.4,
    w_citation: float = 0.4,
    w_completeness: float = 0.2,
) -> Confidence:
    r = _retrieval_score(chunks)
    c = _citation_coverage(generation, verdicts)
    cm = _completeness(generation)
    composite = w_retrieval * r + w_citation * c + w_completeness * cm
    return Confidence(retrieval=r, citation_coverage=c, completeness=cm, composite=composite)


def should_refuse(retrieval_score: float, floor: float = RETRIEVAL_FLOOR) -> bool:
    return retrieval_score < floor


def refusal_response(chunks: list[RetrievedChunk]) -> str:
    """Structured 'I don't know' that points to potentially-relevant docs."""
    if not chunks:
        return "I don't have enough information in the indexed documents to answer that."
    seen: set[str] = set()
    paths: list[str] = []
    for c in chunks[:3]:
        if c.source not in seen:
            paths.append(c.source)
            seen.add(c.source)
    return (
        "I don't have enough information in the indexed documents to answer confidently. "
        f"You may want to manually check: {', '.join(paths)}."
    )
