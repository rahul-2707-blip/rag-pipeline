"""Grounded answer generation with inline [n] citations.

The model is instructed to:
  1. Only answer from the provided context
  2. Cite specific numbered chunks using [1], [2], …
  3. Explicitly say "I don't know" when context is insufficient
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Optional

from groq import Groq

from .rate_limit import rate_limited
from .retrieve import RetrievedChunk


LLM_MODEL = os.environ.get("LLM_MODEL", "llama-3.3-70b-versatile")


@lru_cache(maxsize=1)
def _client() -> Groq:
    return Groq(api_key=os.environ["GROQ_API_KEY"])


SYSTEM_PROMPT = """You answer technical questions strictly from the provided context.

Rules:
1. Only use facts from the numbered Context blocks below. Do not invent.
2. Every factual claim MUST be followed by a citation in square brackets, e.g. "FastAPI uses Pydantic [2]."
3. Citations refer to the number of the Context block. Use multiple citations when relevant, e.g. [1][3].
4. If the context is insufficient to answer, respond exactly with:
   "I don't have enough information in the provided documents to answer that. The closest related material is in [<best_match_number>]."
5. Be concise. Do not preface or apologize.
6. Do not cite a chunk you didn't use.
"""


@dataclass
class Generation:
    answer: str
    citations: list[int]
    used_chunks: list[RetrievedChunk]
    raw_prompt: str


_CITATION_RE = re.compile(r"\[(\d+)\]")


def _format_context(chunks: list[RetrievedChunk]) -> str:
    blocks = []
    for i, c in enumerate(chunks, start=1):
        header = f"[Context {i}] source: {c.source}"
        if c.section:
            header += f" · section: {c.section}"
        blocks.append(f"{header}\n{c.text}")
    return "\n\n---\n\n".join(blocks)


@rate_limited
def _generate_call(messages: list) -> object:
    return _client().chat.completions.create(
        model=LLM_MODEL,
        temperature=0.1,
        messages=messages,
    )


def generate(question: str, chunks: list[RetrievedChunk]) -> Generation:
    context = _format_context(chunks)
    user_msg = f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer with citations:"

    response = _generate_call([
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ])
    answer = response.choices[0].message.content or ""

    # Parse citation numbers actually used
    citations_used = sorted({int(m) for m in _CITATION_RE.findall(answer) if 1 <= int(m) <= len(chunks)})

    return Generation(
        answer=answer.strip(),
        citations=citations_used,
        used_chunks=chunks,
        raw_prompt=user_msg,
    )


# ---------- Citation verification ----------

VERIFY_SYSTEM = """You are a citation auditor. Given a claim and a source passage, decide whether
the passage SUPPORTS the claim (the claim is directly inferable from the passage).
Respond ONLY with JSON: {"supported": true | false, "reason": "<one sentence>"}"""


def _extract_claims_and_citations(answer: str) -> list[tuple[str, list[int]]]:
    """Split answer into sentences, then for each sentence extract its citation numbers."""
    # Split on sentence-ending punctuation
    sentences = re.split(r"(?<=[.!?])\s+", answer.strip())
    out: list[tuple[str, list[int]]] = []
    for s in sentences:
        cites = [int(m) for m in _CITATION_RE.findall(s)]
        if s.strip():
            out.append((s.strip(), sorted(set(cites))))
    return out


@dataclass
class CitationVerdict:
    sentence: str
    citation_num: int
    supported: bool
    reason: str


@rate_limited
def _verify_call(messages: list) -> object:
    return _client().chat.completions.create(
        model=LLM_MODEL,
        temperature=0,
        response_format={"type": "json_object"},
        messages=messages,
    )


def _verify_one(sentence: str, n: int, chunk) -> CitationVerdict:
    try:
        response = _verify_call([
            {"role": "system", "content": VERIFY_SYSTEM},
            {
                "role": "user",
                "content": (
                    f"Claim: {sentence}\n\nSource passage:\n{chunk.text}\n\nVerify now."
                ),
            },
        ])
        data = json.loads(response.choices[0].message.content or "{}")
        return CitationVerdict(
            sentence=sentence,
            citation_num=n,
            supported=bool(data.get("supported", False)),
            reason=str(data.get("reason", "")),
        )
    except Exception as e:
        return CitationVerdict(
            sentence=sentence,
            citation_num=n,
            supported=False,
            reason=f"verification error: {e}",
        )


def verify_citations(generation: Generation) -> list[CitationVerdict]:
    """Verify each cited claim against its source chunk. Parallelized across calls."""
    from concurrent.futures import ThreadPoolExecutor

    chunk_by_num = {i + 1: c for i, c in enumerate(generation.used_chunks)}
    tasks: list[tuple[str, int, object]] = []
    for sentence, cites in _extract_claims_and_citations(generation.answer):
        for n in cites:
            chunk = chunk_by_num.get(n)
            if chunk:
                tasks.append((sentence, n, chunk))

    if not tasks:
        return []
    with ThreadPoolExecutor(max_workers=min(5, len(tasks))) as pool:
        verdicts = list(pool.map(lambda t: _verify_one(*t), tasks))
    return verdicts
