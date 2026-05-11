"""Eval runner. Measures four metrics per case:
  - correctness: LLM-as-judge comparison against ideal answer (1-5)
  - faithfulness: are all claims grounded in retrieved context? (0/1 per claim)
  - retrieval_relevance: fraction of expected_sources actually retrieved
  - citation_accuracy: % of citations verified as supporting their claim

Also supports running the same suite across multiple chunking strategies for
the chunking comparison report.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Optional

from groq import Groq

from .ask import AnswerBundle, ask
from .rate_limit import rate_limited
from .retrieve import RetrievalConfig


EVAL_DIR = Path(__file__).resolve().parents[2] / "eval"


@dataclass
class EvalCase:
    id: str
    question: str
    ideal_answer: str
    expected_sources: list[str]
    type: str
    difficulty: str


@dataclass
class CaseScore:
    case_id: str
    correctness: float       # 1-5
    faithfulness: float      # 0-1
    retrieval_relevance: float  # 0-1
    citation_accuracy: float  # 0-1
    refused: bool
    answer: str


@dataclass
class StrategyReport:
    strategy: str
    n_cases: int
    mean_correctness: float
    mean_faithfulness: float
    mean_retrieval_relevance: float
    mean_citation_accuracy: float
    per_case: list[CaseScore]


def load_cases(name: str = "qa_pairs") -> list[EvalCase]:
    with open(EVAL_DIR / f"{name}.json") as f:
        data = json.load(f)
    return [EvalCase(**c) for c in data["cases"]]


@lru_cache(maxsize=1)
def _judge_client() -> Groq:
    return Groq(api_key=os.environ["GROQ_API_KEY"])


_JUDGE_MODEL = os.environ.get("LLM_MODEL", "llama-3.3-70b-versatile")
_CITATION_RE = re.compile(r"\[(\d+)\]")


CORRECTNESS_SYSTEM = """Grade an LLM answer against a gold standard on a 1-5 scale.
5 = equivalent to gold; 4 = minor omission; 3 = significant omission; 2 = mostly wrong; 1 = hallucinated.
For "no-answer" type questions, a refusal that doesn't fabricate facts scores 5.
Respond ONLY with JSON: {"score": 1-5, "reason": "<one sentence>"}"""


@rate_limited
def _grade_call(messages: list) -> object:
    return _judge_client().chat.completions.create(
        model=_JUDGE_MODEL,
        temperature=0,
        response_format={"type": "json_object"},
        messages=messages,
    )


def _grade_correctness(question: str, ideal: str, candidate: str, case_type: str) -> int:
    try:
        prompt = (
            f"Question: {question}\n\n"
            f"Gold answer: {ideal}\n\n"
            f"Candidate answer: {candidate}\n\n"
            f"Question type: {case_type}"
        )
        resp = _grade_call([
            {"role": "system", "content": CORRECTNESS_SYSTEM},
            {"role": "user", "content": prompt},
        ])
        data = json.loads(resp.choices[0].message.content or "{}")
        return max(1, min(5, int(data.get("score", 0)))) or 0
    except Exception:
        return 0


def _faithfulness(bundle: AnswerBundle) -> float:
    """Fraction of cited sentences whose citation_verdict.supported = True."""
    if bundle.refused:
        return 1.0  # refusing is "faithful" — no hallucination
    if not bundle.citation_verdicts:
        # No verdicts means no citations were extracted — score 0 unless answer has no factual claims
        return 0.0
    supported = sum(1 for v in bundle.citation_verdicts if v.supported)
    return supported / len(bundle.citation_verdicts)


def _retrieval_relevance(case: EvalCase, bundle: AnswerBundle) -> float:
    if not case.expected_sources:
        # No-answer questions: relevance = 1.0 if retrieval found nothing relevant or model refused
        return 1.0 if bundle.refused else 0.5
    retrieved_sources = {c.source for c in bundle.chunks}
    hits = sum(1 for s in case.expected_sources if any(s in r for r in retrieved_sources))
    return hits / len(case.expected_sources)


def _citation_accuracy(bundle: AnswerBundle) -> float:
    if not bundle.citation_verdicts:
        return 1.0 if bundle.refused else 0.0
    supported = sum(1 for v in bundle.citation_verdicts if v.supported)
    return supported / len(bundle.citation_verdicts)


def evaluate_case(case: EvalCase, config: RetrievalConfig, verify: bool = True) -> CaseScore:
    bundle = ask(case.question, config=config, verify=verify)
    correctness = _grade_correctness(case.question, case.ideal_answer, bundle.answer, case.type)
    faithfulness = _faithfulness(bundle)
    relevance = _retrieval_relevance(case, bundle)
    cit_accuracy = _citation_accuracy(bundle)
    return CaseScore(
        case_id=case.id,
        correctness=float(correctness),
        faithfulness=faithfulness,
        retrieval_relevance=relevance,
        citation_accuracy=cit_accuracy,
        refused=bundle.refused,
        answer=bundle.answer,
    )


def evaluate_strategy(strategy: str, dataset: str = "qa_pairs", verify: bool = True) -> StrategyReport:
    cases = load_cases(dataset)
    config = RetrievalConfig(strategy=strategy)
    scores = [evaluate_case(c, config, verify=verify) for c in cases]
    n = len(scores) or 1
    return StrategyReport(
        strategy=strategy,
        n_cases=len(scores),
        mean_correctness=sum(s.correctness for s in scores) / n,
        mean_faithfulness=sum(s.faithfulness for s in scores) / n,
        mean_retrieval_relevance=sum(s.retrieval_relevance for s in scores) / n,
        mean_citation_accuracy=sum(s.citation_accuracy for s in scores) / n,
        per_case=scores,
    )
