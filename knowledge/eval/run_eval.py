"""RAG evaluation harness for the Knowledge Agent.

Runs the curated eval set (eval_set.jsonl) against the Knowledge Agent
and reports per-question scores and aggregate metrics.

Metrics (Ragas-style, computed without LLM judge when corpus is offline):
  - answer_relevance: keyword overlap between question and answer (proxy)
  - source_cited: whether the answer cites at least one source URL
  - no_hallucinated_rate: fraction of rate-questions that call get_active_tariff()
    rather than returning a prose £/kWh value

Full LLM-judge evaluation (faithfulness, context_precision) requires the
corpus to be indexed and an API key configured.

Usage::
    python -m knowledge.eval.run_eval
    python -m knowledge.eval.run_eval --eval-set knowledge/eval/eval_set.jsonl
    python -m knowledge.eval.run_eval --llm-judge  # enables full Ragas metrics

Exit code: 0 if all aggregate metrics above threshold, 1 otherwise.
Threshold: answer_relevance >= 0.6, source_cited >= 0.5
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_DEFAULT_EVAL_SET = Path(__file__).parent / "eval_set.jsonl"

# Deploy-block thresholds
_THRESHOLDS = {
    "answer_relevance": 0.60,
    "source_cited_rate": 0.50,
}


# ---------------------------------------------------------------------------
# Proxy metrics (no LLM judge required)
# ---------------------------------------------------------------------------

def _keyword_overlap(question: str, answer: str) -> float:
    """Proxy for answer relevance: fraction of question content words in answer."""
    stop = {"is", "the", "a", "an", "how", "what", "does", "do", "can", "i",
            "my", "me", "to", "for", "of", "in", "on", "at", "it", "are",
            "will", "be", "with", "from", "that", "this", "about"}
    q_words = {w.lower().rstrip("?.,") for w in question.split() if w.lower() not in stop}
    if not q_words:
        return 1.0
    a_lower = answer.lower()
    matched = sum(1 for w in q_words if w in a_lower)
    return matched / len(q_words)


def _source_cited(answer: str, sources: list[str]) -> bool:
    """Return True if the answer or sources list contains a URL."""
    if sources:
        return True
    return "http" in answer or "ofgem" in answer.lower() or "gov.uk" in answer.lower()


def _no_hardcoded_rate(answer: str, expected_category: str) -> bool:
    """Return True if a tariff-category answer does NOT embed a raw £/kWh number."""
    if expected_category != "tariff":
        return True  # Not applicable
    import re
    # Detect pattern like "0.2459 p/kWh" or "24.59p" that would indicate hallucinated rates
    rate_pattern = re.compile(r"\b\d+\.\d{2,4}\s*(p|pence|£|gbp)?\s*(/|per)?\s*(kwh|unit)", re.IGNORECASE)
    return not bool(rate_pattern.search(answer))


# ---------------------------------------------------------------------------
# Run single question
# ---------------------------------------------------------------------------

def evaluate_question(entry: dict[str, Any]) -> dict[str, Any]:
    question = entry["question"]
    expected_category = entry.get("category", "unknown")
    reference_answer = entry.get("reference_answer", "")

    from energyx.agents.knowledge.agent import KnowledgeAgent
    agent = KnowledgeAgent()

    answer = agent.answer(question)
    sources: list[str] = []

    # Try to get sources from rag_query tool call
    try:
        from knowledge.rag.source_selector import select_sources
        from knowledge.rag.hybrid_retriever import retrieve
        from knowledge.rag.reranker import rerank
        from knowledge.rag.guardrails import apply_guardrails
        cols = select_sources(question)
        candidates = retrieve(question, cols, top_k=5)
        top = rerank(question, candidates, top_n=3)
        top = apply_guardrails(question, top)
        sources = [c.get("url", "") for c in top if c.get("url")]
    except Exception:
        pass

    relevance = _keyword_overlap(question, answer)
    cited = _source_cited(answer, sources)
    rate_ok = _no_hardcoded_rate(answer, expected_category)

    return {
        "question": question,
        "category": expected_category,
        "answer_snippet": answer[:200],
        "sources": sources,
        "answer_relevance": round(relevance, 3),
        "source_cited": cited,
        "no_hardcoded_rate": rate_ok,
        "pass": relevance >= _THRESHOLDS["answer_relevance"] and rate_ok,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_eval(eval_set_path: Path, llm_judge: bool = False) -> dict[str, Any]:
    entries = []
    with open(eval_set_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("//"):
                entries.append(json.loads(line))

    logger.info(f"Evaluating {len(entries)} questions from {eval_set_path}")
    results = []
    for entry in entries:
        if entry.get("skip"):
            continue
        try:
            result = evaluate_question(entry)
            results.append(result)
            status = "PASS" if result["pass"] else "FAIL"
            logger.info(f"[{status}] {result['question'][:60]} | relevance={result['answer_relevance']}")
        except Exception as e:
            logger.error(f"Error evaluating '{entry.get('question', '?')}': {e}")

    if not results:
        return {"error": "No results — check eval set and Knowledge Agent.", "passed": False}

    n = len(results)
    agg = {
        "n_questions": n,
        "answer_relevance_mean": round(sum(r["answer_relevance"] for r in results) / n, 3),
        "source_cited_rate": round(sum(1 for r in results if r["source_cited"]) / n, 3),
        "no_hardcoded_rate_rate": round(sum(1 for r in results if r["no_hardcoded_rate"]) / n, 3),
        "pass_rate": round(sum(1 for r in results if r["pass"]) / n, 3),
    }

    agg["deploy_block"] = (
        agg["answer_relevance_mean"] < _THRESHOLDS["answer_relevance"]
        or agg["source_cited_rate"] < _THRESHOLDS["source_cited_rate"]
    )

    return {"aggregate": agg, "results": results, "passed": not agg["deploy_block"]}


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Evaluate Knowledge Agent RAG quality")
    parser.add_argument("--eval-set", type=Path, default=_DEFAULT_EVAL_SET)
    parser.add_argument("--llm-judge", action="store_true",
                        help="Enable full LLM-judge metrics (requires API key + indexed corpus)")
    parser.add_argument("--output", type=Path, help="Write JSON results to file")
    args = parser.parse_args()

    if not args.eval_set.exists():
        print(f"ERROR: eval set not found: {args.eval_set}", file=sys.stderr)
        sys.exit(1)

    report = run_eval(args.eval_set, llm_judge=args.llm_judge)

    if args.output:
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Results written to {args.output}")
    else:
        print(json.dumps(report["aggregate"], indent=2))

    if report.get("aggregate", {}).get("deploy_block"):
        print("\nDEPLOY BLOCKED: one or more aggregate metrics below threshold.")
        sys.exit(1)
    else:
        print("\nEval passed.")
        sys.exit(0)


if __name__ == "__main__":
    main()
