"""LLM-as-judge for reply quality evaluation."""
import json
import logging
import hashlib
from typing import Optional
from pathlib import Path

logger = logging.getLogger(__name__)

_judge_cache: dict = {}
_cache_file = Path(__file__).parent.parent / "outputs" / "judge_cache.json"


def load_judge_cache():
    """Load judge cache from disk."""
    global _judge_cache
    if _cache_file.exists():
        try:
            with open(_cache_file, 'r') as f:
                _judge_cache = json.load(f)
            logger.info(f"Loaded {len(_judge_cache)} cached judge scores")
        except Exception:
            _judge_cache = {}


def save_judge_cache():
    """Save judge cache to disk."""
    try:
        _cache_file.parent.mkdir(parents=True, exist_ok=True)
        with open(_cache_file, 'w') as f:
            json.dump(_judge_cache, f, indent=2)
    except Exception as e:
        logger.warning(f"Failed to save judge cache: {e}")


def judge_reply(customer_text: str, context: str, intent: str,
                retrieved_evidence: str, candidate_reply: str,
                llm_call_fn, use_cache: bool = True) -> dict:
    """Score a generated reply using LLM-as-judge rubric.

    Returns scores 1-5 for each dimension plus rationale.
    """
    # Check cache
    if use_cache:
        cache_key = hashlib.sha256(
            f"{customer_text}:{candidate_reply}".encode()
        ).hexdigest()[:16]
        if cache_key in _judge_cache:
            return _judge_cache[cache_key]

    prompt = f"""You are evaluating a customer support reply for quality.

CUSTOMER MESSAGE: {customer_text}
{f'CONTEXT: {context}' if context else ''}
INTENT: {intent}
RETRIEVED EVIDENCE: {retrieved_evidence}

CANDIDATE REPLY: {candidate_reply}

Score this reply on each dimension (1-5, where 5 is best):

1. CORRECTNESS: Is the reply factually correct and accurate?
2. RELEVANCE: Does the reply address the customer's actual question?
3. GROUNDEDNESS: Is the reply supported by the retrieved historical evidence?
4. HELPFULNESS: Does the reply provide actionable guidance or resolution?
5. BRAND_CONSISTENCY: Does the reply match professional support tone?
6. SAFETY: Does the reply avoid harmful, misleading, or risky claims?

Respond with ONLY a JSON object:
{{
  "correctness": <1-5>,
  "relevance": <1-5>,
  "groundedness": <1-5>,
  "helpfulness": <1-5>,
  "brand_consistency": <1-5>,
  "safety": <1-5>,
  "rationale": "<brief explanation>"
}}"""

    try:
        response = llm_call_fn(prompt, temperature=0.0)
        response = response.strip()
        if response.startswith("```"):
            response = response.split("\n", 1)[1].rsplit("```", 1)[0].strip()

        scores = json.loads(response)

        # Validate scores
        dimensions = ["correctness", "relevance", "groundedness",
                      "helpfulness", "brand_consistency", "safety"]
        for dim in dimensions:
            if dim not in scores:
                scores[dim] = 3
            scores[dim] = max(1, min(5, int(scores[dim])))

        if "rationale" not in scores:
            scores["rationale"] = "No rationale provided"

        # Cache
        if use_cache:
            _judge_cache[cache_key] = scores
            save_judge_cache()

        return scores
    except Exception as e:
        logger.error(f"Judge scoring failed: {e}")
        default = {
            "correctness": 3, "relevance": 3, "groundedness": 3,
            "helpfulness": 3, "brand_consistency": 3, "safety": 3,
            "rationale": f"Judge failed: {str(e)}"
        }
        return default


def aggregate_scores(all_scores: list[dict]) -> dict:
    """Aggregate judge scores across multiple examples."""
    if not all_scores:
        return {}

    dimensions = ["correctness", "relevance", "groundedness",
                  "helpfulness", "brand_consistency", "safety"]

    aggregated = {}
    for dim in dimensions:
        values = [s.get(dim, 3) for s in all_scores]
        aggregated[dim] = {
            "mean": round(sum(values) / len(values), 2),
            "min": min(values),
            "max": max(values),
            "std": round((sum((v - sum(values)/len(values))**2 for v in values) / len(values))**0.5, 2)
        }

    # Overall score
    overall = []
    for s in all_scores:
        dim_scores = [s.get(d, 3) for d in dimensions]
        overall.append(sum(dim_scores) / len(dim_scores))

    aggregated["overall"] = {
        "mean": round(sum(overall) / len(overall), 2),
        "min": round(min(overall), 2),
        "max": round(max(overall), 2)
    }

    aggregated["num_examples"] = len(all_scores)

    return aggregated
