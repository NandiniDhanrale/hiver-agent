"""Reply quality evaluation."""
import json
import logging
from typing import Optional

from .judge import judge_reply, aggregate_scores, load_judge_cache, save_judge_cache

logger = logging.getLogger(__name__)


def evaluate_replies(golden: list[dict], predictions: list[dict],
                     llm_call_fn, sample_size: int = 50,
                     use_cache: bool = True) -> dict:
    """Evaluate reply quality using LLM-as-judge.

    Returns aggregated scores per dimension.
    """
    load_judge_cache()

    # Sample if too many examples
    import random
    if len(golden) > sample_size:
        indices = random.sample(range(len(golden)), sample_size)
        eval_golden = [golden[i] for i in indices]
        eval_predictions = [predictions[i] for i in indices]
    else:
        eval_golden = golden
        eval_predictions = predictions

    all_scores = []
    for g, p in zip(eval_golden, eval_predictions):
        # Build evidence string from retrieved cases
        evidence_parts = []
        for rc in p.get("retrieved_cases", [])[:3]:
            evidence_parts.append(f"Case {rc.get('case_id', '?')}: similarity={rc.get('similarity', 0):.2f}")
        evidence = "; ".join(evidence_parts) if evidence_parts else "No evidence retrieved"

        scores = judge_reply(
            customer_text=g.get("customer_text", ""),
            context=g.get("conversation_context", ""),
            intent=p.get("predicted_intent", "OTHER"),
            retrieved_evidence=evidence,
            candidate_reply=p.get("generated_reply", ""),
            llm_call_fn=llm_call_fn,
            use_cache=use_cache
        )
        all_scores.append(scores)

    aggregated = aggregate_scores(all_scores)
    aggregated["sample_size"] = len(all_scores)

    return {
        "scores": aggregated,
        "per_example": all_scores
    }
