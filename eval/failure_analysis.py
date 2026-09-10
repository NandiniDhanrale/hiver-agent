"""Failure analysis for support agent evaluation."""
import json
import logging
from collections import Counter
from typing import Optional
from pathlib import Path

logger = logging.getLogger(__name__)


def categorize_failures(golden: list[dict], predictions: list[dict],
                        judge_scores: Optional[list[dict]] = None) -> list[dict]:
    """Categorize failures based on mismatched predictions and gold labels.

    Returns list of failure records with categories and analysis.
    """
    failures = []

    for i, (g, p) in enumerate(zip(golden, predictions)):
        is_wrong_intent = g.get("gold_intent") != p.get("predicted_intent")
        is_wrong_escalation = g.get("gold_escalate") != (p.get("predicted_decision") == "ESCALATE")

        if not is_wrong_intent and not is_wrong_escalation:
            continue

        # Categorize the failure
        categories = []

        if is_wrong_intent:
            categories.append("intent_confusion")

        if is_wrong_escalation:
            gold_esc = g.get("gold_escalate", False)
            pred_esc = p.get("predicted_decision") == "ESCALATE"
            if gold_esc and not pred_esc:
                categories.append("unsafe_auto_handling")
            elif not gold_esc and pred_esc:
                categories.append("over_escalation")

        # Context-dependent
        if g.get("conversation_context") and len(g.get("conversation_context", "")) > 50:
            categories.append("context_dependent")

        # Short message
        if len(g.get("customer_text", "").split()) < 5:
            categories.append("very_short_message")

        # Multi-intent potential
        if any(c in g.get("customer_text", "").lower() for c in [" and ", " also ", " plus "]):
            categories.append("multi_intent")

        # Weak retrieval
        retrieved = p.get("retrieved_similarities", [])
        if retrieved and max(retrieved) < 0.5:
            categories.append("weak_retrieval")
        elif not retrieved:
            categories.append("no_retrieval_evidence")

        # Judge scores
        judge_score = None
        if judge_scores and i < len(judge_scores):
            judge_score = judge_scores[i]

        failure = {
            "index": i,
            "categories": categories,
            "customer_text": g.get("customer_text", ""),
            "context": g.get("conversation_context", ""),
            "gold_intent": g.get("gold_intent", ""),
            "predicted_intent": p.get("predicted_intent", ""),
            "gold_escalate": g.get("gold_escalate", False),
            "predicted_escalation": p.get("predicted_decision", ""),
            "retrieved_cases": p.get("retrieved_cases", []),
            "generated_reply": p.get("generated_reply", ""),
            "judge_scores": judge_score,
            "gold_reason": g.get("gold_escalation_reason", ""),
            "predicted_reason": p.get("predicted_reason", "")
        }
        failures.append(failure)

    return failures


def analyze_top_failures(failures: list[dict], top_n: int = 5) -> list[dict]:
    """Analyze the top N most important failure modes.

    Returns detailed analysis with examples.
    """
    # Count categories
    category_counts = Counter()
    for f in failures:
        for cat in f["categories"]:
            category_counts[cat] += 1

    top_categories = category_counts.most_common(top_n)

    analysis = []
    for category, count in top_categories:
        examples = [f for f in failures if category in f["categories"]][:3]

        analysis.append({
            "category": category,
            "count": count,
            "pct_of_failures": round(count / max(len(failures), 1) * 100, 1),
            "examples": examples,
            "hypothesis": _get_hypothesis(category),
            "possible_fix": _get_fix(category)
        })

    return analysis


def _get_hypothesis(category: str) -> str:
    """Get hypothesis for why this failure mode occurs."""
    hypotheses = {
        "intent_confusion": "The intent taxonomy may have overlapping categories or the LLM struggles to differentiate similar intents.",
        "unsafe_auto_handling": "The escalation policy may have thresholds that are too permissive, or risk signals are not being detected.",
        "over_escalation": "The escalation policy may be too conservative, flagging safe messages for human review unnecessarily.",
        "context_dependent": "Messages require conversation history to understand, but the context is not being properly passed to the classifier.",
        "very_short_message": "Short messages lack enough signal for reliable classification or escalation decisions.",
        "multi_intent": "Messages containing multiple issues confuse the single-intent classifier.",
        "weak_retrieval": "The embedding model or retrieval index is not finding sufficiently similar historical cases.",
        "no_retrieval_evidence": "No similar historical cases exist in the development set for this type of query."
    }
    return hypotheses.get(category, "Unknown failure mode.")


def _get_fix(category: str) -> str:
    """Get possible fix for this failure mode."""
    fixes = {
        "intent_confusion": "Refine intent taxonomy boundaries, add more training examples for confused classes, or use a stronger classifier.",
        "unsafe_auto_handling": "Lower confidence thresholds, add more escalation keywords, or implement a dedicated risk classifier.",
        "over_escalation": "Raise confidence thresholds, reduce sensitivity of keyword triggers, or add allowlist for common safe patterns.",
        "context_dependent": "Improve conversation context reconstruction, or use multi-turn models that can handle dialogue history.",
        "very_short_message": "Always escalate very short messages, or use retrieval-only mode for them.",
        "multi_intent": "Implement multi-label classification or decompose messages into separate issues.",
        "weak_retrieval": "Fine-tune embeddings on support data, use hybrid retrieval (keyword + semantic), or increase top_k.",
        "no_retrieval_evidence": "Expand the historical case database, or implement out-of-detection for unknown queries."
    }
    return fixes.get(category, "Further analysis needed.")


def save_failure_analysis(failures: list[dict], analysis: list[dict],
                          output_dir: str = None):
    """Save failure analysis outputs."""
    if output_dir is None:
        output_dir = str(Path(__file__).parent.parent / "outputs")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save all failures
    with open(output_path / "failure_examples.json", 'w') as f:
        json.dump(failures, f, indent=2)

    # Save analysis
    with open(output_path / "failure_analysis.json", 'w') as f:
        json.dump(analysis, f, indent=2)

    # Save markdown report
    lines = ["# Failure Analysis\n"]
    lines.append(f"Total failures: {len(failures)}\n")

    for a in analysis:
        lines.append(f"## {a['category'].replace('_', ' ').title()}")
        lines.append(f"Count: {a['count']} ({a['pct_of_failures']}% of failures)\n")
        lines.append(f"**Hypothesis:** {a['hypothesis']}\n")
        lines.append(f"**Possible fix:** {a['possible_fix']}\n")

        if a['examples']:
            lines.append("**Example:**")
            ex = a['examples'][0]
            lines.append(f"- Customer: {ex['customer_text'][:150]}")
            lines.append(f"- Gold intent: {ex['gold_intent']}")
            lines.append(f"- Predicted intent: {ex['predicted_intent']}")
            lines.append(f"- Gold escalate: {ex['gold_escalate']}")
            lines.append(f"- Predicted: {ex['predicted_escalation']}")
            lines.append("")

    with open(output_path / "failure_analysis.md", 'w') as f:
        f.write("\n".join(lines))

    logger.info(f"Saved failure analysis to {output_path}")
