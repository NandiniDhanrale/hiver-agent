"""Escalation policy engine."""
import yaml
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

SETTINGS_FILE = Path(__file__).parent.parent / "config" / "settings.yaml"


def load_escalation_config() -> dict:
    """Load escalation configuration."""
    with open(SETTINGS_FILE, 'r') as f:
        config = yaml.safe_load(f)
    return config.get("escalation", {})


def check_escalation(intent: str, confidence: float, retrieved_cases: list[dict],
                      reply: str, customer_text: str) -> tuple[bool, str, list[str]]:
    """Apply explicit hybrid escalation policy.

    Returns (should_escalate, reason, triggered_rules)
    """
    config = load_escalation_config()
    auto_config = config.get("auto_handle", {})
    flags_config = config.get("flags", {})

    confidence_threshold = auto_config.get("confidence_threshold", 0.75)
    similarity_threshold = auto_config.get("retrieval_similarity_threshold", 0.60)

    triggered_rules = []

    # Check keyword-based escalation flags
    text_lower = customer_text.lower()

    # Account compromise check
    compromise_keywords = flags_config.get("account_compromise_keywords", [])
    for kw in compromise_keywords:
        if kw.lower() in text_lower:
            triggered_rules.append(f"account_compromise: '{kw}' detected")

    # Payment dispute check
    payment_keywords = flags_config.get("payment_keywords", [])
    for kw in payment_keywords:
        if kw.lower() in text_lower:
            triggered_rules.append(f"payment_dispute: '{kw}' detected")

    # Safety/legal check
    safety_keywords = flags_config.get("safety_keywords", [])
    for kw in safety_keywords:
        if kw.lower() in text_lower:
            triggered_rules.append(f"safety_concern: '{kw}' detected")

    # Check confidence threshold
    if confidence < confidence_threshold:
        triggered_rules.append(f"low_confidence: {confidence:.2f} < {confidence_threshold}")

    # Check retrieval evidence
    if not retrieved_cases:
        triggered_rules.append("no_retrieval_evidence: no similar cases found")
    else:
        max_similarity = max(c.get("similarity", 0) for c in retrieved_cases)
        if max_similarity < similarity_threshold:
            triggered_rules.append(
                f"weak_retrieval: max similarity {max_similarity:.2f} < {similarity_threshold}"
            )

        # Check for disagreement in retrieved cases
        intents_in_retrieved = set()
        for c in retrieved_cases:
            # Cases don't have intent labels, but we can check reply diversity
            pass

    # Check for very short messages (potential spam/noise)
    if len(customer_text.split()) < 3:
        triggered_rules.append("very_short_message: potential noise")

    # Check for ALL CAPS (potential frustration)
    words = customer_text.split()
    if len(words) > 3 and all(w.isupper() for w in words if w.isalpha()):
        triggered_rules.append("all_caps: potential frustration")

    # Make escalation decision
    # Strong signals always escalate
    strong_signals = [r for r in triggered_rules if any(
        kw in r for kw in ["account_compromise", "payment_dispute", "safety_concern"]
    )]

    # Weak signals need accumulation
    weak_signals = [r for r in triggered_rules if r not in strong_signals]

    should_escalate = len(strong_signals) > 0 or len(weak_signals) >= 2

    # Build reason
    if should_escalate:
        if strong_signals:
            reason = f"Escalating due to: {'; '.join(strong_signals)}"
        else:
            reason = f"Escalating due to multiple risk signals: {'; '.join(weak_signals)}"
    else:
        if triggered_rules:
            reason = f"Auto-handling despite: {'; '.join(triggered_rules)} (below threshold)"
        else:
            reason = "Auto-handling: high confidence, strong evidence, no risk signals"

    return should_escalate, reason, triggered_rules
