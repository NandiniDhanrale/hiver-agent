"""Trivial baseline: always predict the majority class."""
import json
import logging
from collections import Counter
from typing import Optional

logger = logging.getLogger(__name__)


class TrivialBaseline:
    """Baseline that always predicts the most common intent and reply."""

    def __init__(self):
        self.majority_intent: str = "OTHER"
        self.majority_reply: str = "Thank you for reaching out. We'll get back to you shortly."
        self.majority_decision: str = "ESCALATE"
        self._fitted = False

    def fit(self, cases: list[dict]):
        """Learn majority classes from development data."""
        # Count intents (if available) - for now use text heuristics
        # In practice, we'd need labels for the dev set
        intent_counts = Counter()
        reply_counts = Counter()

        for case in cases:
            # Heuristic intent assignment for baseline fitting
            intent = self._heuristic_intent(case["customer_text"])
            intent_counts[intent] += 1
            reply_counts[case["brand_reply"]] += 1

        if intent_counts:
            self.majority_intent = intent_counts.most_common(1)[0][0]
        if reply_counts:
            self.majority_reply = reply_counts.most_common(1)[0][0]

        # Default to ESCALATE for trivial baseline (conservative)
        self.majority_decision = "ESCALATE"
        self._fitted = True
        logger.info(f"Trivial baseline: majority intent={self.majority_intent}, "
                    f"decision={self.majority_decision}")

    def predict(self, customer_text: str, context: str = "") -> dict:
        """Predict using trivial rules."""
        return {
            "intent": self.majority_intent,
            "intent_confidence": 0.5,
            "retrieved_cases": [],
            "reply": self.majority_reply,
            "decision": self.majority_decision,
            "reason": "Trivial baseline: always use majority class",
            "retrieved_case_ids": []
        }

    def _heuristic_intent(self, text: str) -> str:
        """Simple heuristic for intent during fitting."""
        text_lower = text.lower()
        if any(w in text_lower for w in ['charge', 'refund', 'bill', 'payment']):
            return "BILLING_PAYMENT"
        if any(w in text_lower for w in ['login', 'password', 'locked', "can't log"]):
            return "LOGIN_ACCOUNT_ACCESS"
        if any(w in text_lower for w in ['crash', 'bug', 'slow', 'freeze']):
            return "PLAYBACK_APP_ISSUE"
        if any(w in text_lower for w in ['premium', 'free', 'subscribe', 'plan']):
            return "SUBSCRIPTION_PLAN"
        if any(w in text_lower for w in ['family', 'student']):
            return "FAMILY_STUDENT_PLAN"
        if any(w in text_lower for w in ['hacked', 'compromised', 'security']):
            return "ACCOUNT_SECURITY"
        if any(w in text_lower for w in ['add', 'feature', 'suggestion']):
            return "FEATURE_REQUEST"
        return "OTHER"
