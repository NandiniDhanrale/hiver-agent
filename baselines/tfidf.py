"""TF-IDF + LogisticRegression baseline."""
import json
import logging
import numpy as np
from collections import Counter
from typing import Optional

logger = logging.getLogger(__name__)


class TFIDFBaseline:
    """Baseline using TF-IDF features for classification and retrieval."""

    def __init__(self):
        self._vectorizer = None
        self._classifier = None
        self._cases = None
        self._tfidf_matrix = None
        self._majority_intent = "OTHER"
        self._majority_reply = "Thank you for reaching out. We'll help you soon."
        self._fitted = False

    def fit(self, cases: list[dict], labels: Optional[list[dict]] = None):
        """Fit the baseline on development data.

        If labels are not provided, uses heuristic labeling for intent.
        """
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression

        self._cases = cases
        texts = [self._format_text(c) for c in cases]

        # Create labels if not provided
        if labels is None:
            intents = [self._heuristic_intent(c["customer_text"]) for c in cases]
        else:
            intents = [l.get("intent", "OTHER") for l in labels]

        # Count intents
        intent_counts = Counter(intents)
        if intent_counts:
            self._majority_intent = intent_counts.most_common(1)[0][0]

        # Most common reply
        reply_counts = Counter(c["brand_reply"] for c in cases)
        if reply_counts:
            self._majority_reply = reply_counts.most_common(1)[0][0]

        # TF-IDF features
        self._vectorizer = TfidfVectorizer(max_features=10000, stop_words='english')
        self._tfidf_matrix = self._vectorizer.fit_transform(texts)

        # Train classifier
        self._classifier = LogisticRegression(max_iter=1000, random_state=42)
        self._classifier.fit(self._tfidf_matrix, intents)

        self._fitted = True
        logger.info(f"TF-IDF baseline fitted on {len(cases)} cases, "
                    f"{len(set(intents))} intent classes")

    def predict(self, customer_text: str, context: str = "") -> dict:
        """Predict using TF-IDF features."""
        if not self._fitted:
            return self._default_prediction()

        text = self._format_text({"customer_text": customer_text, "conversation_context": context})
        text_tfidf = self._vectorizer.transform([text])

        # Predict intent
        intent = self._classifier.predict(text_tfidf)[0]
        proba = self._classifier.predict_proba(text_tfidf)[0]
        confidence = float(max(proba))

        # Retrieve nearest neighbor
        retrieved = self._retrieve_nn(text, top_k=5)

        # Use nearest neighbor reply
        if retrieved:
            reply = retrieved[0]["brand_reply"]
        else:
            reply = self._majority_reply

        return {
            "intent": intent,
            "intent_confidence": round(confidence, 3),
            "retrieved_cases": [
                {"case_id": c["case_id"], "similarity": round(c["similarity"], 3)}
                for c in retrieved
            ],
            "reply": reply,
            "decision": "AUTO_HANDLE" if confidence > 0.7 else "ESCALATE",
            "reason": f"TF-IDF baseline: confidence={confidence:.2f}",
            "retrieved_case_ids": [c["case_id"] for c in retrieved]
        }

    def _retrieve_nn(self, query_text: str, top_k: int = 5) -> list[dict]:
        """Retrieve nearest neighbors using TF-IDF."""
        from sklearn.metrics.pairwise import cosine_similarity

        query_tfidf = self._vectorizer.transform([query_text])
        similarities = cosine_similarity(query_tfidf, self._tfidf_matrix).flatten()

        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for idx in top_indices:
            case = self._cases[idx].copy()
            case["similarity"] = float(similarities[idx])
            results.append(case)

        return results

    def _format_text(self, case: dict) -> str:
        """Format case text for TF-IDF."""
        parts = []
        context = case.get("conversation_context", "")
        if context and isinstance(context, str) and context != "nan":
            parts.append(context)
        customer_text = case.get("customer_text", "")
        if customer_text and isinstance(customer_text, str):
            parts.append(customer_text)
        return " ".join(parts) if parts else ""

    def _heuristic_intent(self, text: str) -> str:
        """Simple heuristic for intent."""
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

    def _default_prediction(self) -> dict:
        """Default prediction when not fitted."""
        return {
            "intent": self._majority_intent,
            "intent_confidence": 0.5,
            "retrieved_cases": [],
            "reply": self._majority_reply,
            "decision": "ESCALATE",
            "reason": "TF-IDF baseline: not fitted",
            "retrieved_case_ids": []
        }
