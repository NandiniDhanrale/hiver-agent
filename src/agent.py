"""Main support agent that combines classification, retrieval, and escalation."""
import json
import logging
from typing import Optional

from .intents import classify_intent_llm, get_intent_descriptions
from .retrieval import RetrievalIndex
from .escalation import check_escalation
from .llm import llm_call
from .schemas import AgentOutput

logger = logging.getLogger(__name__)


class SupportAgent:
    """AI customer support agent for SpotifyCares."""

    def __init__(self, retrieval_index: RetrievalIndex):
        self.retrieval = retrieval_index
        self.intent_descriptions = get_intent_descriptions()

    def process(self, customer_text: str, conversation_context: str = "") -> AgentOutput:
        """Process a customer message end-to-end.

        Returns structured AgentOutput with intent, reply, and escalation decision.
        """
        # Step 1: Classify intent
        intent, confidence = classify_intent_llm(
            customer_text, conversation_context,
            self.intent_descriptions, llm_call
        )
        logger.info(f"Intent: {intent} (confidence: {confidence:.2f})")

        # Step 2: Retrieve similar historical cases
        query_text = f"{conversation_context} {customer_text}".strip()
        retrieved = self.retrieval.retrieve(query_text, top_k=5)
        logger.info(f"Retrieved {len(retrieved)} cases, top similarity: {retrieved[0]['similarity']:.2f}" if retrieved else "No cases retrieved")

        retrieved_cases = [
            {"case_id": c["case_id"], "similarity": round(c["similarity"], 3)}
            for c in retrieved
        ]
        retrieved_case_ids = [c["case_id"] for c in retrieved]

        # Step 3: Generate reply grounded in retrieved evidence
        reply = self._generate_reply(
            customer_text, conversation_context, intent, retrieved
        )

        # Step 4: Apply escalation policy
        should_escalate, reason, rules = check_escalation(
            intent, confidence, retrieved, reply, customer_text
        )

        decision = "ESCALATE" if should_escalate else "AUTO_HANDLE"

        return AgentOutput(
            intent=intent,
            intent_confidence=round(confidence, 3),
            retrieved_cases=retrieved_cases,
            reply=reply,
            decision=decision,
            reason=reason,
            retrieved_case_ids=retrieved_case_ids
        )

    def _generate_reply(self, customer_text: str, context: str, intent: str,
                         retrieved_cases: list[dict]) -> str:
        """Generate a grounded reply using retrieved historical cases."""
        # Format retrieved evidence
        evidence_lines = []
        for i, case in enumerate(retrieved_cases[:3], 1):
            evidence_lines.append(
                f"Case {i} (similarity: {case['similarity']:.2f}):\n"
                f"  Customer: {case['customer_text'][:200]}\n"
                f"  Resolution: {case['brand_reply'][:200]}"
            )
        evidence = "\n\n".join(evidence_lines)

        prompt = f"""You are a Spotify customer support agent. Draft a reply to this customer message.

CUSTOMER MESSAGE: {customer_text}
{f'CONTEXT: {context}' if context else ''}
PREDICTED INTENT: {intent}

HISTORICAL RESOLUTIONS (use as evidence):
{evidence}

RULES:
- Do NOT invent Spotify policies
- Do NOT invent account information
- Do NOT claim an action was completed if it was not
- Do NOT promise refunds or account changes without evidence
- Use the historical resolutions above as evidence
- If evidence is insufficient, give only safe generic guidance and recommend escalation
- Never request passwords
- Do not expose historical customer PII
- Preserve concise customer-support tone
- Do NOT include @mentions or hashtags in your reply

Draft a concise, helpful reply:"""

        reply = llm_call(prompt, temperature=0.3)

        # Clean the reply
        reply = reply.strip()
        # Remove wrapping quotes if present
        if reply.startswith('"') and reply.endswith('"'):
            reply = reply[1:-1]
        # Remove @mentions at start
        while reply.startswith("@"):
            parts = reply.split(" ", 1)
            if len(parts) > 1:
                reply = parts[1]
            else:
                break

        return reply


def create_agent(cases: list[dict], model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                  cache_dir: str = "embeddings_cache") -> SupportAgent:
    """Factory function to create a configured SupportAgent."""
    index = RetrievalIndex(model_name=model_name, cache_dir=cache_dir)
    index.build_index(cases)
    return SupportAgent(retrieval_index=index)
