"""Pydantic schemas for structured data models."""
from typing import Optional
from pydantic import BaseModel, Field


class AgentOutput(BaseModel):
    """Structured output from the support agent."""
    intent: str = Field(description="Predicted intent category")
    intent_confidence: float = Field(ge=0.0, le=1.0, description="Confidence score")
    retrieved_cases: list[dict] = Field(default_factory=list, description="Retrieved cases with similarity scores")
    reply: str = Field(description="Generated support reply")
    decision: str = Field(description="AUTO_HANDLE or ESCALATE")
    reason: str = Field(description="Explanation for the escalation decision")
    retrieved_case_ids: list[str] = Field(default_factory=list, description="IDs of retrieved cases for audit")


class EvaluatedCase(BaseModel):
    """A case evaluated against the golden set."""
    example_id: str
    customer_text: str
    conversation_context: str
    predicted_intent: str
    predicted_confidence: float
    predicted_decision: str
    predicted_reason: str
    generated_reply: str
    retrieved_case_ids: list[str] = Field(default_factory=list)
    retrieved_similarities: list[float] = Field(default_factory=list)


class GoldenExample(BaseModel):
    """A golden evaluation example."""
    example_id: str
    conversation_id: str
    customer_text: str
    conversation_context: str
    gold_intent: str
    gold_escalate: bool
    gold_escalation_reason: str
    reference_resolution: str
    difficulty: str
    label_notes: str = ""
    human_verified: bool = False
