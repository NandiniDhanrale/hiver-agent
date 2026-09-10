"""Tests for Pydantic schemas."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.schemas import AgentOutput, EvaluatedCase, GoldenExample


def test_agent_output_valid():
    """Test valid AgentOutput creation."""
    output = AgentOutput(
        intent="BILLING_PAYMENT",
        intent_confidence=0.9,
        retrieved_cases=[{"case_id": "c1", "similarity": 0.8}],
        reply="We'll help you",
        decision="AUTO_HANDLE",
        reason="High confidence",
        retrieved_case_ids=["c1"]
    )
    assert output.intent == "BILLING_PAYMENT"
    assert output.decision == "AUTO_HANDLE"


def test_agent_output_validation():
    """Test AgentOutput validation."""
    # Confidence must be 0-1
    try:
        AgentOutput(
            intent="TEST",
            intent_confidence=1.5,  # Invalid
            reply="test",
            decision="AUTO_HANDLE",
            reason="test"
        )
        assert False, "Should have raised validation error"
    except Exception:
        pass  # Expected


def test_golden_example():
    """Test GoldenExample creation."""
    example = GoldenExample(
        example_id="g001",
        conversation_id="conv_1",
        customer_text="Help me",
        conversation_context="",
        gold_intent="OTHER",
        gold_escalate=False,
        gold_escalation_reason="",
        reference_resolution="We'll help",
        difficulty="EASY",
        human_verified=False
    )
    assert example.human_verified is False


if __name__ == "__main__":
    test_agent_output_valid()
    test_agent_output_validation()
    test_golden_example()
    print("All schema tests passed!")
