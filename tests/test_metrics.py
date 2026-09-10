"""Tests for metrics calculation."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from eval.evaluate_intents import evaluate_intents
from eval.evaluate_escalation import evaluate_escalation


def test_intent_metrics():
    """Test intent evaluation metrics."""
    golden = [
        {"gold_intent": "BILLING_PAYMENT"},
        {"gold_intent": "BILLING_PAYMENT"},
        {"gold_intent": "OTHER"},
        {"gold_intent": "PLAYBACK_APP_ISSUE"},
    ]
    predictions = [
        {"predicted_intent": "BILLING_PAYMENT"},
        {"predicted_intent": "OTHER"},
        {"predicted_intent": "OTHER"},
        {"predicted_intent": "PLAYBACK_APP_ISSUE"},
    ]

    metrics = evaluate_intents(golden, predictions)
    assert "accuracy" in metrics
    assert "macro_f1" in metrics
    assert "per_class" in metrics
    assert metrics["accuracy"] == 0.75  # 3/4 correct


def test_escalation_metrics():
    """Test escalation evaluation metrics."""
    golden = [
        {"gold_escalate": True},
        {"gold_escalate": True},
        {"gold_escalate": False},
        {"gold_escalate": False},
    ]
    predictions = [
        {"predicted_decision": "ESCALATE"},
        {"predicted_decision": "AUTO_HANDLE"},  # Unsafe
        {"predicted_decision": "AUTO_HANDLE"},
        {"predicted_decision": "ESCALATE"},  # Over-escalation
    ]

    metrics = evaluate_escalation(golden, predictions)
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1" in metrics
    assert "unsafe_auto_handle_rate" in metrics
    assert metrics["unsafe_auto_handles"] == 1


if __name__ == "__main__":
    test_intent_metrics()
    test_escalation_metrics()
    print("All metrics tests passed!")
