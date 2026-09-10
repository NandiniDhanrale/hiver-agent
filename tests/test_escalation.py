"""Tests for escalation policy."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.escalation import check_escalation


def test_escalate_account_compromise():
    """Test escalation for account compromise."""
    should_escalate, reason, rules = check_escalation(
        intent="ACCOUNT_SECURITY",
        confidence=0.9,
        retrieved_cases=[{"similarity": 0.8}],
        reply="Please contact us",
        customer_text="My account was hacked by someone"
    )
    assert should_escalate is True
    assert any("account_compromise" in r for r in rules)


def test_escalate_payment_dispute():
    """Test escalation for payment dispute."""
    should_escalate, reason, rules = check_escalation(
        intent="BILLING_PAYMENT",
        confidence=0.8,
        retrieved_cases=[{"similarity": 0.7}],
        reply="Refund pending",
        customer_text="I want a refund for the unauthorized charge"
    )
    assert should_escalate is True
    assert any("payment_dispute" in r for r in rules)


def test_escalate_low_confidence():
    """Test escalation for low confidence."""
    should_escalate, reason, rules = check_escalation(
        intent="OTHER",
        confidence=0.3,
        retrieved_cases=[{"similarity": 0.8}],
        reply="Please clarify",
        customer_text="asdfghjkl"
    )
    assert should_escalate is True
    assert any("low_confidence" in r for r in rules)


def test_auto_handle_high_confidence():
    """Test auto-handling with high confidence."""
    should_escalate, reason, rules = check_escalation(
        intent="PLAYBACK_APP_ISSUE",
        confidence=0.9,
        retrieved_cases=[{"similarity": 0.85}],
        reply="Try restarting the app",
        customer_text="Spotify keeps crashing on my phone"
    )
    # May or may not escalate depending on keyword detection
    # But should not escalate solely due to confidence or retrieval
    assert not any("low_confidence" in r for r in rules)
    assert not any("weak_retrieval" in r for r in rules)


def test_escalate_weak_retrieval():
    """Test escalation for weak retrieval evidence."""
    should_escalate, reason, rules = check_escalation(
        intent="SUBSCRIPTION_PLAN",
        confidence=0.8,
        retrieved_cases=[{"similarity": 0.3}],
        reply="Please clarify",
        customer_text="I need help with my subscription"
    )
    assert any("weak_retrieval" in r for r in rules)


def test_escalate_no_retrieval():
    """Test escalation with no retrieval evidence."""
    should_escalate, reason, rules = check_escalation(
        intent="OTHER",
        confidence=0.7,
        retrieved_cases=[],
        reply="Please contact support",
        customer_text="Something unusual happened"
    )
    assert any("no_retrieval_evidence" in r for r in rules)


if __name__ == "__main__":
    test_escalate_account_compromise()
    test_escalate_payment_dispute()
    test_escalate_low_confidence()
    test_auto_handle_high_confidence()
    test_escalate_weak_retrieval()
    test_escalate_no_retrieval()
    print("All escalation tests passed!")
