"""Tests for conversation reconstruction."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.conversations import (
    parse_response_ids, reconstruct_conversations, flatten_to_cases, split_cases
)
import pandas as pd


def test_parse_response_ids():
    """Test parsing of response_tweet_id field."""
    assert parse_response_ids(None) == []
    assert parse_response_ids(float('nan')) == []
    assert parse_response_ids("123") == [123]
    assert parse_response_ids("123,456,789") == [123, 456, 789]
    assert parse_response_ids("") == []
    assert parse_response_ids("abc") == []


def test_reconstruct_conversations_basic():
    """Test basic conversation reconstruction."""
    data = {
        "tweet_id": [1, 2, 3, 4, 5],
        "author_id": ["user1", "SpotifyCares", "user1", "SpotifyCares", "user2"],
        "inbound": [True, False, True, False, True],
        "created_at": ["Mon", "Mon", "Mon", "Mon", "Mon"],
        "text": ["Help me", "Sure!", "Thanks", "You're welcome", "Hello"],
        "response_tweet_id": [2, None, 4, None, None],
        "in_response_to_tweet_id": [None, 1, None, 3, None]
    }
    df = pd.DataFrame(data)

    convos = reconstruct_conversations(df, brand="SpotifyCares")
    assert len(convos) >= 1

    # Check structure
    for conv in convos:
        assert "conversation_id" in conv
        assert "turns" in conv
        assert len(conv["turns"]) >= 2


def test_flatten_to_cases():
    """Test flattening conversations to cases."""
    conversations = [
        {
            "conversation_id": "test_1",
            "turns": [
                {"tweet_id": 1, "author_id": "user1", "text": "Help", "inbound": True},
                {"tweet_id": 2, "author_id": "SpotifyCares", "text": "Sure!", "inbound": False}
            ]
        }
    ]

    cases = flatten_to_cases(conversations)
    assert len(cases) == 1
    assert cases[0]["customer_text"] == "Help"
    assert cases[0]["brand_reply"] == "Sure!"


def test_split_cases_deterministic():
    """Test that splitting is deterministic."""
    cases = [
        {"case_id": f"case_{i}", "conversation_id": f"conv_{i//2}", "customer_text": f"text_{i}"}
        for i in range(100)
    ]

    dev1, golden1 = split_cases(cases, dev_ratio=0.8, seed=42)
    dev2, golden2 = split_cases(cases, dev_ratio=0.8, seed=42)

    assert len(dev1) == len(dev2)
    assert len(golden1) == len(golden2)


def test_no_leakage():
    """Test that no conversation appears in both splits."""
    cases = [
        {"case_id": f"case_{i}", "conversation_id": f"conv_{i}", "customer_text": f"text_{i}"}
        for i in range(50)
    ]

    dev, golden = split_cases(cases, dev_ratio=0.8, seed=42)

    dev_convs = set(c["conversation_id"] for c in dev)
    golden_convs = set(c["conversation_id"] for c in golden)

    assert len(dev_convs & golden_convs) == 0, "Leakage detected!"


if __name__ == "__main__":
    test_parse_response_ids()
    test_reconstruct_conversations_basic()
    test_flatten_to_cases()
    test_split_cases_deterministic()
    test_no_leakage()
    print("All conversation tests passed!")
