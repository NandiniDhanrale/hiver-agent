"""Tests for retrieval system."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.retrieval import RetrievalIndex


def test_retrieval_basic():
    """Test basic retrieval functionality."""
    cases = [
        {"case_id": "c1", "customer_text": "I can't login to my account", "brand_reply": "Try resetting password"},
        {"case_id": "c2", "customer_text": "Music keeps stopping", "brand_reply": "Check your connection"},
        {"case_id": "c3", "customer_text": "I was charged twice", "brand_reply": "Contact billing support"},
    ]

    index = RetrievalIndex(model_name="sentence-transformers/all-MiniLM-L6-v2")
    index.build_index(cases)

    results = index.retrieve("my login is broken", top_k=2)
    assert len(results) == 2
    assert all("similarity" in r for r in results)
    assert all("case_id" in r for r in results)


def test_retrieval_empty():
    """Test retrieval with empty index."""
    index = RetrievalIndex()
    index.build_index([])

    results = index.retrieve("test", top_k=5)
    assert len(results) == 0


def test_retrieval_top_k():
    """Test that top_k parameter works."""
    cases = [
        {"case_id": f"c{i}", "customer_text": f"Issue number {i}", "brand_reply": f"Reply {i}"}
        for i in range(20)
    ]

    index = RetrievalIndex(model_name="sentence-transformers/all-MiniLM-L6-v2")
    index.build_index(cases)

    results = index.retrieve("issue", top_k=3)
    assert len(results) == 3


if __name__ == "__main__":
    test_retrieval_basic()
    test_retrieval_empty()
    test_retrieval_top_k()
    print("All retrieval tests passed!")
