"""Tests for text preprocessing."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocess import clean_text, normalize_mention


def test_clean_text_html():
    """Test HTML entity decoding."""
    assert clean_text("Hello &amp; world") == "Hello & world"
    assert clean_text("Price: &lt;$10") == "Price: <$10"


def test_clean_text_whitespace():
    """Test whitespace normalization."""
    assert clean_text("  hello   world  ") == "hello world"
    assert clean_text("hello\n\nworld") == "hello world"


def test_clean_text_empty():
    """Test empty/None handling."""
    assert clean_text("") == ""
    assert clean_text(None) == ""


def test_normalize_mention():
    """Test brand mention removal."""
    result = normalize_mention("@SpotifyCares Help me please")
    assert result == "Help me please"

    result = normalize_mention("  @SpotifyCares  Hello")
    assert result == "Hello"

    result = normalize_mention("No mention here")
    assert result == "No mention here"


if __name__ == "__main__":
    test_clean_text_html()
    test_clean_text_whitespace()
    test_clean_text_empty()
    test_normalize_mention()
    print("All preprocessing tests passed!")
