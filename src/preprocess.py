"""Text preprocessing utilities."""
import re
import html


def clean_text(text: str) -> str:
    """Clean Twitter text conservatively.

    Removes:
    - HTML entities (&amp; -> &)
    - Excessive whitespace
    - Empty tweets

    Preserves:
    - Mentions (@user)
    - URLs
    - Hashtags
    - Emojis
    - Case and punctuation
    """
    if not isinstance(text, str):
        return ""

    # Decode HTML entities
    text = html.unescape(text)

    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def normalize_mention(text: str, brand: str = "SpotifyCares") -> str:
    """Remove the brand mention from the start of customer messages.

    Customer tweets often start with @SpotifyCares which doesn't add
    meaning for the agent.
    """
    if not isinstance(text, str):
        return text

    # Remove leading @Brand mentions
    pattern = rf'^\s*@{re.escape(brand)}\s*'
    text = re.sub(pattern, '', text)

    return text.strip()
