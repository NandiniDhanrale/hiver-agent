"""Intent classification for customer support messages."""
import yaml
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

INTENTS_FILE = Path(__file__).parent.parent / "config" / "intents.yaml"


def load_intents() -> dict:
    """Load the frozen intent taxonomy from config."""
    with open(INTENTS_FILE, 'r') as f:
        config = yaml.safe_load(f)
    return config["intents"]


def get_intent_names() -> list[str]:
    """Get list of all intent names."""
    intents = load_intents()
    return list(intents.keys())


def get_intent_descriptions() -> str:
    """Get formatted intent descriptions for LLM prompts."""
    intents = load_intents()
    lines = []
    for name, info in intents.items():
        lines.append(f"- {name}: {info['description'].strip()}")
        if 'excludes' in info:
            lines.append(f"  Excludes: {'; '.join(info['excludes'])}")
    return "\n".join(lines)


def rule_based_intent_hint(text: str) -> str | None:
    """Quick rule-based intent suggestion.

    This is a heuristic hint, not the final classifier.
    Used to augment LLM classification.
    """
    text_lower = text.lower()

    # Security keywords
    security_kw = ['hacked', 'compromised', 'unauthorized', 'stolen', 'suspicious']
    if any(kw in text_lower for kw in security_kw):
        return "ACCOUNT_SECURITY"

    # Billing keywords
    billing_kw = ['charge', 'charged', 'refund', 'payment', 'bill', 'money', 'credit card']
    if any(kw in text_lower for kw in billing_kw):
        return "BILLING_PAYMENT"

    # Family/Student keywords
    family_kw = ['family', 'student', 'invite', 'member', 'household']
    if any(kw in text_lower for kw in family_kw):
        return "FAMILY_STUDENT_PLAN"

    # Login keywords
    login_kw = ['login', 'log in', 'sign in', 'password', 'forgot', 'locked out', "can't access"]
    if any(kw in text_lower for kw in login_kw):
        return "LOGIN_ACCOUNT_ACCESS"

    # Playback keywords
    playback_kw = ['crash', 'crashing', 'buffer', 'lag', 'slow', 'freeze', 'stuck', "doesn't play", 'audio']
    if any(kw in text_lower for kw in playback_kw):
        return "PLAYBACK_APP_ISSUE"

    # Subscription keywords
    sub_kw = ['premium', 'free', 'upgrade', 'downgrade', 'subscribe', 'subscription', 'trial']
    if any(kw in text_lower for kw in sub_kw):
        return "SUBSCRIPTION_PLAN"

    # Feature request keywords
    feature_kw = ['feature', 'suggestion', 'add', 'please add', 'wish', 'would be great']
    if any(kw in text_lower for kw in feature_kw):
        return "FEATURE_REQUEST"

    return None


def classify_intent_llm(customer_text: str, context: str, intent_descriptions: str,
                         llm_call_fn) -> tuple[str, float]:
    """Classify intent using LLM with rule-based augmentation.

    Returns (intent, confidence)
    """
    # Get rule-based hint
    hint = rule_based_intent_hint(customer_text)

    hint_section = ""
    if hint:
        hint_section = f"\nA quick keyword analysis suggests: {hint}. Verify if this matches the message."

    prompt = f"""Classify this customer support message into exactly ONE intent category.

INTENTS:
{intent_descriptions}
{hint_section}

CUSTOMER MESSAGE: {customer_text}
{f'CONVERSATION CONTEXT: {context}' if context else ''}

Respond with ONLY a JSON object:
{{"intent": "INTENT_NAME", "confidence": 0.0-1.0}}"""

    try:
        response = llm_call_fn(prompt)
        import json
        # Try to parse JSON from response
        response = response.strip()
        if response.startswith("```"):
            response = response.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        result = json.loads(response)
        intent = result.get("intent", "OTHER")
        confidence = float(result.get("confidence", 0.5))

        # Validate intent name
        valid_intents = get_intent_names()
        if intent not in valid_intents:
            intent = "OTHER"
            confidence = 0.3

        return intent, min(max(confidence, 0.0), 1.0)
    except Exception as e:
        logger.warning(f"LLM classification failed: {e}, using rule-based fallback")
        if hint:
            return hint, 0.5
        return "OTHER", 0.3
