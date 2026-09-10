"""LLM abstraction layer."""
import os
import hashlib
import json
import logging
import time
from typing import Optional
from pathlib import Path

logger = logging.getLogger(__name__)

# Simple in-memory cache for LLM calls
_llm_cache: dict[str, str] = {}
_cache_file = Path(__file__).parent.parent / "outputs" / "llm_cache.json"


def _cache_key(prompt: str, model: str) -> str:
    """Generate a deterministic cache key."""
    content = f"{model}:{prompt}"
    return hashlib.sha256(content.encode()).hexdigest()[:16]


def load_cache():
    """Load LLM cache from disk."""
    global _llm_cache
    if _cache_file.exists():
        try:
            with open(_cache_file, 'r') as f:
                _llm_cache = json.load(f)
            logger.info(f"Loaded {len(_llm_cache)} cached LLM responses")
        except Exception:
            _llm_cache = {}


def save_cache():
    """Save LLM cache to disk."""
    try:
        _cache_file.parent.mkdir(parents=True, exist_ok=True)
        with open(_cache_file, 'w') as f:
            json.dump(_llm_cache, f, indent=2)
    except Exception as e:
        logger.warning(f"Failed to save LLM cache: {e}")


def llm_call(prompt: str, model: Optional[str] = None, temperature: float = 0.0,
             use_cache: bool = True) -> str:
    """Make an LLM API call with caching.

    Supports OpenAI-compatible APIs. Configure via environment variables:
    - LLM_API_KEY: API key
    - LLM_BASE_URL: API base URL (optional)
    - LLM_MODEL: Model name (default: gpt-4o-mini)
    """
    model = model or os.environ.get("LLM_MODEL", "gpt-4o-mini")
    api_key = os.environ.get("LLM_API_KEY", "")
    base_url = os.environ.get("LLM_BASE_URL", "")

    if not api_key:
        logger.warning("No LLM_API_KEY set, using rule-based fallback")
        return _rule_based_fallback(prompt)

    # Check cache
    if use_cache:
        key = _cache_key(prompt, model)
        if key in _llm_cache:
            return _llm_cache[key]

    try:
        import requests

        url = f"{base_url}/chat/completions" if base_url else "https://api.openai.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": 1024
        }

        response = requests.post(url, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        result = response.json()["choices"][0]["message"]["content"]

        # Cache the result
        if use_cache:
            _llm_cache[_cache_key(prompt, model)] = result
            save_cache()

        return result
    except Exception as e:
        logger.error(f"LLM call failed: {e}")
        return _rule_based_fallback(prompt)


def _rule_based_fallback(prompt: str) -> str:
    """Simple rule-based fallback when LLM is unavailable."""
    prompt_lower = prompt.lower()

    # Intent classification fallback
    if "classify" in prompt_lower and "intent" in prompt_lower:
        return '{"intent": "OTHER", "confidence": 0.3}'

    # Reply generation fallback
    if "draft" in prompt_lower or "reply" in prompt_lower or "respond" in prompt_lower:
        return "Thank you for reaching out. We're looking into your issue and will get back to you shortly. If you need immediate assistance, please visit our help center at support.spotify.com."

    return "I understand your concern. Let me look into this for you."
