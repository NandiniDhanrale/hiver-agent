"""Generate golden set candidates with stratified sampling."""
import json
import random
import sys
import logging
import pandas as pd
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def heuristic_intent(text: str) -> str:
    """Assign intent heuristic for stratification."""
    text_lower = text.lower()
    if any(w in text_lower for w in ['charge', 'refund', 'bill', 'payment', 'money']):
        return "BILLING_PAYMENT"
    if any(w in text_lower for w in ['login', 'password', 'locked', "can't log", "sign in"]):
        return "LOGIN_ACCOUNT_ACCESS"
    if any(w in text_lower for w in ['crash', 'bug', 'slow', 'freeze', "doesn't work", 'error']):
        return "PLAYBACK_APP_ISSUE"
    if any(w in text_lower for w in ['premium', 'free', 'subscribe', 'subscription', 'plan', 'trial']):
        return "SUBSCRIPTION_PLAN"
    if any(w in text_lower for w in ['family', 'student', 'invite', 'member']):
        return "FAMILY_STUDENT_PLAN"
    if any(w in text_lower for w in ['hacked', 'compromised', 'security', 'unauthorized']):
        return "ACCOUNT_SECURITY"
    if any(w in text_lower for w in ['add', 'feature', 'suggestion', 'please add', 'wish']):
        return "FEATURE_REQUEST"
    return "OTHER"


def main(target_size: int = 200, seed: int = 42):
    """Generate stratified golden set candidates."""
    # Load dev cases
    dev_file = PROJECT_ROOT / "data" / "processed" / "dev_cases.json"
    if not dev_file.exists():
        print("ERROR: dev_cases.json not found. Run prepare_data.py first.")
        return

    with open(dev_file, 'r') as f:
        cases = json.load(f)

    print(f"Loaded {len(cases)} development cases")

    # Assign intents
    for case in cases:
        case["heuristic_intent"] = heuristic_intent(case["customer_text"])

    # Group by intent
    intent_groups = {}
    for case in cases:
        intent = case["heuristic_intent"]
        if intent not in intent_groups:
            intent_groups[intent] = []
        intent_groups[intent].append(case)

    print("\nIntent distribution:")
    for intent, group in sorted(intent_groups.items(), key=lambda x: -len(x[1])):
        print(f"  {intent}: {len(group)}")

    # Stratified sampling
    rng = random.Random(seed)
    candidates = []

    # Target: ~25 examples per intent (8 intents * 25 = 200)
    per_intent = target_size // len(intent_groups)

    for intent, group in intent_groups.items():
        n_samples = min(per_intent, len(group))

        # Prioritize variety
        sampled = rng.sample(group, n_samples)

        for case in sampled:
            # Determine difficulty
            text_len = len(case["customer_text"].split())
            if text_len < 5:
                difficulty = "HARD"
            elif text_len > 20:
                difficulty = "MEDIUM"
            else:
                difficulty = "EASY"

            candidates.append({
                "example_id": f"golden_{len(candidates):04d}",
                "conversation_id": case["conversation_id"],
                "customer_text": case["customer_text"],
                "conversation_context": case.get("conversation_context", ""),
                "brand_reply": case.get("brand_reply", ""),
                "heuristic_intent": case["heuristic_intent"],
                "difficulty": difficulty,
                "gold_intent": "",  # To be labeled
                "gold_escalate": None,
                "gold_escalation_reason": "",
                "reference_resolution": case.get("brand_reply", ""),
                "label_notes": "",
                "human_verified": False
            })

    # Pad to target size if needed
    remaining = [c for c in cases if c not in [x for group in intent_groups.values() for x in group]]
    while len(candidates) < target_size and remaining:
        case = rng.choice(remaining)
        remaining.remove(case)
        candidates.append({
            "example_id": f"golden_{len(candidates):04d}",
            "conversation_id": case["conversation_id"],
            "customer_text": case["customer_text"],
            "conversation_context": case.get("conversation_context", ""),
            "brand_reply": case.get("brand_reply", ""),
            "heuristic_intent": case.get("heuristic_intent", "OTHER"),
            "difficulty": "MEDIUM",
            "gold_intent": "",
            "gold_escalate": None,
            "gold_escalation_reason": "",
            "reference_resolution": case.get("brand_reply", ""),
            "label_notes": "",
            "human_verified": False
        })

    print(f"\nGenerated {len(candidates)} golden candidates")

    # Save candidates
    output_file = PROJECT_ROOT / "data" / "golden" / "golden_candidates.csv"
    df = pd.DataFrame(candidates)
    df.to_csv(output_file, index=False)
    print(f"Saved to {output_file}")

    # Print summary
    print("\nCandidate distribution:")
    intent_dist = pd.Series([c["heuristic_intent"] for c in candidates]).value_counts()
    for intent, count in intent_dist.items():
        print(f"  {intent}: {count}")


if __name__ == "__main__":
    import sys
    target = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    main(target_size=target)
