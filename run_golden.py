"""Quick script to generate golden candidates."""
import sys
import json
import random
import pandas as pd
from pathlib import Path

# Get the actual project root - this script is in hiver-support-agent/
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))


def heuristic_intent(text: str) -> str:
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


def main():
    # Load dev cases
    dev_file = PROJECT_ROOT / "data" / "processed" / "dev_cases.json"
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

    # Stratified sampling
    rng = random.Random(42)
    candidates = []
    target_size = 200
    per_intent = target_size // len(intent_groups)

    for intent, group in intent_groups.items():
        n_samples = min(per_intent, len(group))
        sampled = rng.sample(group, n_samples)

        for case in sampled:
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
                "gold_intent": "",
                "gold_escalate": None,
                "gold_escalation_reason": "",
                "reference_resolution": case.get("brand_reply", ""),
                "label_notes": "",
                "human_verified": False
            })

    print(f"Generated {len(candidates)} golden candidates")

    # Save
    output_file = PROJECT_ROOT / "data" / "golden" / "golden_candidates.csv"
    df = pd.DataFrame(candidates)
    df.to_csv(output_file, index=False)
    print(f"Saved to {output_file}")

    # Print distribution
    intent_dist = pd.Series([c["heuristic_intent"] for c in candidates]).value_counts()
    print("\nCandidate distribution:")
    for intent, count in intent_dist.items():
        print(f"  {intent}: {count}")


if __name__ == "__main__":
    main()
