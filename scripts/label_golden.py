"""Terminal-based golden set labeling tool."""
import csv
import json
import sys
import os
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
CANDIDATES_FILE = PROJECT_ROOT / "data" / "golden" / "golden_candidates.csv"
OUTPUT_FILE = PROJECT_ROOT / "data" / "golden" / "golden_set.csv"

INTENTS = [
    "LOGIN_ACCOUNT_ACCESS",
    "BILLING_PAYMENT",
    "SUBSCRIPTION_PLAN",
    "PLAYBACK_APP_ISSUE",
    "FAMILY_STUDENT_PLAN",
    "ACCOUNT_SECURITY",
    "FEATURE_REQUEST",
    "OTHER"
]

INTENT_DESCRIPTIONS = {
    "LOGIN_ACCOUNT_ACCESS": "Login problems, password reset, account locked",
    "BILLING_PAYMENT": "Unexpected charges, failed payments, refunds",
    "SUBSCRIPTION_PLAN": "Premium/Free plans, upgrading, canceling",
    "PLAYBACK_APP_ISSUE": "App crashes, playback bugs, streaming errors",
    "FAMILY_STUDENT_PLAN": "Family plan, student verification",
    "ACCOUNT_SECURITY": "Hacked account, suspicious activity",
    "FEATURE_REQUEST": "Suggestions for new features",
    "OTHER": "General feedback, praise, off-topic"
}


def load_candidates() -> list[dict]:
    """Load golden candidates."""
    if not CANDIDATES_FILE.exists():
        print(f"ERROR: {CANDIDATES_FILE} not found.")
        print("Run: python -m scripts.create_golden_candidates")
        sys.exit(1)

    df = pd.read_csv(CANDIDATES_FILE)
    return df.to_dict('records')


def save_progress(candidates: list[dict]):
    """Save labeled progress."""
    import pandas as pd
    df = pd.DataFrame(candidates)
    df.to_csv(OUTPUT_FILE, index=False)


def main():
    """Interactive labeling tool."""
    import pandas as pd

    candidates = load_candidates()
    print(f"Loaded {len(candidates)} candidates")

    # Check for already labeled
    if OUTPUT_FILE.exists():
        existing = pd.read_csv(OUTPUT_FILE)
        labeled_ids = set(existing[existing["human_verified"] == True]["example_id"])
        print(f"Already labeled: {len(labeled_ids)}")
    else:
        labeled_ids = set()

    # Find next unlabeled
    remaining = [c for c in candidates if c["example_id"] not in labeled_ids]
    print(f"Remaining to label: {len(remaining)}")
    print()

    if not remaining:
        print("All examples labeled!")
        return

    for i, candidate in enumerate(remaining):
        print(f"\n{'='*60}")
        print(f"Example {i+1}/{len(remaining)}: {candidate['example_id']}")
        print(f"Difficulty: {candidate['difficulty']}")
        print(f"Heuristic intent: {candidate['heuristic_intent']}")
        print(f"\nCustomer: {candidate['customer_text']}")
        if candidate.get('conversation_context'):
            print(f"Context: {candidate['conversation_context']}")
        print(f"\nHistorical reply: {candidate.get('brand_reply', 'N/A')[:200]}")

        print(f"\nIntent options:")
        for j, intent in enumerate(INTENTS, 1):
            desc = INTENT_DESCRIPTIONS[intent]
            print(f"  {j}. {intent}: {desc}")

        intent_choice = input(f"\nSelect intent (1-{len(INTENTS)}): ").strip()
        try:
            intent_idx = int(intent_choice) - 1
            gold_intent = INTENTS[intent_idx]
        except (ValueError, IndexError):
            print("Invalid choice, skipping...")
            continue

        escalate_input = input("Escalate? (y/n): ").strip().lower()
        gold_escalate = escalate_input in ('y', 'yes', 'true', '1')

        reason = input("Escalation reason (or Enter to skip): ").strip()
        notes = input("Label notes (or Enter to skip): ").strip()

        # Update candidate
        candidate["gold_intent"] = gold_intent
        candidate["gold_escalate"] = gold_escalate
        candidate["gold_escalation_reason"] = reason
        candidate["label_notes"] = notes
        candidate["human_verified"] = True

        # Update in main list
        for c in candidates:
            if c["example_id"] == candidate["example_id"]:
                c.update(candidate)
                break

        # Save progress
        save_progress(candidates)
        print(f"Saved. ({len([c for c in candidates if c['human_verified']])} labeled)")

        if input("\nContinue? (y/n): ").strip().lower() not in ('y', 'yes', ''):
            break

    total_labeled = sum(1 for c in candidates if c["human_verified"])
    print(f"\nTotal labeled: {total_labeled}/{len(candidates)}")


if __name__ == "__main__":
    import pandas as pd
    main()
