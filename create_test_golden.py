"""Create a test golden set with heuristic labels for pipeline testing."""
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

PROJECT_ROOT = Path(__file__).parent


def main():
    """Create golden_set.csv from candidates with heuristic labels."""
    candidates_file = PROJECT_ROOT / "data" / "golden" / "golden_candidates.csv"
    output_file = PROJECT_ROOT / "data" / "golden" / "golden_set.csv"

    if not candidates_file.exists():
        print(f"ERROR: {candidates_file} not found")
        return

    df = pd.read_csv(candidates_file)
    print(f"Loaded {len(df)} candidates")

    # Use heuristic intents as gold intents (for pipeline testing only)
    df["gold_intent"] = df["heuristic_intent"]

    # Simple escalation heuristic for testing
    def should_escalate(row):
        text = str(row["customer_text"]).lower()
        if any(w in text for w in ['hacked', 'compromised', 'security', 'unauthorized', 'stolen']):
            return True
        if any(w in text for w in ['refund', 'charged', 'charge', 'dispute', 'money back']):
            return True
        if any(w in text for w in ['lawyer', 'sue', 'legal', 'threaten']):
            return True
        # Low confidence heuristic
        if row.get("difficulty") == "HARD":
            return True
        return False

    df["gold_escalate"] = df.apply(should_escalate, axis=1)

    # Generate escalation reasons
    df["gold_escalation_reason"] = df.apply(
        lambda r: "Security concern" if r["gold_escalate"] and "hack" in str(r["customer_text"]).lower()
        else "Payment dispute" if r["gold_escalate"] and any(w in str(r["customer_text"]).lower() for w in ['charge', 'refund'])
        else "High difficulty" if r["gold_escalate"]
        else "",
        axis=1
    )

    df["reference_resolution"] = df["brand_reply"]
    df["label_notes"] = "HEURISTIC LABELS - NOT HUMAN VERIFIED"
    df["human_verified"] = False

    df.to_csv(output_file, index=False)
    print(f"Saved golden set to {output_file}")
    print(f"Gold intent distribution:")
    print(df["gold_intent"].value_counts().to_string())
    print(f"\nGold escalate distribution:")
    print(df["gold_escalate"].value_counts().to_string())


if __name__ == "__main__":
    main()
