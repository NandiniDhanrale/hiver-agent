"""Inspect random golden candidates for correctness."""
import json
import random
import sys
import io
from pathlib import Path

# Fix encoding for Windows console
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

sys.path.insert(0, str(Path(__file__).parent))


def main(n_samples: int = 20, seed: int = 123):
    """Sample and display random golden candidates for manual inspection."""
    import pandas as pd

    PROJECT_ROOT = Path(__file__).parent
    candidates_file = PROJECT_ROOT / "data" / "golden" / "golden_candidates.csv"

    if not candidates_file.exists():
        print(f"ERROR: {candidates_file} not found")
        return

    df = pd.read_csv(candidates_file)
    print(f"Total candidates: {len(df)}")

    # Sample random rows
    rng = random.Random(seed)
    sample_indices = rng.sample(range(len(df)), min(n_samples, len(df)))
    sample = df.iloc[sample_indices]

    print(f"\n{'='*80}")
    print(f"INSPECTING {len(sample)} RANDOM GOLDEN CANDIDATES")
    print(f"{'='*80}")

    issues = []
    for idx, (_, row) in enumerate(sample.iterrows(), 1):
        print(f"\n--- Example {idx} (row {sample_indices[idx-1]}) ---")
        print(f"Example ID: {row['example_id']}")
        print(f"Heuristic Intent: {row['heuristic_intent']}")
        print(f"Difficulty: {row['difficulty']}")
        print(f"\nCustomer Text: {str(row['customer_text'])[:200]}")
        print(f"\nBrand Reply: {str(row['brand_reply'])[:200]}")

        # Basic sanity checks
        has_issue = False
        if pd.isna(row['customer_text']) or str(row['customer_text']).strip() == "":
            print("  WARNING: Empty customer_text!")
            has_issue = True
        if pd.isna(row['brand_reply']) or str(row['brand_reply']).strip() == "":
            print("  WARNING: Empty brand_reply!")
            has_issue = True

        # Check if customer_text looks like it's from SpotifyCares
        customer_lower = str(row['customer_text']).lower()
        if any(phrase in customer_lower for phrase in [
            'premium gives you', 'spotify premium gives', 'our team',
            'we suggest', 'we recommend', 'you can visit our'
        ]):
            print("  WARNING: customer_text looks like it might be from the brand!")
            has_issue = True

        # Check if brand_reply looks like a customer complaint
        brand_lower = str(row['brand_reply']).lower()
        if any(phrase in brand_lower for phrase in [
            'this is ridiculous', 'i am furious', 'worst service',
            'i want a refund now', 'you guys suck', 'i hate'
        ]):
            print("  WARNING: brand_reply looks like it might be from a customer!")
            has_issue = True

        if has_issue:
            issues.append(sample_indices[idx-1])

    print(f"\n{'='*80}")
    if issues:
        print(f"POTENTIAL ISSUES FOUND in rows: {issues}")
        print("These rows need manual review.")
    else:
        print("ALL CHECKS PASSED - No obvious issues detected.")
    print(f"{'='*80}")


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    main(n_samples=n)
