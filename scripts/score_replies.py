"""Generate human scoring template for reply quality."""
import csv
import json
import random
import sys
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
PREDICTIONS_FILE = PROJECT_ROOT / "outputs" / "predictions.csv"
OUTPUT_FILE = PROJECT_ROOT / "data" / "golden" / "human_reply_scores.csv"


def main(sample_size: int = 50, seed: int = 42):
    """Generate human scoring template from predictions."""
    import pandas as pd

    if not PREDICTIONS_FILE.exists():
        print(f"ERROR: {PREDICTIONS_FILE} not found. Run evaluation first.")
        return

    pred_df = pd.read_csv(PREDICTIONS_FILE)
    predictions = pred_df.to_dict('records')

    # Load golden data for context
    golden_file = PROJECT_ROOT / "data" / "golden" / "golden_set.csv"
    if golden_file.exists():
        golden_df = pd.read_csv(golden_file)
        golden = golden_df.to_dict('records')
    else:
        golden = [{} for _ in predictions]

    # Sample
    if len(predictions) > sample_size:
        indices = random.sample(range(len(predictions)), sample_size)
    else:
        indices = list(range(len(predictions)))

    # Create template
    rows = []
    for idx in indices:
        pred = predictions[idx]
        gold = golden[idx] if idx < len(golden) else {}

        rows.append({
            "example_id": f"example_{idx:04d}",
            "customer_text": gold.get("customer_text", ""),
            "generated_reply": pred.get("generated_reply", ""),
            "correctness": "",
            "relevance": "",
            "groundedness": "",
            "helpfulness": "",
            "brand_consistency": "",
            "safety": "",
            "notes": ""
        })

    # Save template
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Generated human scoring template with {len(rows)} examples")
    print(f"Saved to {OUTPUT_FILE}")
    print("\nInstructions:")
    print("1. Open the CSV in Excel or a text editor")
    print("2. For each row, score the 'generated_reply' on dimensions 1-5")
    print("3. Fill in all dimension columns")
    print("4. Save the file")
    print("5. Run: python -m eval.judge_agreement")


if __name__ == "__main__":
    import sys
    size = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    main(sample_size=size)
