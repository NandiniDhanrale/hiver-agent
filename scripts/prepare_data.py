"""Prepare the development dataset from raw twcs.csv."""
import sys
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def main():
    """Run the data preparation pipeline."""
    from src.ingest import ingest_and_prepare

    csv_path = PROJECT_ROOT / "data" / "raw" / "twcs.csv"
    if not csv_path.exists():
        print(f"ERROR: {csv_path} not found.")
        print("Download twcs.csv from Kaggle and place it in data/raw/")
        sys.exit(1)

    print("Running data preparation pipeline...")
    stats = ingest_and_prepare(str(csv_path))

    print("\n=== Data Preparation Stats ===")
    for k, v in stats.items():
        print(f"  {k}: {v}")

    # Save stats
    output_dir = PROJECT_ROOT / "data" / "processed"
    with open(output_dir / "stats.json", 'w') as f:
        json.dump(stats, f, indent=2)

    print(f"\nSaved to {output_dir}")


if __name__ == "__main__":
    main()
