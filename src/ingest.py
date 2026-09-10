"""Data ingestion pipeline."""
import pandas as pd
import logging
from pathlib import Path
from .conversations import (
    load_raw_data, reconstruct_conversations, flatten_to_cases, split_cases
)

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent


def ingest_and_prepare(csv_path: str = None, brand: str = "SpotifyCares",
                        output_dir: str = None, nrows: int = None) -> dict:
    """Full ingestion pipeline.

    1. Load raw CSV
    2. Reconstruct conversations
    3. Flatten to cases
    4. Split dev/golden
    5. Save processed data

    Returns stats dict.
    """
    if csv_path is None:
        csv_path = str(PROJECT_ROOT / "data" / "raw" / "twcs.csv")
    if output_dir is None:
        output_dir = str(PROJECT_ROOT / "data" / "processed")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load raw data
    df = load_raw_data(csv_path, nrows=nrows)
    total_rows = len(df)

    # Reconstruct conversations
    conversations = reconstruct_conversations(df, brand=brand)
    num_conversations = len(conversations)

    # Flatten to cases
    cases = flatten_to_cases(conversations)
    num_cases = len(cases)

    # Split
    dev_cases, golden_candidates = split_cases(cases)

    # Save
    import json

    with open(output_path / "dev_cases.json", 'w') as f:
        json.dump(dev_cases, f, indent=2)
    logger.info(f"Saved {len(dev_cases)} dev cases")

    with open(output_path / "golden_candidates.json", 'w') as f:
        json.dump(golden_candidates, f, indent=2)
    logger.info(f"Saved {len(golden_candidates)} golden candidates")

    with open(output_path / "conversations.json", 'w') as f:
        json.dump(conversations[:100], f, indent=2)  # Save sample

    # Compute stats
    # Check for broken relationships
    brand_reply_count = len(df[(df["author_id"] == brand) & (df["inbound"] == False)])
    broken_count = brand_reply_count - num_conversations

    stats = {
        "total_rows": total_rows,
        "brand": brand,
        "brand_outgoing_replies": brand_reply_count,
        "reconstructed_conversations": num_conversations,
        "flattened_cases": num_cases,
        "dev_cases": len(dev_cases),
        "golden_candidates": len(golden_candidates),
        "broken_relationships": broken_count,
        "broken_pct": round(broken_count / max(brand_reply_count, 1) * 100, 2)
    }

    logger.info(f"Stats: {json.dumps(stats, indent=2)}")
    return stats
