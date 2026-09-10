"""Conversation reconstruction from Twitter data."""
import pandas as pd
import numpy as np
import logging
from typing import Optional
from .preprocess import clean_text

logger = logging.getLogger(__name__)


def load_raw_data(csv_path: str, nrows: Optional[int] = None) -> pd.DataFrame:
    """Load the raw twcs.csv dataset."""
    logger.info(f"Loading data from {csv_path}" + (f" (first {nrows} rows)" if nrows else ""))
    df = pd.read_csv(csv_path, nrows=nrows)
    logger.info(f"Loaded {len(df)} rows")
    return df


def build_tweet_map(df: pd.DataFrame) -> dict:
    """Build a tweet_id -> row mapping for fast lookups."""
    tweet_map = {}
    for _, row in df.iterrows():
        tid = row["tweet_id"]
        tweet_map[tid] = row.to_dict()
    return tweet_map


def parse_response_ids(val) -> list[int]:
    """Parse response_tweet_id field which can be comma-separated."""
    if pd.isna(val):
        return []
    try:
        val_str = str(val).strip()
        if ',' in val_str:
            return [int(x.strip()) for x in val_str.split(',') if x.strip().isdigit()]
        elif val_str.isdigit():
            return [int(val_str)]
    except (ValueError, TypeError):
        pass
    return []


def reconstruct_conversations(df: pd.DataFrame, brand: str = "SpotifyCares") -> list[dict]:
    """Reconstruct conversations using tweet_id, response_tweet_id, and in_response_to_tweet_id.

    Returns list of conversation dicts with conversation_id and turns.
    """
    tweet_map = df.set_index("tweet_id").to_dict("index")

    # Filter brand replies
    brand_replies = df[(df["author_id"] == brand) & (df["inbound"] == False)]
    logger.info(f"Found {len(brand_replies)} {brand} outgoing replies")

    conversations = []
    processed = 0

    for _, brand_row in brand_replies.iterrows():
        in_resp = brand_row["in_response_to_tweet_id"]
        if pd.isna(in_resp):
            continue

        try:
            parent_id = int(float(in_resp))
        except (ValueError, TypeError):
            continue

        if parent_id not in tweet_map:
            continue

        parent = tweet_map[parent_id]
        if parent["author_id"] == brand:
            continue  # Brand replied to brand

        # Build thread by following the chain
        thread = []
        current_id = parent_id
        visited = set()

        while current_id in tweet_map and current_id not in visited:
            visited.add(current_id)
            current = tweet_map[current_id]
            thread.insert(0, {
                "tweet_id": current_id,
                "author_id": current["author_id"],
                "text": clean_text(str(current["text"])) if pd.notna(current["text"]) else "",
                "inbound": current["inbound"],
                "created_at": current.get("created_at", "")
            })

            if current["author_id"] == brand:
                break

            in_resp_parent = current["in_response_to_tweet_id"]
            if pd.isna(in_resp_parent):
                break
            try:
                current_id = int(float(in_resp_parent))
            except (ValueError, TypeError):
                break

        # Add the brand reply
        thread.append({
            "tweet_id": brand_row["tweet_id"],
            "author_id": brand,
            "text": clean_text(str(brand_row["text"])) if pd.notna(brand_row["text"]) else "",
            "inbound": False,
            "created_at": brand_row.get("created_at", "")
        })

        if len(thread) >= 2:
            conv_id = f"{brand.lower()}_{thread[0]['tweet_id']}"
            conversations.append({
                "conversation_id": conv_id,
                "turns": thread
            })

        processed += 1
        if processed % 5000 == 0:
            logger.info(f"Processed {processed} conversations...")

    logger.info(f"Reconstructed {len(conversations)} conversations from {brand}")
    return conversations


def flatten_to_cases(conversations: list[dict], brand: str = "SpotifyCares") -> list[dict]:
    """Flatten conversations to customer-brand case pairs.

    Only creates a case when the current turn is an inbound customer message
    and the next turn is an outbound brand reply. Never creates reversed pairs.
    """
    cases = []
    for conv in conversations:
        turns = conv["turns"]
        for i in range(len(turns) - 1):
            current = turns[i]
            next_turn = turns[i + 1]

            # Only create case: customer inbound -> brand outbound
            is_customer_to_brand = (
                current["author_id"] != brand
                and current.get("inbound", True) is True
                and next_turn["author_id"] == brand
                and next_turn.get("inbound", False) is False
            )

            if not is_customer_to_brand:
                continue

            # Build context from prior customer messages (excluding brand turns)
            context_parts = []
            for j in range(i):
                if turns[j]["author_id"] != brand:
                    context_parts.append(turns[j]["text"])
            context = " ".join(context_parts)

            cases.append({
                "case_id": f"{conv['conversation_id']}_{i}",
                "conversation_id": conv["conversation_id"],
                "customer_text": current["text"],
                "conversation_context": context,
                "brand_reply": next_turn["text"],
                "customer_author_id": current["author_id"],
                "brand": brand,
                "created_at": current.get("created_at", ""),
                "num_turns": len(turns)
            })

    logger.info(f"Flattened to {len(cases)} cases (customer -> {brand} only)")
    return cases


def split_cases(cases: list[dict], dev_ratio: float = 0.85, seed: int = 42) -> tuple[list[dict], list[dict]]:
    """Split cases into development and golden pools at conversation level.

    Returns (dev_cases, golden_candidates)
    """
    # Group by conversation_id
    conv_groups = {}
    for case in cases:
        cid = case["conversation_id"]
        if cid not in conv_groups:
            conv_groups[cid] = []
        conv_groups[cid].append(case)

    # Deterministic shuffle
    rng = np.random.RandomState(seed)
    conv_ids = list(conv_groups.keys())
    rng.shuffle(conv_ids)

    split_idx = int(len(conv_ids) * dev_ratio)
    dev_conv_ids = set(conv_ids[:split_idx])
    golden_conv_ids = set(conv_ids[split_idx:])

    dev_cases = [c for c in cases if c["conversation_id"] in dev_conv_ids]
    golden_candidates = [c for c in cases if c["conversation_id"] in golden_conv_ids]

    logger.info(f"Split: {len(dev_cases)} dev cases, {len(golden_candidates)} golden candidates")
    return dev_cases, golden_candidates
