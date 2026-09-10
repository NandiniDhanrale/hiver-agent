"""Human vs LLM judge agreement evaluation."""
import json
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

HUMAN_SCORES_FILE = Path(__file__).parent.parent / "data" / "golden" / "human_reply_scores.csv"


def load_human_scores() -> list[dict]:
    """Load human judge scores from CSV."""
    import pandas as pd

    if not HUMAN_SCORES_FILE.exists():
        logger.warning(f"Human scores file not found: {HUMAN_SCORES_FILE}")
        return []

    df = pd.read_csv(HUMAN_SCORES_FILE)
    return df.to_dict('records')


def compute_agreement(human_scores: list[dict], llm_scores: list[dict]) -> dict:
    """Compute agreement metrics between human and LLM judges.

    Returns exact agreement, ±1 agreement, Spearman correlation, and kappa.
    """
    if not human_scores or not llm_scores:
        return {"status": "PENDING HUMAN VALIDATION", "human_examples_scored": 0}

    dimensions = ["correctness", "relevance", "groundedness",
                  "helpfulness", "brand_consistency", "safety"]

    results = {
        "per_dimension": {},
        "aggregate": {},
        "human_examples_scored": len(human_scores),
        "status": "AVAILABLE"
    }

    all_human = []
    all_llm = []

    for dim in dimensions:
        human_vals = [h.get(dim, 3) for h in human_scores]
        llm_vals = [l.get(dim, 3) for l in llm_scores[:len(human_scores)]]

        # Exact agreement
        exact_matches = sum(1 for h, l in zip(human_vals, llm_vals) if h == l)
        exact_agreement = exact_matches / max(len(human_vals), 1)

        # ±1 agreement
        close_matches = sum(1 for h, l in zip(human_vals, llm_vals) if abs(h - l) <= 1)
        close_agreement = close_matches / max(len(human_vals), 1)

        # Spearman correlation
        try:
            from scipy.stats import spearmanr
            corr, pval = spearmanr(human_vals, llm_vals)
            spearman = float(corr) if not (corr != corr) else 0.0  # Check for NaN
        except ImportError:
            spearman = 0.0
            pval = 1.0

        # Weighted Cohen's kappa
        try:
            from sklearn.metrics import cohen_kappa_score
            kappa = float(cohen_kappa_score(human_vals, llm_vals, weights='quadratic'))
        except ImportError:
            kappa = 0.0

        results["per_dimension"][dim] = {
            "exact_agreement": round(exact_agreement, 4),
            "close_agreement": round(close_agreement, 4),
            "spearman_correlation": round(spearman, 4),
            "weighted_kappa": round(kappa, 4)
        }

        all_human.extend(human_vals)
        all_llm.extend(llm_vals)

    # Aggregate metrics
    if all_human:
        exact_matches = sum(1 for h, l in zip(all_human, all_llm) if h == l)
        close_matches = sum(1 for h, l in zip(all_human, all_llm) if abs(h - l) <= 1)

        try:
            from scipy.stats import spearmanr
            agg_corr, _ = spearmanr(all_human, all_llm)
            agg_spearman = float(agg_corr) if not (agg_corr != agg_corr) else 0.0
        except ImportError:
            agg_spearman = 0.0

        try:
            from sklearn.metrics import cohen_kappa_score
            agg_kappa = float(cohen_kappa_score(all_human, all_llm, weights='quadratic'))
        except ImportError:
            agg_kappa = 0.0

        results["aggregate"] = {
            "exact_agreement": round(exact_matches / len(all_human), 4),
            "close_agreement": round(close_matches / len(all_human), 4),
            "spearman_correlation": round(agg_spearman, 4),
            "weighted_kappa": round(agg_kappa, 4)
        }

    return results


def save_human_scores_template(sampled_responses: list[dict], output_path: str = None):
    """Save a template CSV for manual human scoring.

    This creates the file that the user fills in manually.
    """
    import csv

    if output_path is None:
        output_path = str(HUMAN_SCORES_FILE)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    headers = ["example_id", "customer_text", "generated_reply", "correctness",
               "relevance", "groundedness", "helpfulness", "brand_consistency",
               "safety", "notes"]

    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()

        for resp in sampled_responses:
            writer.writerow({
                "example_id": resp.get("example_id", ""),
                "customer_text": resp.get("customer_text", ""),
                "generated_reply": resp.get("generated_reply", ""),
                "correctness": "",
                "relevance": "",
                "groundedness": "",
                "helpfulness": "",
                "brand_consistency": "",
                "safety": "",
                "notes": ""
            })

    logger.info(f"Saved human scoring template to {output_path}")
