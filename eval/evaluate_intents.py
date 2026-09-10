"""Evaluate intent classification."""
import json
import logging
from collections import Counter
from typing import Optional

logger = logging.getLogger(__name__)


def evaluate_intents(golden: list[dict], predictions: list[dict]) -> dict:
    """Compute intent classification metrics.

    Returns dict with accuracy, macro F1, weighted F1, per-class metrics,
    and confusion matrix.
    """
    from sklearn.metrics import (
        accuracy_score, precision_recall_fscore_support,
        confusion_matrix, classification_report
    )

    gold_intents = [g.get("gold_intent", "OTHER") for g in golden]
    pred_intents = [p.get("predicted_intent", "OTHER") for p in predictions]

    # Accuracy
    accuracy = accuracy_score(gold_intents, pred_intents)

    # Per-class metrics
    all_intents = sorted(set(gold_intents + pred_intents))

    precision, recall, f1, support = precision_recall_fscore_support(
        gold_intents, pred_intents, labels=all_intents, average=None, zero_division=0
    )

    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        gold_intents, pred_intents, average='macro', zero_division=0
    )

    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        gold_intents, pred_intents, average='weighted', zero_division=0
    )

    # Per-class metrics
    per_class = {}
    for i, intent in enumerate(all_intents):
        per_class[intent] = {
            "precision": round(float(precision[i]), 4),
            "recall": round(float(recall[i]), 4),
            "f1": round(float(f1[i]), 4),
            "support": int(support[i])
        }

    # Confusion matrix
    cm = confusion_matrix(gold_intents, pred_intents, labels=all_intents)
    cm_list = cm.tolist()

    return {
        "accuracy": round(float(accuracy), 4),
        "macro_precision": round(float(macro_precision), 4),
        "macro_recall": round(float(macro_recall), 4),
        "macro_f1": round(float(macro_f1), 4),
        "weighted_precision": round(float(weighted_precision), 4),
        "weighted_recall": round(float(weighted_recall), 4),
        "weighted_f1": round(float(weighted_f1), 4),
        "per_class": per_class,
        "confusion_matrix": {
            "labels": all_intents,
            "matrix": cm_list
        },
        "num_examples": len(golden),
        "headline_metric": "macro_f1"
    }
