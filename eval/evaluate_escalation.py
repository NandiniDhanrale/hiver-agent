"""Evaluate escalation decisions."""
import logging

logger = logging.getLogger(__name__)


def evaluate_escalation(golden: list[dict], predictions: list[dict]) -> dict:
    """Compute escalation metrics.

    Returns dict with accuracy, precision, recall, F1, and unsafe auto-handle rate.
    """
    from sklearn.metrics import (
        accuracy_score, precision_recall_fscore_support,
        confusion_matrix
    )

    gold_escalate = [g.get("gold_escalate", False) for g in golden]
    pred_escalate = [p.get("predicted_decision", "ESCALATE") == "ESCALATE" for p in predictions]

    # Basic metrics
    accuracy = accuracy_score(gold_escalate, pred_escalate)

    # Treat ESCALATE as positive class (1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        gold_escalate, pred_escalate, average='binary', pos_label=True, zero_division=0
    )

    # Confusion matrix
    cm = confusion_matrix(gold_escalate, pred_escalate, labels=[False, True])

    # Unsafe auto-handle rate
    # Gold=ESCALATE but predicted=AUTO_HANDLE
    gold_escalate_count = sum(gold_escalate)
    unsafe_auto_handles = sum(1 for g, p in zip(gold_escalate, pred_escalate)
                              if g and not p)
    unsafe_rate = unsafe_auto_handles / max(gold_escalate_count, 1)

    # Auto-handle rate
    auto_handle_count = sum(1 for p in pred_escalate if not p)
    auto_handle_rate = auto_handle_count / max(len(pred_escalate), 1)

    return {
        "accuracy": round(float(accuracy), 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1": round(float(f1), 4),
        "unsafe_auto_handle_rate": round(float(unsafe_rate), 4),
        "auto_handle_rate": round(float(auto_handle_rate), 4),
        "confusion_matrix": {
            "labels": ["AUTO_HANDLE", "ESCALATE"],
            "matrix": cm.tolist()
        },
        "gold_escalate_count": int(gold_escalate_count),
        "predicted_escalate_count": int(sum(pred_escalate)),
        "unsafe_auto_handles": int(unsafe_auto_handles),
        "headline_metric": "f1"
    }
