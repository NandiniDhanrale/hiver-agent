"""Quick evaluation without LLM API key using TF-IDF baseline."""
import json
import time
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

PROJECT_ROOT = Path(__file__).parent.parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


def main():
    start_time = time.time()

    # Load golden set
    golden_file = PROJECT_ROOT / "data" / "golden" / "golden_set.csv"
    if not golden_file.exists():
        print("ERROR: Golden set not found")
        return

    golden_df = pd.read_csv(golden_file)
    golden = golden_df.to_dict('records')
    print(f"Loaded {len(golden)} golden examples")

    # Load dev cases
    dev_file = PROJECT_ROOT / "data" / "processed" / "dev_cases.json"
    with open(dev_file, 'r') as f:
        dev_cases = json.load(f)

    # Use TF-IDF baseline (fast, no API needed)
    from baselines.tfidf import TFIDFBaseline
    baseline = TFIDFBaseline()
    baseline.fit(dev_cases)
    print("TF-IDF baseline fitted")

    # Generate predictions
    predictions = []
    for i, g in enumerate(golden):
        result = baseline.predict(g["customer_text"], g.get("conversation_context", ""))
        predictions.append({
            "predicted_intent": result["intent"],
            "predicted_confidence": result["intent_confidence"],
            "predicted_decision": result["decision"],
            "predicted_reason": result["reason"],
            "generated_reply": result["reply"],
            "retrieved_cases": result["retrieved_cases"],
            "retrieved_case_ids": result["retrieved_case_ids"],
            "retrieved_similarities": [rc.get("similarity", 0) for rc in result["retrieved_cases"]]
        })

    # Save predictions
    pred_df = pd.DataFrame(predictions)
    pred_df.to_csv(OUTPUTS_DIR / "predictions.csv", index=False)

    # Intent evaluation
    from eval.evaluate_intents import evaluate_intents
    intent_metrics = evaluate_intents(golden, predictions)

    # Escalation evaluation
    from eval.evaluate_escalation import evaluate_escalation
    escalation_metrics = evaluate_escalation(golden, predictions)

    # Save metrics
    all_metrics = {
        "brand": "SpotifyCares",
        "golden_examples": len(golden),
        "intent": intent_metrics,
        "escalation": escalation_metrics,
        "reply": {"status": "SKIPPED (no LLM API key)"},
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "baseline": "TF-IDF"
    }

    with open(OUTPUTS_DIR / "metrics.json", 'w') as f:
        json.dump(all_metrics, f, indent=2)

    # Print results
    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"HIVER SUPPORT AGENT EVALUATION (TF-IDF Baseline)")
    print(f"{'='*60}")
    print(f"Brand                         SpotifyCares")
    print(f"Golden examples               {len(golden)}")
    print()
    print(f"INTENT")
    print(f"Accuracy                      {intent_metrics['accuracy']:.4f}")
    print(f"Macro F1                      {intent_metrics['macro_f1']:.4f}")
    print(f"Weighted F1                   {intent_metrics['weighted_f1']:.4f}")
    print()
    print(f"ESCALATION")
    print(f"Precision                     {escalation_metrics['precision']:.4f}")
    print(f"Recall                        {escalation_metrics['recall']:.4f}")
    print(f"F1                            {escalation_metrics['f1']:.4f}")
    print(f"Unsafe auto-handle rate       {escalation_metrics['unsafe_auto_handle_rate']:.4f}")
    print()
    print(f"REPLY                          PENDING LLM JUDGE")
    print(f"JUDGE VALIDATION               PENDING HUMAN VALIDATION")
    print(f"{'='*60}")
    print(f"Completed in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
