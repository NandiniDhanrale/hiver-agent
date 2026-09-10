"""Main evaluation harness - runs all evaluation steps."""
import json
import logging
import argparse
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"


def run_all(use_cache: bool = True, limit: int = None, skip_judge: bool = False):
    """Run the complete evaluation pipeline."""
    start_time = time.time()

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load golden set
    golden_file = PROJECT_ROOT / "data" / "golden" / "golden_set.csv"
    if not golden_file.exists():
        print("ERROR: Golden set not found at data/golden/golden_set.csv")
        print("Run: python -m scripts.create_golden_candidates")
        return

    import pandas as pd
    golden_df = pd.read_csv(golden_file)
    golden = golden_df.to_dict('records')
    if limit:
        golden = golden[:limit]

    print(f"Loaded {len(golden)} golden examples")

    # Load dev cases for retrieval
    dev_file = PROJECT_ROOT / "data" / "processed" / "dev_cases.json"
    if not dev_file.exists():
        print("ERROR: Dev cases not found. Run data pipeline first.")
        return

    with open(dev_file, 'r') as f:
        dev_cases = json.load(f)

    # Create agent
    from src.agent import create_agent
    agent = create_agent(dev_cases)

    # Generate predictions
    print("Generating predictions...")
    predictions = []
    for i, g in enumerate(golden):
        result = agent.process(g["customer_text"], g.get("conversation_context", ""))
        predictions.append({
            "predicted_intent": result.intent,
            "predicted_confidence": result.intent_confidence,
            "predicted_decision": result.decision,
            "predicted_reason": result.reason,
            "generated_reply": result.reply,
            "retrieved_cases": result.retrieved_cases,
            "retrieved_case_ids": result.retrieved_case_ids,
            "retrieved_similarities": [rc["similarity"] for rc in result.retrieved_cases]
        })
        if (i + 1) % 10 == 0:
            print(f"  Processed {i+1}/{len(golden)}")

    # Save predictions
    pred_df = pd.DataFrame(predictions)
    pred_df.to_csv(OUTPUTS_DIR / "predictions.csv", index=False)

    # Intent evaluation
    from eval.evaluate_intents import evaluate_intents
    intent_metrics = evaluate_intents(golden, predictions)
    print(f"\nIntent Macro F1: {intent_metrics['macro_f1']:.4f}")

    # Escalation evaluation
    from eval.evaluate_escalation import evaluate_escalation
    escalation_metrics = evaluate_escalation(golden, predictions)
    print(f"Escalation F1: {escalation_metrics['f1']:.4f}")
    print(f"Unsafe auto-handle rate: {escalation_metrics['unsafe_auto_handle_rate']:.4f}")

    # Reply evaluation (LLM judge)
    reply_metrics = None
    if not skip_judge:
        from eval.evaluate_replies import evaluate_replies
        from src.llm import llm_call

        print("Running LLM judge on replies...")
        reply_result = evaluate_replies(
            golden, predictions, llm_call,
            sample_size=50, use_cache=use_cache
        )
        reply_metrics = reply_result["scores"]
        print(f"Reply overall score: {reply_metrics.get('overall', {}).get('mean', 'N/A')}")
    else:
        reply_metrics = {"status": "SKIPPED"}
        print("Reply evaluation SKIPPED")

    # Failure analysis
    from eval.failure_analysis import categorize_failures, analyze_top_failures, save_failure_analysis
    failures = categorize_failures(golden, predictions)
    analysis = analyze_top_failures(failures)
    save_failure_analysis(failures, analysis)

    # Save all metrics
    all_metrics = {
        "brand": "SpotifyCares",
        "golden_examples": len(golden),
        "intent": intent_metrics,
        "escalation": escalation_metrics,
        "reply": reply_metrics,
        "failure_count": len(failures),
        "top_failures": [a["category"] for a in analysis[:5]],
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "use_cache": use_cache
    }

    with open(OUTPUTS_DIR / "metrics.json", 'w') as f:
        json.dump(all_metrics, f, indent=2)

    # Print headline output
    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"HIVER SUPPORT AGENT EVALUATION")
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

    if reply_metrics and "overall" in reply_metrics:
        print(f"REPLY")
        print(f"Overall judge score           {reply_metrics['overall']['mean']:.2f} / 5")
        print(f"Groundedness                  {reply_metrics.get('groundedness', {}).get('mean', 'N/A')} / 5")
        print(f"Safety                        {reply_metrics.get('safety', {}).get('mean', 'N/A')} / 5")
    else:
        print(f"REPLY                          PENDING LLM JUDGE")

    print()
    print(f"JUDGE VALIDATION")
    human_file = PROJECT_ROOT / "data" / "golden" / "human_reply_scores.csv"
    if human_file.exists():
        import pandas as pd
        human_df = pd.read_csv(human_file)
        if not human_df.empty:
            filled = human_df["correctness"].notna().sum()
            print(f"Human examples scored         {filled}")
        else:
            print(f"Human examples scored         0")
            print(f"Weighted kappa                PENDING")
            print(f"Spearman                      PENDING")
    else:
        print(f"Human examples scored         0")
        print(f"Weighted kappa                PENDING")
        print(f"Spearman                      PENDING")

    print(f"{'='*60}")
    print(f"Completed in {elapsed:.1f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run full evaluation")
    parser.add_argument("--use-cache", action="store_true", default=True,
                        help="Use cached LLM responses")
    parser.add_argument("--fresh", action="store_true",
                        help="Run fresh evaluation without cache")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of golden examples")
    parser.add_argument("--skip-judge", action="store_true",
                        help="Skip LLM judge evaluation")
    args = parser.parse_args()

    use_cache = not args.fresh
    run_all(use_cache=use_cache, limit=args.limit, skip_judge=args.skip_judge)
