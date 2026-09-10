"""Command-line interface for the support agent."""
import argparse
import json
import sys
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


def main():
    parser = argparse.ArgumentParser(description="SpotifyCares AI Support Agent")
    parser.add_argument("--text", type=str, help="Customer message text")
    parser.add_argument("--context", type=str, default="", help="Conversation context")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--processed-dir", type=str, default="data/processed",
                        help="Path to processed data directory")
    args = parser.parse_args()

    if not args.text:
        print("Error: --text is required", file=sys.stderr)
        sys.exit(1)

    # Load cases
    processed_dir = Path(args.processed_dir)
    cases_file = processed_dir / "dev_cases.json"
    if not cases_file.exists():
        print(f"Error: {cases_file} not found. Run data pipeline first.", file=sys.stderr)
        sys.exit(1)

    import json
    with open(cases_file, 'r') as f:
        cases = json.load(f)

    # Create agent
    from .agent import create_agent
    agent = create_agent(cases)

    # Process
    result = agent.process(args.text, args.context)

    if args.json:
        print(json.dumps(result.model_dump(), indent=2))
    else:
        print(f"\n{'='*50}")
        print(f"Customer: {args.text}")
        print(f"\nIntent: {result.intent}")
        print(f"Confidence: {result.intent_confidence}")
        print(f"\nRetrieved cases:")
        for rc in result.retrieved_cases:
            print(f"  {rc['case_id']} (similarity: {rc['similarity']:.3f})")
        print(f"\nDraft reply:")
        print(f"  {result.reply}")
        print(f"\nDecision: {result.decision}")
        print(f"Reason: {result.reason}")
        print(f"{'='*50}")


if __name__ == "__main__":
    main()
