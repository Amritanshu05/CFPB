"""
scripts/run_pipeline.py

Run the full CFPB Complaint-to-Resolution pipeline on one or more complaints.

Run from the project root:
    python scripts/run_pipeline.py --complaint "I was charged twice for the same transaction."
    python scripts/run_pipeline.py --file path/to/complaints.txt

Options:
  --complaint TEXT    Single complaint text (quoted)
  --file PATH         Text file with one complaint per line
  --model             Classifier model to use: logistic_regression, svm (default: svm)
  --provider          LLM provider: mock, openai, anthropic (default: mock)
  --output-file PATH  Save results as JSON to this path
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cfpb_assistant.classification.baselines import load_model
from cfpb_assistant.retrieval.hybrid_retriever import HybridRetriever
from cfpb_assistant.generation.generator import ResolutionGenerator
from cfpb_assistant.utils.logging_utils import get_logger

logger = get_logger("run_pipeline")


def run_on_text(text: str, generator: ResolutionGenerator) -> dict:
    logger.info(f"Processing complaint ({len(text)} chars) ...")
    result = generator.process(text)
    return result


def main():
    parser = argparse.ArgumentParser(description="Run CFPB complaint pipeline.")
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--complaint", type=str, help="Single complaint text")
    input_group.add_argument("--file", type=str, help="File with one complaint per line")
    parser.add_argument("--model", default="svm", help="Classifier to use (default: svm)")
    parser.add_argument("--provider", default=None, help="LLM provider override")
    parser.add_argument("--output-file", type=str, default=None)
    args = parser.parse_args()

    # Load classifier
    logger.info(f"Loading classifier: {args.model}")
    classifier = load_model(args.model)

    # Load retriever
    logger.info("Loading hybrid retriever ...")
    retriever = HybridRetriever()
    retriever.load()

    # Build generator; pass provider override through the constructor
    generator = ResolutionGenerator(
        classifier=classifier,
        retriever=retriever,
        provider=args.provider,
    )

    # Gather complaint texts
    if args.complaint:
        complaints = [args.complaint]
    else:
        file_path = Path(args.file)
        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            sys.exit(1)
        with open(file_path, "r", encoding="utf-8") as f:
            complaints = [line.strip() for line in f if line.strip()]

    # Process each complaint
    all_results = []
    for i, complaint in enumerate(complaints, start=1):
        logger.info(f"\n--- Complaint {i}/{len(complaints)} ---")
        result = run_on_text(complaint, generator)
        result["input_complaint"] = complaint[:300]
        all_results.append(result)

        # Print to console
        print(f"\n{'='*60}")
        print(f"COMPLAINT: {complaint[:150]}{'...' if len(complaint) > 150 else ''}")
        print(f"{'='*60}")
        print(json.dumps(result, indent=2))

    # Optionally save to file
    if args.output_file:
        out_path = Path(args.output_file)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2)
        logger.info(f"Results saved to {out_path}")

    return all_results


if __name__ == "__main__":
    main()
