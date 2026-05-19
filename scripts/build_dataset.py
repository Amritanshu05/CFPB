"""
scripts/build_dataset.py

Entry-point script to build the processed dataset from the raw CFPB CSV.

Run from the project root:
    python scripts/build_dataset.py

What it does:
  1. Streams the raw CSV in chunks (never loads full file into memory)
  2. Filters rows without a complaint narrative
  3. Cleans narrative text and normalizes labels
  4. Saves a stratified sample (data/samples/complaints_sample.parquet)
  5. Splits into train / validation / test (data/processed/)

Options:
  --force   Rebuild all outputs even if they already exist
  --sample-only   Only build the sample, skip splitting
"""

import argparse
import sys
from pathlib import Path

# Allow running as a script from any directory
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cfpb_assistant.data.sampler import build_sample, build_splits
from cfpb_assistant.utils.logging_utils import get_logger

logger = get_logger("build_dataset", log_to_file=True)


def main():
    parser = argparse.ArgumentParser(description="Build CFPB complaint dataset.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild even if output files already exist",
    )
    parser.add_argument(
        "--sample-only",
        action="store_true",
        help="Only build the sample; skip train/val/test splitting",
    )
    args = parser.parse_args()

    logger.info("=== CFPB Dataset Build Script ===")

    logger.info("Step 1: Building stratified sample ...")
    sample_path = build_sample(force=args.force)
    logger.info(f"Sample ready at: {sample_path}")

    if not args.sample_only:
        logger.info("Step 2: Building train/validation/test splits ...")
        split_paths = build_splits(force=args.force)
        for name, path in split_paths.items():
            logger.info(f"  {name}: {path}")

    logger.info("=== Dataset build complete ===")


if __name__ == "__main__":
    main()
