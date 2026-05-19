"""
scripts/build_index.py

Build BM25 and/or dense FAISS retrieval indexes from the processed train split.

Run from the project root:
    python scripts/build_index.py

Options:
  --index   One of: bm25, dense, hybrid (default: hybrid — builds both)
  --force   Rebuild even if indexes already exist
  --split   Which split to index: train, validation, test (default: train)
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cfpb_assistant.data.sampler import load_split
from cfpb_assistant.retrieval.bm25_retriever import BM25Retriever
from cfpb_assistant.retrieval.dense_retriever import DenseRetriever
from cfpb_assistant.retrieval.hybrid_retriever import HybridRetriever
from cfpb_assistant.utils.config import load_config, resolve_path
from cfpb_assistant.utils.logging_utils import get_logger

logger = get_logger("build_index", log_to_file=True)


def index_exists(index_type: str) -> bool:
    paths_cfg = load_config("paths")
    base = resolve_path(paths_cfg["data"]["indexes_dir"])
    if index_type == "bm25":
        return (base / "bm25" / "bm25_index.pkl").exists()
    elif index_type == "dense":
        return (base / "dense" / "dense_index.faiss").exists()
    return False


def main():
    parser = argparse.ArgumentParser(description="Build retrieval indexes.")
    parser.add_argument(
        "--index",
        default="hybrid",
        choices=["bm25", "dense", "hybrid"],
        help="Which index to build (default: hybrid)",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--split", default="train", choices=["train", "validation", "test"])
    args = parser.parse_args()

    logger.info(f"Loading '{args.split}' split ...")
    corpus_df = load_split(args.split)
    logger.info(f"Corpus: {len(corpus_df):,} rows")

    if args.index in ("bm25", "hybrid"):
        if not args.force and index_exists("bm25"):
            logger.info("BM25 index already exists — skipping. Use --force to rebuild.")
        else:
            logger.info("Building BM25 index ...")
            bm25 = BM25Retriever()
            bm25.build(corpus_df, save=True)

    if args.index in ("dense", "hybrid"):
        if not args.force and index_exists("dense"):
            logger.info("Dense index already exists — skipping. Use --force to rebuild.")
        else:
            logger.info("Building dense embedding index ...")
            dense = DenseRetriever()
            dense.build(corpus_df, save=True)

    logger.info("=== Index build complete ===")

    # Smoke test
    logger.info("Running smoke test query ...")
    sample_query = "I was charged twice for the same transaction and the bank refuses to refund me."

    if args.index in ("bm25", "hybrid"):
        bm25_r = BM25Retriever()
        bm25_r.load()
        bm25_results = bm25_r.query(sample_query, top_k=3)
        logger.info(f"BM25 top result: {bm25_results.iloc[0]['narrative_clean'][:120]}")

    if args.index in ("dense", "hybrid"):
        dense_r = DenseRetriever()
        dense_r.load()
        dense_results = dense_r.query(sample_query, top_k=3)
        logger.info(f"Dense top result: {dense_results.iloc[0]['narrative_clean'][:120]}")

    if args.index == "hybrid":
        hybrid_r = HybridRetriever()
        hybrid_r.load()
        hybrid_results = hybrid_r.query(sample_query, top_k=3)
        logger.info(f"Hybrid top result: {hybrid_results.iloc[0]['narrative_clean'][:120]}")


if __name__ == "__main__":
    main()
