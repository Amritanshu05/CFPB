"""
scripts/evaluate.py

Run evaluation for any combination of: classification, retrieval, generation.

Run from the project root:
    python scripts/evaluate.py --task classification
    python scripts/evaluate.py --task retrieval
    python scripts/evaluate.py --task generation
    python scripts/evaluate.py --task all

Options:
  --task        One of: classification, retrieval, generation, all (default: all)
  --n-queries   Number of test queries for retrieval eval (default: 200)
  --n-gen-samples  Number of complaints for generation eval (default: 20)
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cfpb_assistant.data.sampler import load_split
from cfpb_assistant.classification.baselines import load_model, predict_with_confidence, BASELINE_BUILDERS
from cfpb_assistant.retrieval.bm25_retriever import BM25Retriever
from cfpb_assistant.retrieval.dense_retriever import DenseRetriever
from cfpb_assistant.retrieval.hybrid_retriever import HybridRetriever
from cfpb_assistant.evaluation.classification_eval import evaluate_classifier, compare_classifiers
from cfpb_assistant.evaluation.retrieval_eval import evaluate_retriever, compare_retrievers
from cfpb_assistant.evaluation.generation_eval import evaluate_generation
from cfpb_assistant.utils.config import load_config
from cfpb_assistant.utils.logging_utils import get_logger

logger = get_logger("evaluate", log_to_file=True)


def eval_classification():
    cfg = load_config("classification")
    text_col = cfg.get("text_column", "narrative_clean")
    label_col = cfg.get("target_column", "product_clean")

    test_df = load_split("test").dropna(subset=[text_col, label_col])
    X_test = test_df[text_col].tolist()
    y_test = test_df[label_col].tolist()

    all_results = []

    # Classical baselines
    for model_name in BASELINE_BUILDERS:
        try:
            pipeline = load_model(model_name)
        except FileNotFoundError:
            logger.warning(f"Model not found: {model_name} — skipping.")
            continue
        y_pred, _ = predict_with_confidence(pipeline, X_test)
        results = evaluate_classifier(y_test, y_pred, model_name=model_name, save=True)
        all_results.append(results)

    # Transformer (optional — only if trained + saved)
    try:
        from cfpb_assistant.classification.transformer import TransformerClassifier
        clf = TransformerClassifier.load("transformer")
        y_pred = clf.predict(X_test)
        results = evaluate_classifier(y_test, y_pred, model_name="transformer", save=True)
        all_results.append(results)
    except FileNotFoundError:
        logger.warning("Transformer model not found — skipping.")
    except Exception as e:
        logger.warning(f"Transformer evaluation failed: {e}")

    if all_results:
        logger.info("\n=== Classification Comparison ===")
        compare_classifiers(all_results)


def eval_retrieval(n_queries: int = 200):
    test_df = load_split("test")
    cfg = load_config("retrieval")
    top_k = cfg.get("top_k", 5)

    # Sample queries from test split (only rows with narratives).
    # Keep issue_clean when available so the retrieval evaluator can also
    # compute the stricter 'product+issue' relevance proxy.
    query_df = test_df.dropna(subset=["narrative_clean", "product_clean"])
    if len(query_df) > n_queries:
        query_df = query_df.sample(n=n_queries, random_state=42)
    keep_cols = [c for c in ["narrative_clean", "product_clean", "issue_clean"] if c in query_df.columns]
    queries = query_df[keep_cols].to_dict(orient="records")

    logger.info(f"Evaluating retrieval on {len(queries)} queries (top_k={top_k}) ...")

    all_results = []

    # BM25
    try:
        bm25 = BM25Retriever()
        bm25.load()
        r = evaluate_retriever(bm25, queries, top_k=top_k, model_name="bm25", save=True)
        all_results.append(r)
    except Exception as e:
        logger.warning(f"BM25 evaluation failed: {e}")

    # Dense
    try:
        dense = DenseRetriever()
        dense.load()
        r = evaluate_retriever(dense, queries, top_k=top_k, model_name="dense", save=True)
        all_results.append(r)
    except Exception as e:
        logger.warning(f"Dense evaluation failed: {e}")

    # Hybrid
    try:
        hybrid = HybridRetriever()
        hybrid.load()
        r = evaluate_retriever(hybrid, queries, top_k=top_k, model_name="hybrid_rrf", save=True)
        all_results.append(r)
    except Exception as e:
        logger.warning(f"Hybrid evaluation failed: {e}")

    if all_results:
        logger.info("\n=== Retrieval Comparison ===")
        compare_retrievers(all_results, top_k=top_k)


def eval_generation(n_samples: int = 20, model_name: str = "svm"):
    """
    End-to-end generation evaluation. Runs the full pipeline on a
    sample of test complaints and computes structural + grounding
    metrics. Works in mock mode (no API keys required).
    """
    logger.info(f"Generation evaluation on n={n_samples} test complaints ...")
    try:
        classifier = load_model(model_name)
    except FileNotFoundError:
        logger.warning(f"No saved classifier '{model_name}'. Skipping generation eval.")
        return

    try:
        retriever = HybridRetriever()
        retriever.load()
    except Exception as e:
        logger.warning(f"Retriever not available ({e}). Skipping generation eval.")
        return

    from cfpb_assistant.generation.generator import ResolutionGenerator
    generator = ResolutionGenerator(classifier=classifier, retriever=retriever)

    test_df = load_split("test").dropna(subset=["narrative_clean", "product_clean"])
    if len(test_df) > n_samples:
        test_df = test_df.sample(n=n_samples, random_state=42)
    samples = test_df[["narrative_clean", "product_clean"]].to_dict(orient="records")

    evaluate_generation(generator, samples, model_name="pipeline_generation", save=True)


def main():
    parser = argparse.ArgumentParser(description="Evaluate CFPB pipeline components.")
    parser.add_argument(
        "--task",
        default="all",
        choices=["classification", "retrieval", "generation", "all"],
    )
    parser.add_argument("--n-queries", type=int, default=200)
    parser.add_argument("--n-gen-samples", type=int, default=20)
    args = parser.parse_args()

    if args.task in ("classification", "all"):
        logger.info("\n=== Classification Evaluation ===")
        eval_classification()

    if args.task in ("retrieval", "all"):
        logger.info("\n=== Retrieval Evaluation ===")
        eval_retrieval(n_queries=args.n_queries)

    if args.task in ("generation", "all"):
        logger.info("\n=== Generation Evaluation ===")
        eval_generation(n_samples=args.n_gen_samples)


if __name__ == "__main__":
    main()
