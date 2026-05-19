"""
Retrieval evaluation utilities.

Computes standard IR metrics for the complaint retrieval system:
  - Precision@k
  - Recall@k
  - Mean Reciprocal Rank (MRR)
  - Normalized Discounted Cumulative Gain (nDCG@k)

Relevance criterion (PROXY, not human-annotated):
  A retrieved complaint is "relevant" if it matches the query's label
  on a chosen column. Two modes are supported:
    - "product"         : same product_clean label
    - "product+issue"   : same product_clean AND same issue_clean label
                          (stricter — closer to true topical relevance)
  By default, results are computed for both modes and a per-class
  breakdown on product_clean is also saved. This is explicitly a proxy;
  real relevance requires human annotation (future work).
"""

from __future__ import annotations

import json
import numpy as np
from datetime import datetime
from pathlib import Path

from cfpb_assistant.utils.config import load_config, resolve_path, ensure_dir
from cfpb_assistant.utils.logging_utils import get_logger

logger = get_logger(__name__)


def precision_at_k(relevant: list[bool]) -> float:
    """Fraction of top-k results that are relevant."""
    if not relevant:
        return 0.0
    return sum(relevant) / len(relevant)


def recall_at_k(relevant: list[bool], total_relevant: int) -> float:
    """Fraction of all relevant documents found in top-k results."""
    if total_relevant == 0:
        return 0.0
    return sum(relevant) / total_relevant


def reciprocal_rank(relevant: list[bool]) -> float:
    """1 / rank of the first relevant document (0 if none found)."""
    for i, r in enumerate(relevant):
        if r:
            return 1.0 / (i + 1)
    return 0.0


def ndcg_at_k(relevant: list[bool]) -> float:
    """Normalized Discounted Cumulative Gain at k."""
    k = len(relevant)
    if k == 0:
        return 0.0

    dcg = sum(
        (1.0 / np.log2(i + 2)) for i, r in enumerate(relevant) if r
    )
    ideal_dcg = sum(
        1.0 / np.log2(i + 2) for i in range(min(sum(relevant), k))
    )
    if ideal_dcg == 0:
        return 0.0
    return dcg / ideal_dcg


def _aggregate(values: list[float]) -> float:
    return round(float(np.mean(values)), 4) if values else 0.0


def _run_queries(retriever, queries: list[dict], top_k: int) -> list[dict]:
    """
    Run every query once, return a list of per-query records with the
    retrieved labels attached. This lets us compute multiple relevance
    criteria without re-running retrieval.
    """
    records = []
    for i, row in enumerate(queries):
        q_text = row.get("narrative_clean", "")
        q_product = row.get("product_clean", "")
        q_issue = row.get("issue_clean", "")
        if not q_text or not q_product:
            continue
        try:
            results = retriever.query(q_text, top_k=top_k)
        except Exception as e:
            logger.warning(f"Retrieval failed for query {i}: {e}")
            continue

        retrieved_products = (
            results["product_clean"].tolist() if "product_clean" in results.columns else []
        )
        retrieved_issues = (
            results["issue_clean"].tolist() if "issue_clean" in results.columns else []
        )
        records.append({
            "query_product": q_product,
            "query_issue": q_issue,
            "retrieved_products": retrieved_products,
            "retrieved_issues": retrieved_issues,
        })
        if (i + 1) % 50 == 0:
            logger.info(f"  Evaluated {i+1}/{len(queries)} queries ...")
    return records


def _metrics_for_criterion(records: list[dict], criterion: str, top_k: int) -> dict:
    """Compute P@k, MRR, nDCG@k for one relevance criterion."""
    p_list, mrr_list, ndcg_list = [], [], []
    for r in records:
        if criterion == "product":
            relevant = [p == r["query_product"] for p in r["retrieved_products"]]
        elif criterion == "product+issue":
            relevant = [
                (p == r["query_product"]) and (iss == r["query_issue"])
                for p, iss in zip(r["retrieved_products"], r["retrieved_issues"])
            ]
        else:
            raise ValueError(f"Unknown criterion: {criterion}")
        p_list.append(precision_at_k(relevant))
        mrr_list.append(reciprocal_rank(relevant))
        ndcg_list.append(ndcg_at_k(relevant))

    return {
        f"precision_at_{top_k}": _aggregate(p_list),
        "mrr": _aggregate(mrr_list),
        f"ndcg_at_{top_k}": _aggregate(ndcg_list),
        "num_queries": len(p_list),
    }


def _per_class_breakdown(records: list[dict], top_k: int) -> dict:
    """Product-level P@k grouped by the query's product label."""
    buckets: dict[str, list[float]] = {}
    for r in records:
        relevant = [p == r["query_product"] for p in r["retrieved_products"]]
        buckets.setdefault(r["query_product"], []).append(precision_at_k(relevant))
    return {
        prod: {
            "num_queries": len(vals),
            f"precision_at_{top_k}": _aggregate(vals),
        }
        for prod, vals in sorted(buckets.items())
    }


def evaluate_retriever(
    retriever,
    queries: list[dict],
    top_k: int = 5,
    model_name: str = "retriever",
    save: bool = True,
) -> dict:
    """
    Evaluate a retriever on a set of query complaints.

    Runs two relevance criteria in one pass:
      - "product"        : same product_clean  (loose proxy)
      - "product+issue"  : same product_clean AND issue_clean (stricter proxy)

    Also stores a per-product breakdown of P@k on the loose criterion.

    Both criteria are explicit proxies; the report text calls that out.
    """
    records = _run_queries(retriever, queries, top_k=top_k)

    product_metrics = _metrics_for_criterion(records, "product", top_k)
    strict_metrics = _metrics_for_criterion(records, "product+issue", top_k)
    per_class = _per_class_breakdown(records, top_k)

    metrics = {
        "model": model_name,
        "timestamp": datetime.now().isoformat(),
        "num_queries": product_metrics["num_queries"],
        "top_k": top_k,
        "relevance_note": (
            "Metrics use label-match proxies, not human-annotated relevance. "
            "'product' = same product_clean, 'product+issue' = both match."
        ),
        "product": {
            k: v for k, v in product_metrics.items() if k != "num_queries"
        },
        "product_plus_issue": {
            k: v for k, v in strict_metrics.items() if k != "num_queries"
        },
        # Kept at top level for backward-compat with existing report tables.
        f"precision_at_{top_k}": product_metrics[f"precision_at_{top_k}"],
        "mrr": product_metrics["mrr"],
        f"ndcg_at_{top_k}": product_metrics[f"ndcg_at_{top_k}"],
        "per_class_product": per_class,
    }

    logger.info(
        f"[{model_name}] product  P@{top_k}={product_metrics[f'precision_at_{top_k}']:.4f} "
        f"MRR={product_metrics['mrr']:.4f} nDCG@{top_k}={product_metrics[f'ndcg_at_{top_k}']:.4f}"
    )
    logger.info(
        f"[{model_name}] prod+iss P@{top_k}={strict_metrics[f'precision_at_{top_k}']:.4f} "
        f"MRR={strict_metrics['mrr']:.4f} nDCG@{top_k}={strict_metrics[f'ndcg_at_{top_k}']:.4f}"
    )

    if save:
        paths_cfg = load_config("paths")
        results_dir = ensure_dir(resolve_path(paths_cfg["outputs"]["results_dir"]))
        out_path = results_dir / f"{model_name}_retrieval_results.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Retrieval results saved to {out_path}")

    return metrics


def compare_retrievers(results_list: list[dict], top_k: int = 5) -> None:
    """Print a formatted comparison table of retriever results."""
    header = f"{'Model':<25} {'P@'+str(top_k):>8} {'MRR':>8} {'nDCG@'+str(top_k):>10}"
    sep = "-" * len(header)
    print(sep)
    print(header)
    print(sep)
    for r in results_list:
        print(
            f"{r['model']:<25} "
            f"{r.get(f'precision_at_{top_k}', 0):>8.4f} "
            f"{r.get('mrr', 0):>8.4f} "
            f"{r.get(f'ndcg_at_{top_k}', 0):>10.4f}"
        )
    print(sep)
