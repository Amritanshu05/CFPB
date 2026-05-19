"""
Hybrid retriever combining BM25 (keyword) and dense (semantic) retrieval.

Fusion strategy: Reciprocal Rank Fusion (RRF).
RRF is a well-established rank fusion method that does not require score
normalization and handles the scale difference between BM25 and cosine scores.

Formula (for each document d):
    RRF_score(d) = sum over each retriever r of: 1 / (k + rank_r(d))
    where k=60 is a smoothing constant.

We optionally also support simple weighted score fusion (after min-max
normalization) for comparison purposes.
"""

from __future__ import annotations

import pandas as pd
import numpy as np

from cfpb_assistant.retrieval.bm25_retriever import BM25Retriever
from cfpb_assistant.retrieval.dense_retriever import DenseRetriever
from cfpb_assistant.utils.config import load_config
from cfpb_assistant.utils.logging_utils import get_logger

logger = get_logger(__name__)

RRF_K = 60  # standard RRF smoothing constant


class HybridRetriever:
    """
    Combines BM25 and dense retrieval results via Reciprocal Rank Fusion.

    Usage:
        retriever = HybridRetriever()
        retriever.load()   # assumes indexes already built
        results = retriever.query("my complaint text", top_k=5)
    """

    def __init__(self):
        self.bm25 = BM25Retriever()
        self.dense = DenseRetriever()
        self._cfg = load_config("retrieval")

    def build(self, corpus_df: pd.DataFrame, save: bool = True) -> None:
        """Build both BM25 and dense indexes from the given corpus."""
        logger.info("Building BM25 index ...")
        self.bm25.build(corpus_df, save=save)
        logger.info("Building dense embedding index ...")
        self.dense.build(corpus_df, save=save)

    def load(self) -> None:
        """Load both pre-built indexes from disk."""
        self.bm25.load()
        self.dense.load()

    def is_ready(self) -> bool:
        return self.bm25.is_ready() and self.dense.is_ready()

    def query(
        self,
        text: str,
        top_k: int | None = None,
        method: str = "rrf",
    ) -> pd.DataFrame:
        """
        Retrieve top-k results using hybrid fusion.

        Args:
            text: query complaint narrative
            top_k: number of final results to return
            method: fusion method — 'rrf' (Reciprocal Rank Fusion) or
                    'weighted' (weighted score fusion)

        Returns:
            DataFrame of top-k results with a 'hybrid_score' column, sorted
            by hybrid_score descending.
        """
        if not self.is_ready():
            raise RuntimeError("HybridRetriever not ready. Call build() or load() first.")

        if top_k is None:
            top_k = self._cfg.get("top_k", 5)

        # Retrieve a larger candidate pool from each method before fusion
        candidate_k = max(top_k * 4, 20)
        bm25_results = self.bm25.query(text, top_k=candidate_k)
        dense_results = self.dense.query(text, top_k=candidate_k)

        if method == "rrf":
            return self._rrf_fusion(bm25_results, dense_results, top_k)
        elif method == "weighted":
            return self._weighted_fusion(bm25_results, dense_results, top_k)
        else:
            raise ValueError(f"Unknown fusion method: {method}. Use 'rrf' or 'weighted'.")

    # ------------------------------------------------------------------
    # Fusion methods
    # ------------------------------------------------------------------

    def _rrf_fusion(
        self,
        bm25_results: pd.DataFrame,
        dense_results: pd.DataFrame,
        top_k: int,
    ) -> pd.DataFrame:
        """Reciprocal Rank Fusion."""
        id_col = "complaint_id"

        rrf_scores: dict[str, float] = {}

        for rank, row in enumerate(bm25_results.itertuples(), start=1):
            doc_id = str(getattr(row, id_col, rank))
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (RRF_K + rank)

        for rank, row in enumerate(dense_results.itertuples(), start=1):
            doc_id = str(getattr(row, id_col, rank))
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (RRF_K + rank)

        # Combine candidate rows, keeping the first occurrence of each doc
        all_candidates = pd.concat([bm25_results, dense_results], ignore_index=True)
        all_candidates = all_candidates.drop_duplicates(subset=[id_col], keep="first")

        all_candidates["hybrid_score"] = all_candidates[id_col].astype(str).map(rrf_scores)
        result = (
            all_candidates
            .sort_values("hybrid_score", ascending=False)
            .head(top_k)
            .reset_index(drop=True)
        )
        result["retrieval_rank"] = range(1, len(result) + 1)
        return result

    def _weighted_fusion(
        self,
        bm25_results: pd.DataFrame,
        dense_results: pd.DataFrame,
        top_k: int,
    ) -> pd.DataFrame:
        """Weighted score fusion with min-max normalization."""
        bm25_w = self._cfg["hybrid"].get("bm25_weight", 0.4)
        dense_w = self._cfg["hybrid"].get("dense_weight", 0.6)
        id_col = "complaint_id"

        def minmax(scores: np.ndarray) -> np.ndarray:
            lo, hi = scores.min(), scores.max()
            if hi == lo:
                return np.zeros_like(scores)
            return (scores - lo) / (hi - lo)

        bm25_results = bm25_results.copy()
        dense_results = dense_results.copy()
        bm25_results["norm_score"] = minmax(bm25_results["bm25_score"].values) * bm25_w
        dense_results["norm_score"] = minmax(dense_results["dense_score"].values) * dense_w

        combined: dict[str, dict] = {}
        for _, row in bm25_results.iterrows():
            doc_id = str(row[id_col])
            combined[doc_id] = {"row": row, "score": row["norm_score"]}

        for _, row in dense_results.iterrows():
            doc_id = str(row[id_col])
            if doc_id in combined:
                combined[doc_id]["score"] += row["norm_score"]
            else:
                combined[doc_id] = {"row": row, "score": row["norm_score"]}

        rows = [v["row"] for v in combined.values()]
        scores = [v["score"] for v in combined.values()]
        result_df = pd.DataFrame(rows).reset_index(drop=True)
        result_df["hybrid_score"] = scores
        result_df = (
            result_df
            .sort_values("hybrid_score", ascending=False)
            .head(top_k)
            .reset_index(drop=True)
        )
        result_df["retrieval_rank"] = range(1, len(result_df) + 1)
        return result_df
