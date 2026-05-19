"""
BM25 keyword-based retrieval over the complaint corpus.

Uses the rank_bm25 library. The index is built over tokenized complaint narratives
and serialized to disk so it does not need to be rebuilt every run.

Design notes:
  - Tokenization is simple whitespace split after lowercasing.
    This is intentional: BM25 works on term frequency, not semantics.
  - The corpus DataFrame (with metadata) is stored alongside the index
    so retrieved results include complaint_id, product, issue, and narrative.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi

from cfpb_assistant.utils.config import load_config, resolve_path, ensure_dir
from cfpb_assistant.utils.logging_utils import get_logger

logger = get_logger(__name__)


def _tokenize(text: str) -> list[str]:
    """Simple whitespace tokenizer for BM25."""
    return text.lower().split()


class BM25Retriever:
    """
    BM25 retriever backed by rank_bm25.BM25Okapi.

    Usage:
        retriever = BM25Retriever()
        retriever.build(corpus_df)          # or retriever.load()
        results = retriever.query("my complaint text", top_k=5)
    """

    INDEX_FILENAME = "bm25_index.pkl"
    CORPUS_FILENAME = "bm25_corpus.parquet"

    def __init__(self):
        self._index: BM25Okapi | None = None
        self._corpus: pd.DataFrame | None = None
        self._cfg = load_config("retrieval")
        self._paths_cfg = load_config("paths")
        self._index_dir = ensure_dir(
            resolve_path(self._paths_cfg["data"]["indexes_dir"]) / "bm25"
        )

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(self, corpus_df: pd.DataFrame, save: bool = True) -> None:
        """
        Build a BM25 index from a DataFrame of complaints.

        Args:
            corpus_df: DataFrame with at least a 'narrative_clean' column
            save: if True, persist the index and corpus to disk
        """
        text_col = self._cfg.get("text_column", "narrative_clean")
        if text_col not in corpus_df.columns:
            raise ValueError(f"Column '{text_col}' not found in corpus DataFrame.")

        logger.info(f"Building BM25 index over {len(corpus_df):,} documents ...")
        tokenized = [_tokenize(t) for t in corpus_df[text_col].fillna("").tolist()]
        self._index = BM25Okapi(
            tokenized,
            k1=self._cfg["bm25"]["k1"],
            b=self._cfg["bm25"]["b"],
        )
        self._corpus = corpus_df.reset_index(drop=True)
        logger.info("BM25 index built.")

        if save:
            self._save()

    def _save(self) -> None:
        index_path = self._index_dir / self.INDEX_FILENAME
        corpus_path = self._index_dir / self.CORPUS_FILENAME
        with open(index_path, "wb") as f:
            pickle.dump(self._index, f)
        self._corpus.to_parquet(corpus_path, index=False)
        logger.info(f"BM25 index saved to {index_path}")

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Load a previously saved BM25 index from disk."""
        index_path = self._index_dir / self.INDEX_FILENAME
        corpus_path = self._index_dir / self.CORPUS_FILENAME
        if not index_path.exists():
            raise FileNotFoundError(
                f"BM25 index not found at {index_path}. Run build() first."
            )
        with open(index_path, "rb") as f:
            self._index = pickle.load(f)
        self._corpus = pd.read_parquet(corpus_path)
        logger.info(f"BM25 index loaded ({len(self._corpus):,} documents)")

    def is_ready(self) -> bool:
        return self._index is not None and self._corpus is not None

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def query(self, text: str, top_k: int | None = None) -> pd.DataFrame:
        """
        Retrieve the top-k most relevant complaints for a query.

        Args:
            text: query complaint narrative
            top_k: number of results to return; defaults to config value

        Returns:
            DataFrame of top-k results, sorted by BM25 score descending,
            with an additional 'bm25_score' column.
        """
        if not self.is_ready():
            raise RuntimeError("BM25Retriever is not ready. Call build() or load() first.")
        if top_k is None:
            top_k = self._cfg.get("top_k", 5)

        tokens = _tokenize(text)
        scores = self._index.get_scores(tokens)

        top_indices = np.argsort(scores)[::-1][:top_k]
        results = self._corpus.iloc[top_indices].copy()
        results["bm25_score"] = scores[top_indices]
        results["retrieval_rank"] = range(1, len(results) + 1)
        return results.reset_index(drop=True)
