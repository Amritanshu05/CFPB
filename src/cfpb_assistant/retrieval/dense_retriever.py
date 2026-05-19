"""
Dense embedding retrieval using sentence-transformers + FAISS.

Each complaint narrative is encoded into a fixed-size embedding vector.
FAISS is used for approximate nearest-neighbor search at scale.

Design notes:
  - Encoding is done in batches to avoid OOM on large corpora.
  - Index type "flat" uses exact L2 search (best for < 100k documents).
  - Index type "ivf" uses approximate search (faster for > 100k documents).
  - Embeddings are normalized to unit length before indexing so that
    inner product == cosine similarity.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import faiss
import torch
from sentence_transformers import SentenceTransformer

from cfpb_assistant.utils.config import load_config, resolve_path, ensure_dir
from cfpb_assistant.utils.logging_utils import get_logger

logger = get_logger(__name__)


def _select_device() -> str:
    """Pick 'cuda' if a CUDA-capable GPU is available, else 'cpu'."""
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class DenseRetriever:
    """
    Dense retriever using sentence embeddings + FAISS index.

    Usage:
        retriever = DenseRetriever()
        retriever.build(corpus_df)          # or retriever.load()
        results = retriever.query("my complaint text", top_k=5)
    """

    INDEX_FILENAME = "dense_index.faiss"
    CORPUS_FILENAME = "dense_corpus.parquet"
    META_FILENAME = "dense_meta.pkl"

    def __init__(self):
        self._index: Optional[faiss.Index] = None
        self._corpus: Optional[pd.DataFrame] = None
        self._model: Optional[SentenceTransformer] = None
        self._cfg = load_config("retrieval")
        self._paths_cfg = load_config("paths")
        self._index_dir = ensure_dir(
            resolve_path(self._paths_cfg["data"]["indexes_dir"]) / "dense"
        )
        self._model_name = self._cfg["dense"]["model_name"]
        self._batch_size = self._cfg["dense"].get("batch_size", 64)

    def _get_model(self) -> SentenceTransformer:
        if self._model is None:
            device = _select_device()
            logger.info(
                f"Loading sentence-transformer model: {self._model_name} on device={device}"
            )
            self._model = SentenceTransformer(self._model_name, device=device)
        return self._model

    def _encode(self, texts: list[str]) -> np.ndarray:
        """Encode texts into L2-normalized embeddings."""
        model = self._get_model()
        embeddings = model.encode(
            texts,
            batch_size=self._batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return embeddings.astype("float32")

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(self, corpus_df: pd.DataFrame, save: bool = True) -> None:
        """
        Encode all corpus documents and build a FAISS index.

        Args:
            corpus_df: DataFrame with a 'narrative_clean' column
            save: if True, persist the index and corpus to disk
        """
        text_col = self._cfg.get("text_column", "narrative_clean")
        if text_col not in corpus_df.columns:
            raise ValueError(f"Column '{text_col}' not found in corpus DataFrame.")

        texts = corpus_df[text_col].fillna("").tolist()
        logger.info(f"Encoding {len(texts):,} documents with {self._model_name} ...")
        embeddings = self._encode(texts)

        dim = embeddings.shape[1]
        index_type = self._cfg["dense"].get("index_type", "flat")

        if index_type == "ivf":
            nlist = self._cfg["dense"].get("nlist", 100)
            quantizer = faiss.IndexFlatIP(dim)
            index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
            logger.info(f"Training IVF index with nlist={nlist} ...")
            index.train(embeddings)
        else:
            index = faiss.IndexFlatIP(dim)

        index.add(embeddings)
        self._index = index
        self._corpus = corpus_df.reset_index(drop=True)
        logger.info(f"FAISS index built: {index.ntotal:,} vectors, dim={dim}")

        if save:
            self._save()

    def _save(self) -> None:
        index_path = self._index_dir / self.INDEX_FILENAME
        corpus_path = self._index_dir / self.CORPUS_FILENAME
        meta_path = self._index_dir / self.META_FILENAME

        faiss.write_index(self._index, str(index_path))
        self._corpus.to_parquet(corpus_path, index=False)
        with open(meta_path, "wb") as f:
            pickle.dump({"model_name": self._model_name}, f)
        logger.info(f"Dense index saved to {index_path}")

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    def load(self) -> None:
        """
        Load a previously saved FAISS index from disk.

        Validates that the sentence-transformer model configured in
        retrieval.yaml matches the model that was used to build the
        saved index. Mismatches would silently produce wrong query
        embeddings, so we fail loudly instead.
        """
        index_path = self._index_dir / self.INDEX_FILENAME
        corpus_path = self._index_dir / self.CORPUS_FILENAME
        meta_path = self._index_dir / self.META_FILENAME
        if not index_path.exists():
            raise FileNotFoundError(
                f"Dense index not found at {index_path}. Run build() first."
            )

        if meta_path.exists():
            with open(meta_path, "rb") as f:
                meta = pickle.load(f)
            indexed_model = meta.get("model_name")
            if indexed_model and indexed_model != self._model_name:
                raise RuntimeError(
                    f"Dense index/model mismatch.\n"
                    f"  Index was built with: {indexed_model!r}\n"
                    f"  Config currently asks for: {self._model_name!r}\n"
                    f"Rebuild the index (scripts/build_index.py --index dense --force) "
                    f"or revert the config."
                )

        self._index = faiss.read_index(str(index_path))
        self._corpus = pd.read_parquet(corpus_path)
        logger.info(
            f"Dense index loaded ({self._index.ntotal:,} vectors, model={self._model_name})"
        )

    def is_ready(self) -> bool:
        return self._index is not None and self._corpus is not None

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def query(self, text: str, top_k: int | None = None) -> pd.DataFrame:
        """
        Retrieve the top-k most semantically similar complaints.

        Args:
            text: query complaint narrative
            top_k: number of results; defaults to config value

        Returns:
            DataFrame of top-k results sorted by cosine similarity descending,
            with a 'dense_score' column.
        """
        if not self.is_ready():
            raise RuntimeError("DenseRetriever is not ready. Call build() or load() first.")
        if top_k is None:
            top_k = self._cfg.get("top_k", 5)

        query_vec = self._encode([text])
        scores, indices = self._index.search(query_vec, top_k)

        valid_mask = indices[0] >= 0
        valid_indices = indices[0][valid_mask]
        valid_scores = scores[0][valid_mask]

        results = self._corpus.iloc[valid_indices].copy()
        results["dense_score"] = valid_scores
        results["retrieval_rank"] = range(1, len(results) + 1)
        return results.reset_index(drop=True)
