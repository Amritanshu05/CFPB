"""
Text preprocessing and label normalization for CFPB complaint data.

Responsibilities:
  - Clean raw narrative text (normalize whitespace, strip boilerplate, lowercase)
  - Map verbose product strings to short canonical labels
  - Clip narrative length
  - Produce a DataFrame ready for classification and retrieval
"""

from __future__ import annotations

import re
import pandas as pd

from cfpb_assistant.utils.config import load_config
from cfpb_assistant.utils.logging_utils import get_logger

logger = get_logger(__name__)

# Common CFPB anonymization placeholder — not meaningful text
_REDACTION_PATTERNS = [
    r"\bXXXX+\b",          # credit card / account number redactions
    r"\bxx+\b",             # lowercase variant
    r"\bXX/XX/\d{4}\b",    # date redactions
    r"\bXX/XX/XX\b",
]
_REDACTION_RE = re.compile("|".join(_REDACTION_PATTERNS), flags=re.IGNORECASE)

# Collapse runs of whitespace / newlines into a single space
_WHITESPACE_RE = re.compile(r"\s+")


def clean_narrative(text: str, max_length: int = 5000) -> str:
    """
    Clean a single complaint narrative string.

    Steps:
      1. Strip leading/trailing whitespace
      2. Replace redaction tokens (XXXX, XX/XX/XXXX) with a space
      3. Collapse internal whitespace
      4. Lowercase
      5. Clip to max_length characters

    Args:
        text: raw narrative string
        max_length: maximum character length after cleaning

    Returns:
        Cleaned string
    """
    if not isinstance(text, str):
        return ""

    text = text.strip()
    text = _REDACTION_RE.sub(" ", text)
    text = _WHITESPACE_RE.sub(" ", text)
    text = text.lower()
    text = text[:max_length]
    return text.strip()


def normalize_product_label(product: str, product_map: dict) -> str:
    """
    Map a raw CFPB product string to a short canonical label.

    Args:
        product: raw product string from the CSV
        product_map: mapping dict from config (preprocessing.yaml)

    Returns:
        Canonical short label, or "other" if not in the map
    """
    if not isinstance(product, str):
        return "other"
    return product_map.get(product.strip(), "other")


def preprocess_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all preprocessing steps to a single chunk.

    Adds columns:
      - narrative_clean: cleaned narrative text
      - product_clean: canonical product label

    Args:
        chunk: raw chunk from loader.py (must have 'narrative', 'product' columns)

    Returns:
        DataFrame with new clean columns; rows with empty narrative_clean are dropped
    """
    cfg = load_config("preprocessing")
    max_len = cfg.get("max_narrative_length", 5000)
    # Post-cleaning minimum length: applied AFTER redaction stripping so
    # that narratives that were ~100 chars of "XXXX XX/XX/XXXX ..." and
    # collapse to a handful of tokens are filtered out.
    min_clean_len = cfg.get("min_clean_narrative_length", 50)
    product_map = cfg.get("product_map", {})

    chunk = chunk.copy()

    chunk["narrative_clean"] = chunk["narrative"].apply(
        lambda t: clean_narrative(t, max_length=max_len)
    )

    # Post-cleaning length filter: drops garbage short narratives that
    # only survived the raw-length filter because of XXXX padding.
    before = len(chunk)
    chunk = chunk[chunk["narrative_clean"].str.len() >= min_clean_len].copy()
    dropped = before - len(chunk)
    if dropped > 0:
        logger.debug(
            f"Dropped {dropped} rows with narrative_clean < {min_clean_len} chars"
        )

    # Normalize product labels
    if "product" in chunk.columns:
        chunk["product_clean"] = chunk["product"].apply(
            lambda p: normalize_product_label(p, product_map)
        )
    else:
        chunk["product_clean"] = "other"

    # Normalize issue label: lowercase, replace spaces with underscores
    if "issue" in chunk.columns:
        chunk["issue_clean"] = (
            chunk["issue"]
            .fillna("unknown")
            .str.lower()
            .str.strip()
            .str.replace(r"\s+", "_", regex=True)
            .str.replace(r"[^a-z0-9_]", "", regex=True)
        )
    else:
        chunk["issue_clean"] = "unknown"

    return chunk
