"""
Data loader for the CFPB Consumer Complaint dataset.

The raw CSV is ~8 GB and ~25 million rows. This module reads it in chunks,
never loading the full file into memory at once.

Key responsibilities:
  - Chunked iteration over the raw CSV
  - Column renaming to consistent internal names
  - Type normalization (dates, booleans)
  - Filtering rows without a complaint narrative
"""

from __future__ import annotations

import pandas as pd
from pathlib import Path
from typing import Iterator

from cfpb_assistant.utils.config import load_config, resolve_path
from cfpb_assistant.utils.logging_utils import get_logger

logger = get_logger(__name__)

# Canonical internal column names mapped from raw CFPB CSV headers
COLUMN_RENAME = {
    "Date received": "date_received",
    "Product": "product",
    "Sub-product": "sub_product",
    "Issue": "issue",
    "Sub-issue": "sub_issue",
    "Consumer complaint narrative": "narrative",
    "Company public response": "company_public_response",
    "Company": "company",
    "State": "state",
    "ZIP code": "zip_code",
    "Tags": "tags",
    "Consumer consent provided?": "consumer_consent",
    "Submitted via": "submitted_via",
    "Date sent to company": "date_sent",
    "Company response to consumer": "company_response",
    "Timely response?": "timely_response",
    "Consumer disputed?": "consumer_disputed",
    "Complaint ID": "complaint_id",
}

# Columns we actually need downstream — drop everything else to save memory
COLUMNS_TO_KEEP = [
    "complaint_id",
    "date_received",
    "product",
    "sub_product",
    "issue",
    "sub_issue",
    "narrative",
    "company_response",
    "consumer_disputed",
    "state",
    "submitted_via",
    "timely_response",
]


def iter_raw_chunks(chunk_size: int | None = None) -> Iterator[pd.DataFrame]:
    """
    Yield cleaned DataFrame chunks from the raw CFPB CSV.

    Each chunk has:
      - canonical column names (see COLUMN_RENAME)
      - only the columns in COLUMNS_TO_KEEP
      - date_received parsed to datetime

    Args:
        chunk_size: number of rows per chunk; defaults to preprocessing config value

    Yields:
        pd.DataFrame chunks
    """
    cfg = load_config("preprocessing")
    paths_cfg = load_config("paths")
    if chunk_size is None:
        chunk_size = cfg["chunk_size"]

    raw_path = resolve_path(paths_cfg["data"]["raw_csv"])
    if not raw_path.exists():
        raise FileNotFoundError(
            f"Raw CSV not found at: {raw_path}\n"
            "Place the CFPB consumer_complaints.csv at data/raw/cfbp/consumer_complaints.csv"
        )

    logger.info(f"Reading raw CSV in chunks of {chunk_size:,} rows: {raw_path}")

    reader = pd.read_csv(
        raw_path,
        chunksize=chunk_size,
        dtype=str,          # read everything as string first; we cast selectively
        low_memory=False,
    )

    for i, chunk in enumerate(reader):
        # Rename to canonical names
        chunk = chunk.rename(columns=COLUMN_RENAME)

        # Keep only needed columns (ignore any that don't exist in this chunk)
        keep = [c for c in COLUMNS_TO_KEEP if c in chunk.columns]
        chunk = chunk[keep]

        # Parse date
        if "date_received" in chunk.columns:
            chunk["date_received"] = pd.to_datetime(
                chunk["date_received"], errors="coerce"
            )

        logger.debug(f"Loaded chunk {i} with {len(chunk):,} rows")
        yield chunk


def load_with_narratives_only(chunk_size: int | None = None) -> Iterator[pd.DataFrame]:
    """
    Like iter_raw_chunks, but yields only rows that have a non-empty narrative.

    The CFPB dataset has many rows where consumers did not provide a narrative
    (consent not given or left blank). Those rows are not useful for our pipeline.

    Yields:
        pd.DataFrame chunks filtered to rows with narratives
    """
    cfg = load_config("preprocessing")
    min_len = cfg.get("min_narrative_length", 50)

    for chunk in iter_raw_chunks(chunk_size=chunk_size):
        if "narrative" not in chunk.columns:
            continue
        mask = (
            chunk["narrative"].notna()
            & (chunk["narrative"].str.strip().str.len() >= min_len)
        )
        filtered = chunk[mask].copy()
        if len(filtered) > 0:
            yield filtered
