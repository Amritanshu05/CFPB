"""
Sampler and train/val/test splitter for the CFPB pipeline.

Because the full dataset is ~25M rows, most experiments use a stratified sample.
This module:
  1. Scans the full raw CSV in chunks
  2. Keeps only rows with a valid narrative
  3. Applies preprocessing
  4. Draws a stratified sample (by product_clean label)
  5. Saves the sample as a Parquet file for fast downstream loading
  6. Produces stratified train/validation/test splits and saves them separately
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from pathlib import Path

from cfpb_assistant.data.loader import load_with_narratives_only
from cfpb_assistant.data.preprocessor import preprocess_chunk
from cfpb_assistant.utils.config import load_config, resolve_path, ensure_dir
from cfpb_assistant.utils.logging_utils import get_logger

logger = get_logger(__name__)


def build_sample(force: bool = False) -> Path:
    """
    Stream the full raw CSV, preprocess, and save a stratified sample.

    The sample is stratified by product_clean so all categories are represented.
    Saves to data/samples/complaints_sample.parquet.

    Args:
        force: if True, rebuild even if the sample file already exists

    Returns:
        Path to the saved sample Parquet file
    """
    cfg = load_config("preprocessing")
    paths_cfg = load_config("paths")

    sample_path = resolve_path(paths_cfg["data"]["samples_dir"]) / "complaints_sample.parquet"
    ensure_dir(sample_path.parent)

    if sample_path.exists() and not force:
        logger.info(f"Sample already exists at {sample_path} — skipping rebuild. Use force=True to rebuild.")
        return sample_path

    sample_frac = cfg.get("sample_fraction", 0.05)
    sample_max = cfg.get("sample_max_rows", 50000)
    seed = cfg.get("random_seed", 42)

    rng = np.random.default_rng(seed)
    collected: list[pd.DataFrame] = []
    total_input = 0
    total_kept = 0

    logger.info(
        f"Building sample (fraction={sample_frac}, max={sample_max:,}, seed={seed}) ..."
    )

    for chunk in load_with_narratives_only():
        processed = preprocess_chunk(chunk)
        total_input += len(chunk)

        # Reservoir-style: keep each row with probability = sample_frac
        mask = rng.random(len(processed)) < sample_frac
        selected = processed[mask]

        if len(selected) > 0:
            collected.append(selected)
            total_kept += len(selected)

        if total_kept >= sample_max:
            logger.info(f"Reached sample_max ({sample_max:,}) after scanning {total_input:,} narrative rows")
            break

        if total_input % 500_000 == 0:
            logger.info(f"  Scanned {total_input:,} rows, collected {total_kept:,} so far ...")

    if not collected:
        raise RuntimeError("No data was collected. Check that the raw CSV has narrative rows.")

    sample_df = pd.concat(collected, ignore_index=True)

    # Trim to sample_max if we collected slightly more
    if len(sample_df) > sample_max:
        sample_df = sample_df.sample(n=sample_max, random_state=seed).reset_index(drop=True)

    logger.info(f"Sample built: {len(sample_df):,} rows")
    logger.info(f"Product label distribution:\n{sample_df['product_clean'].value_counts().to_string()}")

    sample_df.to_parquet(sample_path, index=False, engine="pyarrow")
    logger.info(f"Sample saved to {sample_path}")

    return sample_path


def build_splits(force: bool = False) -> dict[str, Path]:
    """
    Load the sample and produce stratified train/validation/test splits.
    Saves each split as a Parquet file in data/processed/.

    Args:
        force: if True, rebuild splits even if they already exist

    Returns:
        dict with keys 'train', 'validation', 'test' mapping to Path objects
    """
    from sklearn.model_selection import train_test_split

    cfg = load_config("preprocessing")
    paths_cfg = load_config("paths")
    split_cfg = cfg["split"]
    seed = cfg.get("random_seed", 42)

    processed_dir = ensure_dir(resolve_path(paths_cfg["data"]["processed_dir"]))
    split_paths = {
        "train": processed_dir / "train.parquet",
        "validation": processed_dir / "validation.parquet",
        "test": processed_dir / "test.parquet",
    }

    if all(p.exists() for p in split_paths.values()) and not force:
        logger.info("Train/val/test splits already exist — skipping. Use force=True to rebuild.")
        return split_paths

    sample_path = resolve_path(paths_cfg["data"]["samples_dir"]) / "complaints_sample.parquet"
    if not sample_path.exists():
        logger.info("Sample not found — building it first ...")
        build_sample()

    df = pd.read_parquet(sample_path)
    logger.info(f"Loaded sample: {len(df):,} rows for splitting")

    label_col = cfg["label_columns"]["primary"]

    # Drop rows where the label is missing
    df = df[df[label_col].notna()].copy()

    train_ratio = split_cfg["train"]
    val_ratio = split_cfg["validation"]
    test_ratio = split_cfg["test"]

    # First split: separate test set
    train_val, test = train_test_split(
        df,
        test_size=test_ratio,
        stratify=df[label_col],
        random_state=seed,
    )

    # Second split: separate validation from remaining
    val_relative = val_ratio / (train_ratio + val_ratio)
    train, val = train_test_split(
        train_val,
        test_size=val_relative,
        stratify=train_val[label_col],
        random_state=seed,
    )

    train.reset_index(drop=True).to_parquet(split_paths["train"], index=False)
    val.reset_index(drop=True).to_parquet(split_paths["validation"], index=False)
    test.reset_index(drop=True).to_parquet(split_paths["test"], index=False)

    logger.info(
        f"Splits saved — train: {len(train):,}, val: {len(val):,}, test: {len(test):,}"
    )
    return split_paths


def load_split(split_name: str) -> pd.DataFrame:
    """
    Load a saved split by name.

    Args:
        split_name: one of 'train', 'validation', 'test'

    Returns:
        pd.DataFrame
    """
    paths_cfg = load_config("paths")
    processed_dir = resolve_path(paths_cfg["data"]["processed_dir"])
    path = processed_dir / f"{split_name}.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"Split '{split_name}' not found at {path}. Run build_splits() first."
        )
    return pd.read_parquet(path)
