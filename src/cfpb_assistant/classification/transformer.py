"""
Transformer-based classifier for CFPB complaint classification.

Fine-tunes a HuggingFace sequence-classification model (DistilBERT by
default) on the cleaned complaint narratives. Uses GPU if available.

Design notes:
  - Training and inference share a single `TransformerClassifier` object
    that mirrors the sklearn-style interface (`predict`, `predict_proba`,
    `classes_`) so it is a drop-in replacement in `evaluate.py` and
    `run_pipeline.py`.
  - Weights and the label encoder are saved together in one directory
    so the classifier is fully reloadable.
  - Batch size and epochs are configured in `configs/classification.yaml`
    under the `transformer:` block.
"""

from __future__ import annotations

import json
from pathlib import Path
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)

from cfpb_assistant.utils.config import load_config, resolve_path, ensure_dir
from cfpb_assistant.utils.logging_utils import get_logger

logger = get_logger(__name__)


# ------------------------------------------------------------------
# Device selection
# ------------------------------------------------------------------

def get_device() -> torch.device:
    """Return CUDA device if available, otherwise CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def log_device_info(device: torch.device) -> None:
    if device.type == "cuda":
        name = torch.cuda.get_device_name(0)
        total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        logger.info(f"Using GPU: {name} ({total:.1f} GB VRAM)")
    else:
        logger.info("Using CPU (no CUDA device detected)")


# ------------------------------------------------------------------
# Dataset
# ------------------------------------------------------------------

class _ComplaintDataset(Dataset):
    """Simple dataset that tokenizes on the fly (fine for ~30-50k rows)."""

    def __init__(self, texts: list[str], labels: list[int] | None, tokenizer, max_length: int):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict:
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


# ------------------------------------------------------------------
# Classifier wrapper
# ------------------------------------------------------------------

@dataclass
class _LabelEncoder:
    """Minimal label <-> id encoder to avoid a sklearn LabelEncoder dependency at inference."""
    labels: list[str]

    def to_id(self, label: str) -> int:
        return self.labels.index(label)

    def to_label(self, idx: int) -> str:
        return self.labels[idx]


class TransformerClassifier:
    """
    Fine-tunes a transformer for sequence classification.

    Sklearn-style surface:
      - fit(X, y)         : train on cleaned narratives + product labels
      - predict(X)        : return array of predicted label strings
      - predict_proba(X)  : return [n_samples, n_classes] probability array
      - classes_          : array of label strings in probability column order
    """

    def __init__(self, config: dict | None = None):
        cfg = config or load_config("classification").get("transformer", {})
        self._model_name = cfg.get("model_name", "distilbert-base-uncased")
        self._max_length = cfg.get("max_length", 256)
        self._batch_size = cfg.get("batch_size", 16)
        self._eval_batch_size = cfg.get("eval_batch_size", 32)
        self._lr = float(cfg.get("learning_rate", 2e-5))
        self._epochs = int(cfg.get("num_epochs", 3))
        self._warmup_steps = int(cfg.get("warmup_steps", 100))
        self._weight_decay = float(cfg.get("weight_decay", 0.01))

        self._device = get_device()
        log_device_info(self._device)

        self._tokenizer = None
        self._model = None
        self._encoder: _LabelEncoder | None = None

    # ------------------------------------------------------------------
    # Properties expected by sklearn-style eval code
    # ------------------------------------------------------------------

    @property
    def classes_(self) -> np.ndarray:
        if self._encoder is None:
            raise RuntimeError("Classifier is not trained/loaded yet.")
        return np.array(self._encoder.labels)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(self, X: list[str], y: list[str]) -> "TransformerClassifier":
        labels = sorted(set(y))
        self._encoder = _LabelEncoder(labels=labels)
        num_labels = len(labels)

        logger.info(
            f"Training {self._model_name} | {num_labels} classes | "
            f"batch={self._batch_size} | epochs={self._epochs} | lr={self._lr}"
        )

        self._tokenizer = AutoTokenizer.from_pretrained(self._model_name)
        self._model = AutoModelForSequenceClassification.from_pretrained(
            self._model_name,
            num_labels=num_labels,
        ).to(self._device)

        y_ids = [self._encoder.to_id(lbl) for lbl in y]
        dataset = _ComplaintDataset(X, y_ids, self._tokenizer, self._max_length)
        loader = DataLoader(
            dataset,
            batch_size=self._batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=(self._device.type == "cuda"),
        )

        optimizer = torch.optim.AdamW(
            self._model.parameters(),
            lr=self._lr,
            weight_decay=self._weight_decay,
        )
        total_steps = len(loader) * self._epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self._warmup_steps,
            num_training_steps=total_steps,
        )

        self._model.train()
        for epoch in range(self._epochs):
            running_loss = 0.0
            for step, batch in enumerate(loader):
                batch = {k: v.to(self._device, non_blocking=True) for k, v in batch.items()}
                optimizer.zero_grad()
                outputs = self._model(**batch)
                loss = outputs.loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self._model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                running_loss += loss.item()

                if (step + 1) % 100 == 0:
                    logger.info(
                        f"  epoch {epoch+1}/{self._epochs} step {step+1}/{len(loader)} "
                        f"loss={running_loss / (step+1):.4f}"
                    )
            logger.info(
                f"Epoch {epoch+1} done | avg_loss={running_loss / max(len(loader),1):.4f}"
            )

        return self

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict_proba(self, X: list[str]) -> np.ndarray:
        if self._model is None or self._tokenizer is None or self._encoder is None:
            raise RuntimeError("Classifier is not trained/loaded yet.")

        self._model.eval()
        dataset = _ComplaintDataset(list(X), None, self._tokenizer, self._max_length)
        loader = DataLoader(
            dataset,
            batch_size=self._eval_batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=(self._device.type == "cuda"),
        )

        all_probs: list[np.ndarray] = []
        for batch in loader:
            batch = {k: v.to(self._device, non_blocking=True) for k, v in batch.items()}
            logits = self._model(**batch).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            all_probs.append(probs)
        return np.concatenate(all_probs, axis=0)

    def predict(self, X: list[str]) -> np.ndarray:
        proba = self.predict_proba(X)
        idxs = np.argmax(proba, axis=1)
        return np.array([self._encoder.to_label(i) for i in idxs])

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, name: str = "transformer") -> Path:
        paths_cfg = load_config("paths")
        out_dir = ensure_dir(resolve_path(paths_cfg["outputs"]["models_dir"]) / name)
        self._model.save_pretrained(out_dir)
        self._tokenizer.save_pretrained(out_dir)
        with open(out_dir / "label_encoder.json", "w", encoding="utf-8") as f:
            json.dump({"labels": self._encoder.labels}, f)
        with open(out_dir / "config.json.meta", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "model_name": self._model_name,
                    "max_length": self._max_length,
                },
                f,
            )
        logger.info(f"Transformer classifier saved to {out_dir}")
        return out_dir

    @classmethod
    def load(cls, name: str = "transformer") -> "TransformerClassifier":
        paths_cfg = load_config("paths")
        model_dir = resolve_path(paths_cfg["outputs"]["models_dir"]) / name
        if not model_dir.exists():
            raise FileNotFoundError(f"Transformer model dir not found: {model_dir}")

        with open(model_dir / "label_encoder.json", "r", encoding="utf-8") as f:
            labels = json.load(f)["labels"]

        with open(model_dir / "config.json.meta", "r", encoding="utf-8") as f:
            meta = json.load(f)

        clf = cls({"model_name": meta["model_name"], "max_length": meta["max_length"]})
        clf._tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
        clf._model = AutoModelForSequenceClassification.from_pretrained(
            str(model_dir)
        ).to(clf._device)
        clf._encoder = _LabelEncoder(labels=labels)
        logger.info(
            f"Transformer classifier loaded from {model_dir} (device={clf._device})"
        )
        return clf
