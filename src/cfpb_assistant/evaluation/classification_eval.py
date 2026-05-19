"""
Classification evaluation utilities.

Computes standard metrics for multi-class text classification:
  - Accuracy
  - Macro F1
  - Weighted F1
  - Per-class F1 (classification report)
  - Confusion matrix

Results are saved as JSON to outputs/results/ for reproducibility.
"""

from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

from cfpb_assistant.utils.config import load_config, resolve_path, ensure_dir
from cfpb_assistant.utils.logging_utils import get_logger

logger = get_logger(__name__)


def evaluate_classifier(
    y_true: list[str] | np.ndarray,
    y_pred: list[str] | np.ndarray,
    model_name: str,
    save: bool = True,
) -> dict:
    """
    Compute and optionally save classification metrics.

    Args:
        y_true: ground-truth labels
        y_pred: predicted labels
        model_name: identifier used in the results filename
        save: if True, write results JSON to outputs/results/

    Returns:
        dict with metrics
    """
    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    report = classification_report(y_true, y_pred, zero_division=0, output_dict=True)

    labels = sorted(set(list(y_true)) | set(list(y_pred)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    results = {
        "model": model_name,
        "timestamp": datetime.now().isoformat(),
        "accuracy": round(float(accuracy), 4),
        "macro_f1": round(float(macro_f1), 4),
        "weighted_f1": round(float(weighted_f1), 4),
        "per_class": {
            k: {m: round(v, 4) for m, v in v_dict.items() if m != "support"}
            | {"support": int(v_dict["support"])}
            for k, v_dict in report.items()
            if isinstance(v_dict, dict) and k not in ("macro avg", "weighted avg")
        },
        "confusion_matrix": {
            "labels": labels,
            "matrix": cm.tolist(),
        },
    }

    logger.info(
        f"[{model_name}] accuracy={accuracy:.4f}  macro_f1={macro_f1:.4f}  weighted_f1={weighted_f1:.4f}"
    )

    if save:
        paths_cfg = load_config("paths")
        results_dir = ensure_dir(resolve_path(paths_cfg["outputs"]["results_dir"]))
        out_path = results_dir / f"{model_name}_classification_results.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {out_path}")

        # Also save a plain CSV for confusion matrix for paper artifacts.
        cm_csv_path = results_dir / f"{model_name}_confusion_matrix.csv"
        with open(cm_csv_path, "w", encoding="utf-8") as f:
            f.write("," + ",".join(labels) + "\n")
            for lbl, row in zip(labels, cm.tolist()):
                f.write(lbl + "," + ",".join(str(v) for v in row) + "\n")
        logger.info(f"Confusion matrix saved to {cm_csv_path}")

    return results


def compare_classifiers(results_list: list[dict]) -> None:
    """
    Print a formatted comparison table of multiple classifier results.

    Args:
        results_list: list of dicts as returned by evaluate_classifier
    """
    header = f"{'Model':<30} {'Accuracy':>10} {'Macro F1':>10} {'Weighted F1':>12}"
    separator = "-" * len(header)
    print(separator)
    print(header)
    print(separator)
    for r in results_list:
        print(
            f"{r['model']:<30} {r['accuracy']:>10.4f} {r['macro_f1']:>10.4f} {r['weighted_f1']:>12.4f}"
        )
    print(separator)
