"""
scripts/train_classifier.py

Train and evaluate classifiers (classical baselines + transformer) on the
processed train/test splits.

Run from the project root:
    python scripts/train_classifier.py --model all
    python scripts/train_classifier.py --model transformer

Options:
  --model    One of: logistic_regression, svm, naive_bayes, transformer,
             baselines (all classical), all (default: baselines)
  --no-save  Do not save trained models to disk
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cfpb_assistant.data.sampler import load_split
from cfpb_assistant.classification.baselines import (
    train_baseline,
    save_model,
    predict_with_confidence,
    BASELINE_BUILDERS,
)
from cfpb_assistant.evaluation.classification_eval import evaluate_classifier, compare_classifiers
from cfpb_assistant.utils.config import load_config
from cfpb_assistant.utils.logging_utils import get_logger

logger = get_logger("train_classifier", log_to_file=True)

TRANSFORMER_KEY = "transformer"


def _load_data():
    cfg = load_config("classification")
    text_col = cfg.get("text_column", "narrative_clean")
    label_col = cfg.get("target_column", "product_clean")

    logger.info(f"Loading train/test splits (text='{text_col}', label='{label_col}') ...")
    train_df = load_split("train").dropna(subset=[text_col, label_col])
    test_df = load_split("test").dropna(subset=[text_col, label_col])

    X_train = train_df[text_col].tolist()
    y_train = train_df[label_col].tolist()
    X_test = test_df[text_col].tolist()
    y_test = test_df[label_col].tolist()
    logger.info(f"Train: {len(X_train):,} | Test: {len(X_test):,}")
    return X_train, y_train, X_test, y_test


def run_baseline(model_name: str, do_save: bool) -> dict:
    X_train, y_train, X_test, y_test = _load_data()
    pipeline = train_baseline(model_name, X_train, y_train)
    y_pred, _ = predict_with_confidence(pipeline, X_test)
    results = evaluate_classifier(y_test, y_pred, model_name=model_name, save=True)
    if do_save:
        save_model(pipeline, model_name)
    return results


def run_transformer(do_save: bool) -> dict:
    # Lazy import so CPU-only users can still run the baselines.
    from cfpb_assistant.classification.transformer import TransformerClassifier

    cfg = load_config("classification").get("transformer", {})
    X_train, y_train, X_test, y_test = _load_data()

    max_train = cfg.get("max_train_samples")
    if max_train and max_train > 0 and len(X_train) > max_train:
        logger.info(f"Subsampling train set to {max_train:,} rows (from {len(X_train):,})")
        X_train = X_train[:max_train]
        y_train = y_train[:max_train]

    clf = TransformerClassifier()
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    results = evaluate_classifier(y_test, y_pred, model_name=TRANSFORMER_KEY, save=True)
    if do_save:
        clf.save(TRANSFORMER_KEY)
    return results


def main():
    choices = list(BASELINE_BUILDERS.keys()) + [TRANSFORMER_KEY, "baselines", "all"]
    parser = argparse.ArgumentParser(description="Train CFPB classifiers.")
    parser.add_argument("--model", default="baselines", choices=choices)
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()

    do_save = not args.no_save
    if args.model == "baselines":
        to_run = list(BASELINE_BUILDERS.keys())
    elif args.model == "all":
        to_run = list(BASELINE_BUILDERS.keys()) + [TRANSFORMER_KEY]
    else:
        to_run = [args.model]

    all_results = []
    for name in to_run:
        logger.info(f"\n{'='*50}\nModel: {name}\n{'='*50}")
        if name == TRANSFORMER_KEY:
            results = run_transformer(do_save)
        else:
            results = run_baseline(name, do_save)
        all_results.append(results)

    if len(all_results) > 1:
        logger.info("\n=== Comparison ===")
        compare_classifiers(all_results)


if __name__ == "__main__":
    main()
