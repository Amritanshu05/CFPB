"""
Baseline text classifiers for CFPB complaint classification.

Implements three classical ML pipelines, each as a sklearn Pipeline:
  1. TF-IDF + Logistic Regression
  2. TF-IDF + Linear SVM
  3. TF-IDF + Multinomial Naive Bayes

Each pipeline is self-contained (vectorizer + classifier in a single object),
can be trained, saved, and loaded independently.

Design choice: we use sklearn Pipeline objects so that vectorization is always
tied to the model — preventing train/test leakage from separate fit calls.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV

from cfpb_assistant.utils.config import load_config, resolve_path, ensure_dir
from cfpb_assistant.utils.logging_utils import get_logger

logger = get_logger(__name__)


def _tfidf_vectorizer(cfg: dict) -> TfidfVectorizer:
    tfidf_cfg = cfg.get("tfidf", {})
    return TfidfVectorizer(
        max_features=tfidf_cfg.get("max_features", 50000),
        ngram_range=tuple(tfidf_cfg.get("ngram_range", [1, 2])),
        sublinear_tf=tfidf_cfg.get("sublinear_tf", True),
        min_df=tfidf_cfg.get("min_df", 2),
        strip_accents="unicode",
        analyzer="word",
        token_pattern=r"\b[a-z][a-z0-9]*\b",
    )


def build_logistic_regression() -> Pipeline:
    """TF-IDF + Logistic Regression pipeline."""
    cfg = load_config("classification")
    lr_cfg = cfg.get("logistic_regression", {})
    return Pipeline([
        ("tfidf", _tfidf_vectorizer(cfg)),
        ("clf", LogisticRegression(
            C=lr_cfg.get("C", 1.0),
            max_iter=lr_cfg.get("max_iter", 1000),
            solver=lr_cfg.get("solver", "lbfgs"),
        )),
    ])


def build_svm() -> Pipeline:
    """
    TF-IDF + Linear SVM pipeline.

    LinearSVC does not natively produce probabilities, so we wrap it with
    CalibratedClassifierCV (Platt scaling) for confidence estimation.
    """
    cfg = load_config("classification")
    svm_cfg = cfg.get("svm", {})
    svm = LinearSVC(
        C=svm_cfg.get("C", 1.0),
        max_iter=svm_cfg.get("max_iter", 5000),
    )
    calibrated_svm = CalibratedClassifierCV(svm, cv=3, method="sigmoid")
    return Pipeline([
        ("tfidf", _tfidf_vectorizer(cfg)),
        ("clf", calibrated_svm),
    ])


def build_naive_bayes() -> Pipeline:
    """TF-IDF + Multinomial Naive Bayes pipeline."""
    cfg = load_config("classification")
    nb_cfg = cfg.get("naive_bayes", {})
    return Pipeline([
        ("tfidf", _tfidf_vectorizer(cfg)),
        ("clf", MultinomialNB(alpha=nb_cfg.get("alpha", 1.0))),
    ])


BASELINE_BUILDERS = {
    "logistic_regression": build_logistic_regression,
    "svm": build_svm,
    "naive_bayes": build_naive_bayes,
}


def train_baseline(
    model_name: str,
    X_train: list[str],
    y_train: list[str],
) -> Pipeline:
    """
    Build and fit a named baseline model.

    Args:
        model_name: one of 'logistic_regression', 'svm', 'naive_bayes'
        X_train: list of cleaned narrative strings
        y_train: list of label strings

    Returns:
        Fitted sklearn Pipeline
    """
    if model_name not in BASELINE_BUILDERS:
        raise ValueError(f"Unknown model: {model_name}. Choose from {list(BASELINE_BUILDERS)}")

    logger.info(f"Training {model_name} ...")
    pipeline = BASELINE_BUILDERS[model_name]()
    pipeline.fit(X_train, y_train)
    logger.info(f"{model_name} training complete.")
    return pipeline


def save_model(pipeline: Pipeline, model_name: str) -> Path:
    """
    Save a fitted pipeline to outputs/models/<model_name>.pkl.

    Args:
        pipeline: fitted sklearn Pipeline
        model_name: filename stem

    Returns:
        Path to saved file
    """
    paths_cfg = load_config("paths")
    models_dir = ensure_dir(resolve_path(paths_cfg["outputs"]["models_dir"]))
    save_path = models_dir / f"{model_name}.pkl"
    with open(save_path, "wb") as f:
        pickle.dump(pipeline, f)
    logger.info(f"Model saved to {save_path}")
    return save_path


def load_model(model_name: str) -> Pipeline:
    """
    Load a fitted pipeline from outputs/models/<model_name>.pkl.

    Args:
        model_name: filename stem (without .pkl)

    Returns:
        Fitted sklearn Pipeline
    """
    paths_cfg = load_config("paths")
    models_dir = resolve_path(paths_cfg["outputs"]["models_dir"])
    load_path = models_dir / f"{model_name}.pkl"
    if not load_path.exists():
        raise FileNotFoundError(f"Model not found: {load_path}")
    with open(load_path, "rb") as f:
        return pickle.load(f)


def predict_with_confidence(
    pipeline: Pipeline,
    texts: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Predict labels and return class probabilities.

    Args:
        pipeline: fitted sklearn Pipeline (must have predict_proba)
        texts: list of complaint narrative strings

    Returns:
        (predicted_labels array, max_confidence array)
    """
    proba = pipeline.predict_proba(texts)
    labels = pipeline.classes_[np.argmax(proba, axis=1)]
    confidence = np.max(proba, axis=1)
    return labels, confidence
