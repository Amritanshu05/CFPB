"""
Generation evaluation — metrics for the full complaint-to-resolution pipeline.

The generator outputs a structured JSON object. Standard NLG metrics
(BLEU / ROUGE / BERTScore) don't directly apply because there are no
reference "gold" resolution summaries in the CFPB dataset. So we
evaluate along four orthogonal axes that ARE computable from the
dataset alone:

  1. Schema validity
       Fraction of outputs that produce every required key with the
       correct Python type. A structured pipeline must hit 1.00 here.

  2. Label fidelity
       Does `predicted_issue` match the gold `product_clean` label?
       This is the classification component measured end-to-end (it
       should roughly match the standalone classifier accuracy; large
       gaps indicate integration bugs).

  3. Grounding proxy (faithfulness)
       Fraction of non-stopword content tokens in `complaint_summary`
       that appear in the original complaint text. Paraphrasing LLMs
       will drift away from 1.0; hallucinations drop this sharply.
       This is a proxy, not a full faithfulness evaluation.

  4. Abstention behavior
       Confidence distribution + rate of `needs_human_review=True`.
       We report mean/median confidence and the abstention rate.

Mock-mode is explicitly flagged in the report so nobody mistakes the
placeholder summary behavior for a real evaluated LLM output.
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from statistics import mean, median

from cfpb_assistant.utils.config import load_config, resolve_path, ensure_dir
from cfpb_assistant.utils.logging_utils import get_logger

logger = get_logger(__name__)

REQUIRED_SCHEMA = {
    "complaint_summary": str,
    "predicted_issue": str,
    "similar_cases": list,
    "recommended_next_step": str,
    "confidence": (int, float),
    "needs_human_review": bool,
    "generation_mocked": bool,
}

# Very small stopword list; keep dependency-free.
_STOPWORDS = {
    "a", "an", "and", "or", "but", "if", "the", "is", "was", "were",
    "to", "of", "in", "on", "for", "with", "at", "by", "from", "this",
    "that", "it", "as", "be", "been", "are", "i", "my", "me", "they",
    "them", "he", "she", "we", "you", "your", "his", "her", "their",
    "our", "its", "so", "not", "no", "do", "does", "did", "have", "has",
    "had", "will", "would", "can", "could", "should", "may", "might",
    "about", "into", "than", "then", "when", "while", "there", "here",
}


def _tokens(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z]+", text.lower()) if t not in _STOPWORDS and len(t) > 2}


def _check_schema(obj: dict) -> tuple[bool, list[str]]:
    missing = []
    for key, expected in REQUIRED_SCHEMA.items():
        if key not in obj:
            missing.append(f"missing:{key}")
            continue
        if not isinstance(obj[key], expected):
            missing.append(f"wrong_type:{key}")
    return (len(missing) == 0), missing


def _grounding_score(summary: str, source: str) -> float:
    """Jaccard-like: |summary_tokens ∩ source_tokens| / |summary_tokens|."""
    s_tokens = _tokens(summary)
    if not s_tokens:
        return 0.0
    src_tokens = _tokens(source)
    if not src_tokens:
        return 0.0
    return len(s_tokens & src_tokens) / len(s_tokens)


def evaluate_generation(
    generator,
    samples: list[dict],
    model_name: str = "pipeline_generation",
    save: bool = True,
) -> dict:
    """
    Evaluate the end-to-end generator on a list of samples.

    Each sample must contain:
      - narrative_clean : input complaint text
      - product_clean   : gold label for the label-fidelity metric
    """
    n = 0
    schema_ok = 0
    label_match = 0
    mock_count = 0
    grounding_scores: list[float] = []
    confidences: list[float] = []
    abstained: list[int] = []
    schema_issues: list[str] = []

    per_example = []

    for row in samples:
        narrative = row.get("narrative_clean", "")
        gold_label = row.get("product_clean", "")
        if not narrative or not gold_label:
            continue
        n += 1
        try:
            out = generator.process(narrative)
        except Exception as e:
            logger.warning(f"Generator failed on sample: {e}")
            continue

        ok, issues = _check_schema(out)
        schema_ok += int(ok)
        schema_issues.extend(issues)

        if ok:
            if out["predicted_issue"] == gold_label:
                label_match += 1
            g = _grounding_score(out["complaint_summary"], narrative)
            grounding_scores.append(g)
            confidences.append(float(out["confidence"]))
            abstained.append(int(bool(out["needs_human_review"])))
            if bool(out.get("generation_mocked", False)):
                mock_count += 1

            per_example.append({
                "gold_label": gold_label,
                "predicted_issue": out["predicted_issue"],
                "confidence": float(out["confidence"]),
                "needs_human_review": bool(out["needs_human_review"]),
                "grounding": round(g, 4),
                "mocked": bool(out.get("generation_mocked", False)),
            })

    metrics = {
        "model": model_name,
        "timestamp": datetime.now().isoformat(),
        "num_samples": n,
        "schema_validity": round(schema_ok / n, 4) if n else 0.0,
        "label_fidelity": round(label_match / n, 4) if n else 0.0,
        "grounding_mean": round(mean(grounding_scores), 4) if grounding_scores else 0.0,
        "grounding_median": round(median(grounding_scores), 4) if grounding_scores else 0.0,
        "confidence_mean": round(mean(confidences), 4) if confidences else 0.0,
        "confidence_median": round(median(confidences), 4) if confidences else 0.0,
        "abstention_rate": round(mean(abstained), 4) if abstained else 0.0,
        "mock_fraction": round(mock_count / n, 4) if n else 0.0,
        "notes": (
            "Grounding is a token-overlap proxy, not a full faithfulness metric. "
            "In mock mode, summaries are deterministic and often near-verbatim; "
            "this inflates grounding relative to real LLM paraphrasing."
        ),
        "schema_issue_counts": _count_issues(schema_issues),
        "per_example": per_example,
    }

    logger.info(
        f"[{model_name}] schema={metrics['schema_validity']:.3f} "
        f"label_fid={metrics['label_fidelity']:.3f} "
        f"grounding={metrics['grounding_mean']:.3f} "
        f"abstain={metrics['abstention_rate']:.3f} "
        f"mock_frac={metrics['mock_fraction']:.2f}"
    )

    if save:
        paths_cfg = load_config("paths")
        results_dir = ensure_dir(resolve_path(paths_cfg["outputs"]["results_dir"]))
        out_path = results_dir / f"{model_name}_results.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Generation results saved to {out_path}")

    return metrics


def _count_issues(issues: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in issues:
        counts[item] = counts.get(item, 0) + 1
    return counts
