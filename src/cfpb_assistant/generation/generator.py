"""
Structured resolution generator — the final stage of the pipeline.

Wires together:
  1. Input normalization (same cleaning used at training time)
  2. A trained classifier (predicts issue category + calibrated probability)
  3. A retriever (finds similar historical cases)
  4. An LLM call (generates structured JSON resolution), OR an explicit mock
  5. A confidence-aware abstention check

Design choices for correctness:
  - All text entering the classifier/retriever is passed through
    `preprocessor.clean_narrative()` so training and inference share the
    exact same text representation.
  - Mock mode does NOT emit a fake calibrated confidence. It emits
    `confidence = None` and an explicit flag `generation_mocked = True`.
    The final confidence is then driven solely by the classifier probability.
  - The confidence surfaced in the final output is ALWAYS the classifier
    probability in mock mode. Only when a real LLM is used do we combine
    classifier confidence with LLM-reported confidence (as min()).

Output schema (stable):
  {
    "complaint_summary": str,
    "predicted_issue": str,
    "similar_cases": [str, ...],
    "recommended_next_step": str,
    "confidence": float,
    "needs_human_review": bool,
    "generation_mocked": bool       # true when mock provider produced summary
  }
"""

from __future__ import annotations

import json
import os
import re

import numpy as np

from cfpb_assistant.utils.config import load_config
from cfpb_assistant.utils.logging_utils import get_logger
from cfpb_assistant.generation.prompts import build_resolution_prompt
from cfpb_assistant.extraction.extractor import extract
from cfpb_assistant.data.preprocessor import clean_narrative

logger = get_logger(__name__)

OUTPUT_SCHEMA_KEYS = [
    "complaint_summary",
    "predicted_issue",
    "similar_cases",
    "recommended_next_step",
    "confidence",
    "needs_human_review",
    "generation_mocked",
]

ALLOWED_NEXT_STEPS = {
    "Request supporting documents",
    "Escalate to senior analyst",
    "Contact company for response",
    "Mark as resolved - similar cases closed",
    "Refer to legal team",
    "Request more information from consumer",
}


class ResolutionGenerator:
    """End-to-end complaint resolution pipeline."""

    def __init__(self, classifier=None, retriever=None, provider: str | None = None):
        self._classifier = classifier
        self._retriever = retriever
        self._cfg = load_config("generation")
        self._confidence_threshold = self._cfg.get("confidence_threshold", 0.6)
        self._num_cases = self._cfg.get("num_retrieved_cases", 3)
        self._provider = provider if provider is not None else self._cfg.get("provider", "mock")
        logger.info(f"ResolutionGenerator initialised with provider='{self._provider}'")

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def process(self, complaint_text: str) -> dict:
        """
        Run the full pipeline for a single complaint.

        The input is first normalized with `clean_narrative` so that
        classifier and retriever see the same text representation that
        was used during training.
        """
        cleaned = clean_narrative(complaint_text)
        if not cleaned:
            return self._empty_result("input was empty after cleaning")

        extraction = extract(cleaned)
        logger.debug(f"Extraction: {extraction.to_dict()}")

        predicted_issue, classifier_confidence = self._classify(cleaned)
        logger.debug(f"Classification: {predicted_issue} (conf={classifier_confidence:.3f})")

        similar_cases = self._retrieve(cleaned)
        logger.debug(f"Retrieved {len(similar_cases)} similar cases")

        raw_output, mocked = self._generate(cleaned, predicted_issue, similar_cases)

        return self._validate_and_finalize(
            raw_output=raw_output,
            predicted_issue=predicted_issue,
            classifier_confidence=classifier_confidence,
            mocked=mocked,
            similar_cases=similar_cases,
        )

    # ------------------------------------------------------------------
    # Pipeline stages
    # ------------------------------------------------------------------

    def _classify(self, text: str) -> tuple[str, float]:
        """Return (predicted_label, max_probability)."""
        if self._classifier is None:
            return "unknown", 0.0
        proba = self._classifier.predict_proba([text])[0]
        idx = int(np.argmax(proba))
        label = str(self._classifier.classes_[idx])
        confidence = float(proba[idx])
        return label, confidence

    def _retrieve(self, text: str) -> list[dict]:
        if self._retriever is None:
            return []
        try:
            results_df = self._retriever.query(text, top_k=self._num_cases)
            return results_df.to_dict(orient="records")
        except Exception as e:
            logger.warning(f"Retrieval failed: {e}")
            return []

    def _generate(
        self,
        complaint_text: str,
        predicted_issue: str,
        similar_cases: list[dict],
    ) -> tuple[dict, bool]:
        """
        Returns (raw_output_dict, mocked_flag).
        mocked_flag is True iff the output came from the rule-based mock path.
        """
        system_prompt, user_prompt = build_resolution_prompt(
            complaint_text=complaint_text,
            predicted_issue=predicted_issue,
            similar_cases=similar_cases,
        )

        if self._provider == "mock":
            return self._mock_generate(complaint_text, predicted_issue, similar_cases), True
        elif self._provider == "openai":
            return self._openai_generate(system_prompt, user_prompt), False
        elif self._provider == "anthropic":
            return self._anthropic_generate(system_prompt, user_prompt), False
        else:
            raise ValueError(f"Unknown provider: {self._provider}")

    # ------------------------------------------------------------------
    # Mock generator (explicit placeholder — no synthetic confidence)
    # ------------------------------------------------------------------

    def _mock_generate(
        self,
        complaint_text: str,
        predicted_issue: str,
        similar_cases: list[dict],
    ) -> dict:
        """
        Deterministic rule-based output for when no LLM is available.

        IMPORTANT: this mock intentionally does NOT produce a calibrated
        confidence value. It returns confidence=None so that the finalizer
        uses only the classifier probability. It also sets a clear
        `generation_mocked` flag so downstream consumers know the summary
        was not produced by a language model.
        """
        extraction = extract(complaint_text)

        summary_words = complaint_text.split()[:40]
        summary = " ".join(summary_words) + ("..." if len(summary_words) >= 40 else "")

        case_snippets = [
            c.get("narrative_clean", "")[:100] + "..."
            for c in similar_cases[:2]
        ] or ["No similar cases found."]

        urgency_map = {
            "high": "Escalate to senior analyst",
            "medium": "Contact company for response",
            "low": "Request more information from consumer",
        }
        next_step = urgency_map.get(extraction.urgency_level, "Contact company for response")

        return {
            "complaint_summary": summary,
            "predicted_issue": predicted_issue,
            "similar_cases": case_snippets,
            "recommended_next_step": next_step,
            "confidence": None,
            "needs_human_review": None,
        }

    # ------------------------------------------------------------------
    # LLM provider implementations
    # ------------------------------------------------------------------

    def _openai_generate(self, system_prompt: str, user_prompt: str) -> dict:
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai package not installed. Run: pip install openai")

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set.")

        client = OpenAI(api_key=api_key)
        cfg = self._cfg.get("openai", {})
        response = client.chat.completions.create(
            model=cfg.get("model", "gpt-4o-mini"),
            temperature=cfg.get("temperature", 0.2),
            max_tokens=cfg.get("max_tokens", 512),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return self._parse_json_response(response.choices[0].message.content)

    def _anthropic_generate(self, system_prompt: str, user_prompt: str) -> dict:
        try:
            import anthropic
        except ImportError:
            raise ImportError("anthropic package not installed. Run: pip install anthropic")

        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable not set.")

        client = anthropic.Anthropic(api_key=api_key)
        cfg = self._cfg.get("anthropic", {})
        message = client.messages.create(
            model=cfg.get("model", "claude-3-haiku-20240307"),
            max_tokens=cfg.get("max_tokens", 512),
            temperature=cfg.get("temperature", 0.2),
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        return self._parse_json_response(message.content[0].text)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_json_response(text: str) -> dict:
        text = text.strip()
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            logger.warning(f"JSON parse failed: {e}. Raw response:\n{text[:300]}")
            return {}

    def _empty_result(self, reason: str) -> dict:
        return {
            "complaint_summary": f"[empty input: {reason}]",
            "predicted_issue": "unknown",
            "similar_cases": [],
            "recommended_next_step": "Request more information from consumer",
            "confidence": 0.0,
            "needs_human_review": True,
            "generation_mocked": True,
        }

    def _validate_and_finalize(
        self,
        raw_output: dict,
        predicted_issue: str,
        classifier_confidence: float,
        mocked: bool,
        similar_cases: list[dict],
    ) -> dict:
        """
        Normalize raw output into the stable schema.

        Confidence handling:
          - Mock mode: confidence = classifier_confidence (no synthetic blending)
          - Real LLM: confidence = min(classifier_confidence, llm_confidence)
        """
        result: dict = {}

        result["complaint_summary"] = str(
            raw_output.get("complaint_summary", "Summary unavailable.")
        )
        result["predicted_issue"] = str(raw_output.get("predicted_issue", predicted_issue))
        result["similar_cases"] = list(raw_output.get("similar_cases", []))

        next_step = raw_output.get("recommended_next_step", "")
        if next_step not in ALLOWED_NEXT_STEPS:
            next_step = "Request more information from consumer"
        result["recommended_next_step"] = next_step

        if mocked:
            final_conf = classifier_confidence
        else:
            try:
                llm_conf = float(raw_output.get("confidence", 0.5))
            except (TypeError, ValueError):
                llm_conf = 0.5
            final_conf = min(classifier_confidence, llm_conf)

        result["confidence"] = round(float(final_conf), 4)
        result["generation_mocked"] = bool(mocked)

        # Human review logic: low confidence OR LLM explicitly flagged it.
        raw_needs_review = raw_output.get("needs_human_review")
        explicit_flag = bool(raw_needs_review) if raw_needs_review is not None else False
        result["needs_human_review"] = (
            result["confidence"] < self._confidence_threshold or explicit_flag
        )

        return result
