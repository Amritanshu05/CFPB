"""
Prompt templates for structured resolution output generation.

All prompts are designed to produce a fixed JSON schema:
  {
    "complaint_summary": str,
    "predicted_issue": str,
    "similar_cases": [str, ...],
    "recommended_next_step": str,
    "confidence": float (0.0–1.0),
    "needs_human_review": bool
  }

The prompts are intentionally constrained:
  - They require the model to ground its answer in retrieved cases
  - They forbid fabricating facts not present in the complaint or retrieved cases
  - They instruct the model to set needs_human_review=true when uncertain
"""

from __future__ import annotations

import json

SYSTEM_PROMPT = """\
You are a complaint analysis assistant for a financial regulatory review system.
Your task is to analyze a consumer complaint and produce a structured JSON summary
to help a human analyst understand the complaint and decide next steps.

Rules:
1. Base your analysis ONLY on the complaint text and the retrieved similar cases provided.
2. Do NOT fabricate details, company names, amounts, or dates not present in the text.
3. Keep the complaint_summary under 80 words.
4. The recommended_next_step must be one of:
   - "Request supporting documents"
   - "Escalate to senior analyst"
   - "Contact company for response"
   - "Mark as resolved - similar cases closed"
   - "Refer to legal team"
   - "Request more information from consumer"
5. Set needs_human_review to true if confidence < 0.6 or if the complaint is ambiguous.
6. Output ONLY the JSON object. No preamble, no explanation, no markdown fences.
"""

RESOLUTION_TEMPLATE = """\
COMPLAINT:
{complaint_text}

PREDICTED ISSUE CATEGORY: {predicted_issue}

SIMILAR HISTORICAL CASES:
{similar_cases_text}

OUTPUT JSON SCHEMA:
{{
  "complaint_summary": "<80-word summary of the core complaint>",
  "predicted_issue": "<issue category>",
  "similar_cases": ["<brief description of similar case 1>", "<brief description of similar case 2>"],
  "recommended_next_step": "<one of the allowed next steps>",
  "confidence": <float 0.0 to 1.0>,
  "needs_human_review": <true or false>
}}

Produce the JSON now:
"""

MOCK_RESOLUTION_TEMPLATE = """\
Based on the following complaint, generate a structured resolution summary.

COMPLAINT: {complaint_text}
PREDICTED ISSUE: {predicted_issue}
SIMILAR CASES: {similar_cases_text}
"""


def build_resolution_prompt(
    complaint_text: str,
    predicted_issue: str,
    similar_cases: list[dict],
) -> tuple[str, str]:
    """
    Build the system and user prompts for structured resolution generation.

    Args:
        complaint_text: cleaned complaint narrative
        predicted_issue: predicted product/issue label
        similar_cases: list of dicts, each with at least 'narrative_clean' and 'product_clean'

    Returns:
        (system_prompt, user_prompt) tuple
    """
    similar_cases_text = _format_similar_cases(similar_cases)

    user_prompt = RESOLUTION_TEMPLATE.format(
        complaint_text=complaint_text[:1500],
        predicted_issue=predicted_issue,
        similar_cases_text=similar_cases_text,
    )
    return SYSTEM_PROMPT, user_prompt


def _format_similar_cases(cases: list[dict]) -> str:
    """Format retrieved cases into a compact text block for the prompt."""
    if not cases:
        return "No similar cases retrieved."

    lines = []
    for i, case in enumerate(cases, start=1):
        narrative = case.get("narrative_clean", "")[:200]
        product = case.get("product_clean", "unknown")
        response = case.get("company_response", "")
        response_str = f" | Company response: {response}" if response else ""
        lines.append(f"Case {i} [{product}]: {narrative}{response_str}")

    return "\n".join(lines)
