"""
Key information extractor for CFPB complaint narratives.

Extracts structured facts from a raw complaint without relying on an LLM.
Uses rule-based patterns and heuristics appropriate for CFPB complaint language.

Extracted fields:
  - product_mention: financial product mentioned in the text
  - urgency_level: 'high', 'medium', or 'low'
  - sentiment: 'negative', 'neutral', or 'positive'
  - key_phrases: list of short phrases capturing the core grievance
  - action_words: financial action verbs found (charged, disputed, denied, etc.)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from cfpb_assistant.utils.logging_utils import get_logger

logger = get_logger(__name__)

# ------------------------------------------------------------------
# Patterns
# ------------------------------------------------------------------

_URGENCY_HIGH = re.compile(
    r"\b(fraud|fraudulent|identity theft|stolen|unauthorized|illegal|threatening|harassing"
    r"|emergency|immediately|urgent|lawsuit|legal action|attorney|sue|court|evict)\b",
    re.IGNORECASE,
)
_URGENCY_MED = re.compile(
    r"\b(dispute|error|incorrect|inaccurate|overcharged|late fee|penalty|denied"
    r"|collections|credit score|damaged|escalate|supervisor|manager)\b",
    re.IGNORECASE,
)

_PRODUCT_PATTERNS = {
    "credit_card": re.compile(
        r"\b(credit card|charge card|prepaid card|visa|mastercard|amex|american express)\b",
        re.IGNORECASE,
    ),
    "mortgage": re.compile(
        r"\b(mortgage|home loan|foreclosure|deed of trust|escrow|refinanc)\b",
        re.IGNORECASE,
    ),
    "student_loan": re.compile(
        r"\b(student loan|federal loan|sallie mae|navient|great lakes|loan forgiveness)\b",
        re.IGNORECASE,
    ),
    "debt_collection": re.compile(
        r"\b(debt collector|collection agency|collections|third.?party|garnish)\b",
        re.IGNORECASE,
    ),
    "credit_reporting": re.compile(
        r"\b(credit report|credit bureau|experian|equifax|transunion|credit score|fico)\b",
        re.IGNORECASE,
    ),
    "bank_account": re.compile(
        r"\b(checking account|savings account|bank account|overdraft|direct deposit|wire transfer)\b",
        re.IGNORECASE,
    ),
    "vehicle_loan": re.compile(
        r"\b(auto loan|car loan|vehicle loan|repossess|repo)\b",
        re.IGNORECASE,
    ),
    "personal_loan": re.compile(
        r"\b(personal loan|payday loan|title loan|installment loan)\b",
        re.IGNORECASE,
    ),
}

_ACTION_VERBS = re.compile(
    r"\b(charged|billed|disputed|denied|refused|closed|opened|transferred|withdrew"
    r"|deposited|reported|sued|threatened|harassed|called|contacted|sent|received"
    r"|applied|rejected|approved|cancelled|reversed|refunded|garnished|foreclosed)\b",
    re.IGNORECASE,
)

_NEGATIVE_WORDS = re.compile(
    r"\b(wrong|incorrect|error|mistake|fraud|unauthorized|unfair|illegal|refused"
    r"|denied|never|not|no|problem|issue|complaint|dispute|angry|frustrated|upset"
    r"|terrible|awful|horrible|unacceptable|ridiculous|outrageous)\b",
    re.IGNORECASE,
)


# ------------------------------------------------------------------
# Data class
# ------------------------------------------------------------------

@dataclass
class ExtractionResult:
    product_mention: str = "unknown"
    urgency_level: str = "low"
    sentiment: str = "negative"
    action_words: list[str] = field(default_factory=list)
    key_phrases: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "product_mention": self.product_mention,
            "urgency_level": self.urgency_level,
            "sentiment": self.sentiment,
            "action_words": self.action_words,
            "key_phrases": self.key_phrases,
        }


# ------------------------------------------------------------------
# Extraction function
# ------------------------------------------------------------------

def extract(text: str) -> ExtractionResult:
    """
    Extract structured information from a complaint narrative.

    Args:
        text: cleaned complaint narrative (lowercase preferred)

    Returns:
        ExtractionResult dataclass
    """
    result = ExtractionResult()

    # Product mention
    for product_name, pattern in _PRODUCT_PATTERNS.items():
        if pattern.search(text):
            result.product_mention = product_name
            break

    # Urgency
    if _URGENCY_HIGH.search(text):
        result.urgency_level = "high"
    elif _URGENCY_MED.search(text):
        result.urgency_level = "medium"
    else:
        result.urgency_level = "low"

    # Sentiment (simple negative-word count heuristic)
    neg_count = len(_NEGATIVE_WORDS.findall(text))
    if neg_count >= 3:
        result.sentiment = "negative"
    elif neg_count >= 1:
        result.sentiment = "mixed"
    else:
        result.sentiment = "neutral"

    # Action words
    action_matches = _ACTION_VERBS.findall(text)
    result.action_words = list(dict.fromkeys(w.lower() for w in action_matches))[:8]

    # Key phrases: sentences containing the most action/negative words
    sentences = re.split(r"[.!?]+", text)
    scored = []
    for sent in sentences:
        sent = sent.strip()
        if len(sent) < 20:
            continue
        score = (
            len(_ACTION_VERBS.findall(sent)) * 2
            + len(_NEGATIVE_WORDS.findall(sent))
        )
        scored.append((score, sent))
    scored.sort(key=lambda x: x[0], reverse=True)
    result.key_phrases = [s for _, s in scored[:3] if s]

    return result
