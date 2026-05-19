"""MCQ option / answer parsing (self-contained; mirrors egocross.parsing subset)."""

from __future__ import annotations

import re

VALID_ANSWERS = frozenset({"A", "B", "C", "D"})


def normalize_option(option_text: str) -> str:
    """Normalize option formatting to match support-set style."""
    option_text = option_text.strip()
    if re.match(r"^[A-D]\s*[:.]", option_text):
        return f"{option_text[0]}. {option_text[2:].strip()}"
    return option_text


def parse_letter(raw_answer: str | None) -> tuple[str | None, str]:
    """Return (letter_or_none, status). status: ok | empty | invalid."""
    if raw_answer is None:
        return None, "empty"
    cleaned = str(raw_answer).strip().upper()
    if not cleaned:
        return None, "empty"
    if cleaned in VALID_ANSWERS:
        return cleaned, "ok"
    for char in cleaned:
        if char in VALID_ANSWERS:
            return char, "ok"
    return None, "invalid"


def extract_option_body(option_text: str) -> str:
    """Strip leading A/B/C/D label from an option string."""
    text = normalize_option(option_text)
    if len(text) >= 2 and text[0] in VALID_ANSWERS and text[1] == ".":
        return text[2:].strip()
    return text
