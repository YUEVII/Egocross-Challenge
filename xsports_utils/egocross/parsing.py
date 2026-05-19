from __future__ import annotations

import re
from typing import Any

from egocross.constants import VALID_ANSWERS


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


def letter_to_index(letter: str) -> int:
    return ord(letter) - ord("A")


def index_to_letter(index: int) -> str:
    return chr(ord("A") + (index % 4))


def option_text_for_letter(options: list[str], letter: str) -> str | None:
    """Return normalized option line matching the given letter, if found."""
    target = letter.upper()
    for opt in options:
        norm = normalize_option(opt)
        if len(norm) >= 2 and norm[0] == target and norm[1] == ".":
            return norm
    return None


def map_shifted_letter_to_original(shifted_letter: str, k: int) -> str:
    """Map model letter under k-step content rotation to original A/B/C/D.

    Slot i (0=A) shows original option index (i - k) mod 4.
    If model picks slot s, original index is (s - k) mod 4.
    """
    s = letter_to_index(shifted_letter)
    orig = (s - k) % 4
    return index_to_letter(orig)


def extract_option_body(option_text: str) -> str:
    """Strip leading A/B/C/D label from an option string."""
    text = normalize_option(option_text)
    if len(text) >= 2 and text[0] in VALID_ANSWERS and text[1] == ".":
        return text[2:].strip()
    return text


def rotate_options_content_right(options: list[str], k: int) -> list[str]:
    """Rotate option *bodies* so label A..D stay fixed (k steps).

    After k rotations, slot i displays the body that originally belonged to
    option index (i - k) mod 4, i.e. new_body[i] = old_body[(i - k) mod 4].
    """
    if len(options) != 4:
        raise ValueError("Expected exactly four options.")
    bodies = [extract_option_body(o) for o in options]
    letters = ["A", "B", "C", "D"]
    rotated: list[str] = []
    for i in range(4):
        src = (i - k) % 4
        rotated.append(f"{letters[i]}. {bodies[src]}")
    return rotated


def parse_judge_output(raw: str) -> tuple[str | None, int | None, str]:
    """Parse YES/NO and CONF: 0-100 from judge text. status: ok | invalid."""
    text = raw.strip().upper()
    yes_no: str | None = None
    if "YES" in text.split()[:3] or text.startswith("YES"):
        yes_no = "YES"
    elif "NO" in text.split()[:3] or text.startswith("NO"):
        yes_no = "NO"
    else:
        first_line = text.split("\n", 1)[0].strip()
        if first_line in {"YES", "NO"}:
            yes_no = first_line

    conf_match = re.search(r"CONF\s*[: ]\s*(\d{1,3})", text)
    conf: int | None = None
    if conf_match:
        conf = int(conf_match.group(1))
        if conf > 100:
            conf = 100

    if yes_no is None:
        return None, conf, "invalid"
    if conf is None:
        return yes_no, None, "invalid"
    return yes_no, conf, "ok"
