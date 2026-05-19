#!/usr/bin/env python3
"""Shared ATL expert-feature utilities for the adaptive router."""

from __future__ import annotations

import math
import re
from typing import Any


HARD_ACTIONS = {"right", "curveright", "jump", "walk"}

TIME_RE = re.compile(r"([-+]?\d+(?:\.\d+)?)\s*s\b", re.IGNORECASE)


def parse_letter(text: str) -> str | None:
    m = re.match(r"\s*([A-D])\s*[:.]", str(text))
    return m.group(1) if m else None


def parse_time(text: str) -> float | None:
    m = TIME_RE.search(str(text))
    return float(m.group(1)) if m else None


def option_time_map(sample: dict[str, Any], pred_rows: list[dict[str, Any]]) -> dict[str, float]:
    mapping: dict[str, float] = {}
    for opt in sample.get("options", []):
        letter = parse_letter(opt)
        sec = parse_time(opt)
        if letter and sec is not None:
            mapping[letter] = sec
    for row in pred_rows:
        letter = row.get("letter")
        sec = row.get("time_seconds")
        if letter and sec is not None:
            mapping[str(letter)] = float(sec)
        elif letter and "option" in row:
            parsed = parse_time(str(row["option"]))
            if parsed is not None:
                mapping[str(letter)] = parsed
    return mapping


def extract_action(question_text: str) -> str:
    text = question_text.strip()
    patterns = [
        r"\bapproximate\s+time\s+does\s+the\s+['\"]?([A-Za-z -]+?)['\"]?\s+action\s+begin",
        r"\btime\s+does\s+the\s+['\"]?([A-Za-z -]+?)['\"]?\s+action\s+begin",
        r"\baction\s+['\"]?([A-Za-z -]+?)['\"]?\s+begin",
        r"\bwhen\s+does\s+the\s+([A-Za-z -]+?)\s+begin",
        r"\bstart\s+of\s+the\s+([A-Za-z -]+?)\b",
    ]
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            return normalize_action(m.group(1))
    m = re.search(r"\bstart\s+to\s+([A-Za-z -?]+)", text, re.IGNORECASE)
    if m:
        return normalize_action(m.group(1))
    return ""


def normalize_action(action: str) -> str:
    action = action.lower().strip(" .?\"'")
    action = action.replace("curve left", "curveleft")
    action = action.replace("curve right", "curveright")
    action = action.replace("left then right", "leftright")
    action = action.replace(" ", "")
    return action


def softmax(values: list[float]) -> list[float]:
    finite = [v for v in values if math.isfinite(v)]
    if not finite:
        return [1.0 / len(values)] * len(values)
    mx = max(finite)
    exps = [math.exp(v - mx) if math.isfinite(v) else 0.0 for v in values]
    total = sum(exps)
    if total <= 0:
        return [1.0 / len(values)] * len(values)
    return [x / total for x in exps]


def entropy_from_scores(values: list[float]) -> float:
    probs = softmax(values)
    return -sum(p * math.log(p + 1e-12) for p in probs)


def quantile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    vals = sorted(values)
    pos = (len(vals) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return vals[lo]
    frac = pos - lo
    return vals[lo] * (1 - frac) + vals[hi] * frac


def rank_for_letter(letter: str, time_map: dict[str, float]) -> int | None:
    if letter not in time_map:
        return None
    sorted_letters = sorted(time_map, key=lambda x: (time_map[x], x))
    return sorted_letters.index(letter) + 1


def expert_features(
    pred: dict[str, Any],
    sample: dict[str, Any],
    kind: str,
) -> dict[str, Any]:
    if kind == "T":
        rows = pred.get("transition_guided_verification", [])
        scores = [float(r.get("transition_score", float("-inf"))) for r in rows]
    elif kind == "O":
        rows = pred.get("option_guided_verification", [])
        scores = [
            float(r.get("yes_logprob", math.log(max(float(r.get("yes_probability", 0.0)), 1e-12))))
            - float(r.get("no_logprob", math.log(max(1.0 - float(r.get("yes_probability", 0.0)), 1e-12))))
            for r in rows
        ]
    else:
        raise ValueError(kind)

    if not rows:
        return {}

    time_map = option_time_map(sample, rows)
    order = sorted(range(len(rows)), key=lambda i: scores[i], reverse=True)
    top = order[0]
    second = order[1] if len(order) > 1 else order[0]
    answer = str(pred.get("answer") or rows[top].get("letter"))
    pred_time = time_map.get(answer)
    margin = scores[top] - scores[second] if len(order) > 1 else float("inf")
    probs = softmax(scores)

    return {
        "answer": answer,
        "pred_time": pred_time,
        "pred_rank": rank_for_letter(answer, time_map),
        "margin": margin,
        "entropy": entropy_from_scores(scores),
        "top_probability": probs[top],
        "scores": {str(rows[i].get("letter")): scores[i] for i in range(len(rows))},
        "time_map": time_map,
    }
