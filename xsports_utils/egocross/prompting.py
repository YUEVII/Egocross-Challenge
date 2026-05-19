from __future__ import annotations

import re
from typing import Any

from egocross.constants import STOPWORDS
from egocross.parsing import normalize_option


def infer_task_family(
    question_text: str,
    question_type: str,
    primary_category: str,
) -> str:
    combined_text = " ".join(
        [question_text, question_type, primary_category]
    ).lower()
    if "next" in combined_text or "predicted next" in combined_text:
        return "prediction"
    if "when does" in combined_text or "temporal" in combined_text:
        return "temporal_localization"
    if "timestamp" in combined_text and "region" not in combined_text:
        return "temporal_localization"
    if "region" in combined_text or "located" in combined_text:
        return "spatial_localization"
    if "how many" in combined_text or "count" in combined_text:
        return "counting"
    if "not visible" in combined_text:
        return "negative_identification"
    return "identification"


def build_task_guidance(
    question_text: str,
    question_type: str,
    primary_category: str,
    original_video_fps: float | None,
) -> str:
    task_family = infer_task_family(
        question_text=question_text,
        question_type=question_type,
        primary_category=primary_category,
    )
    guidance_lines = [
        "The frames are ordered from earliest to latest in time.",
        "Use the full sequence before choosing the best answer option.",
    ]
    if original_video_fps:
        guidance_lines.append(
            f"The original video frame rate was {original_video_fps:.1f} FPS."
        )
    if task_family == "prediction":
        guidance_lines.append(
            "For next-step prediction, focus on the latest visible state and "
            "the most likely immediate transition."
        )
    elif task_family == "temporal_localization":
        guidance_lines.append(
            "For temporal localization, align visible events with the provided "
            "timestamps or time windows."
        )
    elif task_family == "spatial_localization":
        guidance_lines.append(
            "For spatial localization, match the object or interaction to the "
            "best region option in the image plane."
        )
    elif task_family == "counting":
        guidance_lines.append(
            "For counting, count distinct object or action types rather than "
            "repeated appearances of the same type."
        )
    elif task_family == "negative_identification":
        guidance_lines.append(
            "For not-visible questions, eliminate options that appear anywhere "
            "in the sequence and select the absent one."
        )
    else:
        guidance_lines.append(
            "Identify the option best supported by the visible evidence."
        )
    return "\n".join(guidance_lines)


def build_mcq_prompt(
    question_text: str,
    options: list[str],
    question_type: str,
    primary_category: str,
    original_video_fps: float | None,
    enforce_single_letter: bool = True,
) -> str:
    prompt_lines = [
        build_task_guidance(
            question_text=question_text,
            question_type=question_type,
            primary_category=primary_category,
            original_video_fps=original_video_fps,
        ),
        "",
        question_text.strip(),
        "",
    ]
    prompt_lines.extend(normalize_option(option) for option in options)
    prompt_lines.append("")
    if enforce_single_letter:
        prompt_lines.append(
            "Answer with only the single letter: A, B, C, or D."
        )
    return "\n".join(prompt_lines)


def build_judge_prompt(
    question_text: str,
    option_line: str,
    question_type: str,
    primary_category: str,
    original_video_fps: float | None,
) -> str:
    guidance = build_task_guidance(
        question_text=question_text,
        question_type=question_type,
        primary_category=primary_category,
        original_video_fps=original_video_fps,
    )
    opt = normalize_option(option_line)
    return "\n".join(
        [
            guidance,
            "",
            "You are evaluating ONE multiple-choice option for a video question.",
            "The option text may be wrong; treat it only as a hypothesis.",
            "",
            "Question:",
            question_text.strip(),
            "",
            "Option to evaluate:",
            opt,
            "",
            "Decide if this option is the correct answer given the video frames.",
            "Reply with exactly two lines and nothing else:",
            "Line 1: YES or NO",
            "Line 2: CONF: <integer from 0 to 100>",
        ]
    )


def tokenize_for_overlap(text: str) -> set[str]:
    tokens = set(re.findall(r"[a-z0-9]+", text.lower()))
    return {
        token
        for token in tokens
        if token not in STOPWORDS and len(token) > 1
    }


def clean_support_prompt(prompt_text: str) -> str:
    return re.sub(r"(?:<image>)+", "", prompt_text).strip()
