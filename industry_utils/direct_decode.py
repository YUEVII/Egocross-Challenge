"""Direct option-guided yes/no decoding for Industry / ENIGMA questions."""

from __future__ import annotations

import itertools
import math
import re
import statistics
from pathlib import Path
from typing import Any
from typing import Callable

from industry_infer.parsing import extract_option_body
from industry_infer.parsing import normalize_option
from industry_infer.parsing import parse_letter
from industry_infer.sampling import chunk_frames

_LETTERS = ("A", "B", "C", "D")
_NEXT_INTERACTION_PREFIX_RE = re.compile(
    r"^\s*Based on the activity observed up to\s*[\d.]+\s*s\s*,?\s*",
    re.IGNORECASE,
)
_REFINED_CATEGORY_HINT_RE = re.compile(
    r"\(\s*using\s+refined\s+categories\s+like\s+'[^']+'\s+for\s+all\s+[^)]+\)",
    re.IGNORECASE,
)
_POINT_WITH_LABEL_RE = re.compile(
    r"x\s*[:=]\s*(\d{1,4})\s*[,;]\s*y\s*[:=]\s*(\d{1,4})",
    re.IGNORECASE,
)
_POINT_PLAIN_RE = re.compile(
    r"\(?\s*(\d{1,4})\s*[,;]\s*(\d{1,4})\s*\)?",
    re.IGNORECASE,
)
_COUNTING_BBOX_LINE_RE = re.compile(
    r"['\"]?(?P<label>[A-Za-z][A-Za-z0-9 /_-]{0,80}?)['\"]?\s*:\s*\[\s*"
    r"(?P<x1>\d{1,4})\s*[,;]\s*(?P<y1>\d{1,4})\s*[,;]\s*"
    r"(?P<x2>\d{1,4})\s*[,;]\s*(?P<y2>\d{1,4})\s*\]"
    r"(?:\s*\(\s*confidence\s*[:=]\s*(?P<confidence>\d+(?:\.\d+)?)\s*\))?",
    re.IGNORECASE,
)
_COUNTING_BBOX_JSON_RE = re.compile(
    r"(?:label|name)\s*[:=]\s*['\"]?(?P<label>[A-Za-z][A-Za-z0-9 /_-]{0,80}?)['\"]?"
    r"[\s,;:{}]*"
    r"(?:bbox|box)\s*[:=]\s*\[\s*(?P<x1>\d{1,4})\s*[,;]\s*(?P<y1>\d{1,4})\s*[,;]\s*"
    r"(?P<x2>\d{1,4})\s*[,;]\s*(?P<y2>\d{1,4})\s*\]"
    r"[\s,;:{}]*"
    r"(?:(?:confidence|score)\s*[:=]\s*(?P<confidence>\d+(?:\.\d+)?))?",
    re.IGNORECASE,
)
_GRID_AXIS_TO_PERMILLE = {
    "left": 167,
    "center": 500,
    "right": 833,
    "top": 167,
    "bottom": 833,
}


def build_yes_no_token_ids(processor: Any) -> tuple[list[int], list[int]]:
    """Build candidate token ids for yes/no next-token scoring."""

    yes_token_ids = _candidate_token_ids(processor, ("yes", "YES"))
    no_token_ids = _candidate_token_ids(processor, ("no", "NO"))
    return yes_token_ids, no_token_ids


def build_ab_token_ids(processor: Any) -> tuple[list[int], list[int]]:
    """Build candidate token ids for letter-choice pairwise scoring."""

    a_token_ids = _candidate_token_ids(processor, ("A",))
    b_token_ids = _candidate_token_ids(processor, ("B",))
    return a_token_ids, b_token_ids


def build_mcq_token_ids(processor: Any) -> dict[str, list[int]]:
    """Build candidate token ids for A/B/C/D next-token scoring."""

    return {
        letter: _candidate_token_ids(processor, (letter, letter.lower()))
        for letter in _LETTERS
    }


def run_option_guided_verification(
    model: Any,
    processor: Any,
    image_paths: list[str],
    question: dict[str, Any],
    yes_token_ids: list[int],
    no_token_ids: list[int],
    frame_prefix_texts: list[str] | None = None,
) -> dict[str, Any]:
    """Score every answer option with a yes/no visual verification prompt."""

    options = question.get("options", [])
    if len(options) != 4:
        raise ValueError(f"Expected 4 options, got {len(options)}.")

    option_scores = []
    for option_index, option_text in enumerate(options):
        score = _verify_option(
            model=model,
            processor=processor,
            image_paths=image_paths,
            question=question,
            option_text=str(option_text),
            yes_token_ids=yes_token_ids,
            no_token_ids=no_token_ids,
            verify_prompt_builder=_build_verify_prompt,
            frame_prefix_texts=frame_prefix_texts,
        )
        score["letter"] = _LETTERS[option_index]
        option_scores.append(score)

    best_score = max(option_scores, key=lambda item: item["yes_probability"])
    return {
        "answer": best_score["letter"],
        "raw_answer": (
            f"{best_score['letter']} "
            f"(yes_probability={best_score['yes_probability']:.6f})"
        ),
        "decode_method": "option_guided_verification_max_yes",
        "option_guided_verification": option_scores,
    }


def run_next_interaction_logprob(
    model: Any,
    processor: Any,
    image_paths: list[str],
    question: dict[str, Any],
    yes_token_ids: list[int],
    no_token_ids: list[int],
    frame_prefix_texts: list[str] | None = None,
) -> dict[str, Any]:
    """Score each candidate next interaction with yes/no next-token logprobs."""

    options = question.get("options", [])
    if len(options) != 4:
        raise ValueError(f"Expected 4 options, got {len(options)}.")
    if not image_paths:
        raise ValueError("Expected at least one effective frame.")
    if frame_prefix_texts is not None and len(frame_prefix_texts) != len(image_paths):
        raise ValueError("frame_prefix_texts must match image_paths length.")

    option_scores = []
    for option_index, option_text in enumerate(options):
        score = _verify_option(
            model=model,
            processor=processor,
            image_paths=image_paths,
            question=question,
            option_text=str(option_text),
            yes_token_ids=yes_token_ids,
            no_token_ids=no_token_ids,
            verify_prompt_builder=_build_next_interaction_prompt,
            frame_prefix_texts=frame_prefix_texts,
        )
        score["letter"] = _LETTERS[option_index]
        option_scores.append(score)

    best_score = max(option_scores, key=lambda item: item["yes_probability"])
    return {
        "answer": best_score["letter"],
        "raw_answer": (
            f"{best_score['letter']} "
            f"(yes_probability={best_score['yes_probability']:.6f})"
        ),
        "decode_method": "next_interaction_option_guided_max_yes",
        "option_guided_verification": option_scores,
    }


def run_next_interaction_pairwise(
    model: Any,
    processor: Any,
    image_paths: list[str],
    question: dict[str, Any],
    choice_a_token_ids: list[int],
    choice_b_token_ids: list[int],
    frame_prefix_texts: list[str] | None = None,
) -> dict[str, Any]:
    """Compare every option pair and aggregate symmetric pairwise margins."""

    options = question.get("options", [])
    if len(options) != 4:
        raise ValueError(f"Expected 4 options, got {len(options)}.")
    if not image_paths:
        raise ValueError("Expected at least one effective frame.")
    if frame_prefix_texts is not None and len(frame_prefix_texts) != len(image_paths):
        raise ValueError("frame_prefix_texts must match image_paths length.")

    option_bodies = [extract_option_body(str(option_text)) for option_text in options]
    pairwise_scores = [0.0, 0.0, 0.0, 0.0]
    pairwise_comparisons = []
    for option_i, option_j in itertools.combinations(range(4), 2):
        detail = _score_next_interaction_pair(
            model=model,
            processor=processor,
            image_paths=image_paths,
            question=question,
            option_bodies=option_bodies,
            option_i=option_i,
            option_j=option_j,
            choice_a_token_ids=choice_a_token_ids,
            choice_b_token_ids=choice_b_token_ids,
            frame_prefix_texts=frame_prefix_texts,
        )
        margin = float(detail["margin_i_over_j"])
        pairwise_scores[option_i] += margin
        pairwise_scores[option_j] -= margin
        pairwise_comparisons.append(detail)

    best_index = max(range(4), key=lambda idx: pairwise_scores[idx])
    return {
        "answer": _LETTERS[best_index],
        "raw_answer": (
            f"{_LETTERS[best_index]} "
            f"(pairwise_score={pairwise_scores[best_index]:.6f})"
        ),
        "decode_method": "next_interaction_pairwise_margin_sum",
        "pairwise_verification": {
            "option_bodies": {
                _LETTERS[idx]: option_bodies[idx] for idx in range(len(option_bodies))
            },
            "scores": {
                _LETTERS[idx]: pairwise_scores[idx] for idx in range(len(pairwise_scores))
            },
            "pairs": pairwise_comparisons,
        },
    }


def run_next_interaction_video_mcq(
    model: Any,
    processor: Any,
    image_paths: list[str],
    question: dict[str, Any],
    max_new_tokens: int,
    sample_fps: float = 1.0,
    video_max_pixels: int | None = None,
) -> dict[str, Any]:
    """Package tail frames as one video input and decode a direct MCQ answer."""

    options = question.get("options", [])
    if len(options) != 4:
        raise ValueError(f"Expected 4 options, got {len(options)}.")
    if not image_paths:
        raise ValueError("Expected at least one effective frame.")

    prompt = _build_next_interaction_video_mcq_prompt(question)
    raw_answer = _generate_video_text_response(
        model=model,
        processor=processor,
        image_paths=image_paths,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        sample_fps=sample_fps,
        video_max_pixels=video_max_pixels,
    )
    answer, parse_status = parse_letter(raw_answer)
    return {
        "answer": answer or "",
        "raw_answer": raw_answer,
        "parse_status": parse_status,
        "decode_method": "next_interaction_tail_video_mcq",
        "prompt_style": "next_interaction_video_direct_mcq",
        "input_media_type": "video",
        "sample_fps": sample_fps,
        "video_max_pixels": video_max_pixels,
    }


def run_object_counting_logprob(
    model: Any,
    processor: Any,
    image_paths: list[str],
    question: dict[str, Any],
    yes_token_ids: list[int],
    no_token_ids: list[int],
    frame_prefix_texts: list[str] | None = None,
) -> dict[str, Any]:
    """Score each counting option with yes/no next-token logprobs."""

    options = question.get("options", [])
    if len(options) != 4:
        raise ValueError(f"Expected 4 options, got {len(options)}.")
    if not image_paths:
        raise ValueError("Expected at least one effective frame.")
    if frame_prefix_texts is not None and len(frame_prefix_texts) != len(image_paths):
        raise ValueError("frame_prefix_texts must match image_paths length.")

    option_scores = []
    for option_index, option_text in enumerate(options):
        score = _verify_option(
            model=model,
            processor=processor,
            image_paths=image_paths,
            question=question,
            option_text=str(option_text),
            yes_token_ids=yes_token_ids,
            no_token_ids=no_token_ids,
            verify_prompt_builder=_build_object_counting_prompt,
            frame_prefix_texts=frame_prefix_texts,
        )
        score["letter"] = _LETTERS[option_index]
        option_scores.append(score)

    best_score = max(option_scores, key=lambda item: item["yes_probability"])
    return {
        "answer": best_score["letter"],
        "raw_answer": (
            f"{best_score['letter']} "
            f"(yes_probability={best_score['yes_probability']:.6f})"
        ),
        "decode_method": "object_counting_option_guided_max_yes",
        "option_guided_verification": option_scores,
    }


def run_object_counting_labeled_bbox_regression(
    model: Any,
    processor: Any,
    image_paths: list[str],
    question: dict[str, Any],
    candidate_labels: list[str],
    max_new_tokens: int,
    frame_prefix_texts: list[str] | None = None,
) -> dict[str, Any]:
    """Ask once over all frames for labeled boxes, then count unique labels."""

    options = question.get("options", [])
    if len(options) != 4:
        raise ValueError(f"Expected 4 options, got {len(options)}.")
    if not image_paths:
        raise ValueError("Expected at least one effective frame.")
    if frame_prefix_texts is not None and len(frame_prefix_texts) != len(image_paths):
        raise ValueError("frame_prefix_texts must match image_paths length.")
    resolved_labels = [str(label).strip() for label in candidate_labels if str(label).strip()]
    if not resolved_labels:
        raise ValueError("candidate_labels must contain at least one object label.")

    prompt = _build_object_counting_bbox_prompt(question, resolved_labels)
    raw_answer = _generate_text_response(
        model=model,
        processor=processor,
        image_paths=image_paths,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        frame_prefix_texts=frame_prefix_texts,
    )
    parsed = _parse_counting_labeled_bboxes(raw_answer, resolved_labels)
    option_counts = _numeric_option_counts(options)
    if not option_counts:
        raise ValueError("Object counting options must contain numeric counts.")
    if parsed["status"] != "ok":
        return {
            "answer": "",
            "raw_answer": raw_answer,
            "parse_status": parsed["status"],
            "decode_method": "object_counting_labeled_bbox_unique_labels",
            "prompt_style": "counting_labeled_bboxes_global",
            "counting_bbox_prediction": {
                "candidate_labels": resolved_labels,
                "recognized_boxes": [],
                "ignored_boxes": [],
                "unique_labels": [],
                "unique_label_count": 0,
                "count_decision": None,
                "parse_error": parsed.get("error", ""),
                "parse_warning": parsed.get("warning", ""),
            },
        }

    count_decision = _choose_count_option_from_value(
        parsed["unique_label_count"],
        option_counts,
    )
    return {
        "answer": count_decision["answer_letter"],
        "raw_answer": raw_answer,
        "parse_status": "ok",
        "decode_method": "object_counting_labeled_bbox_unique_labels",
        "prompt_style": "counting_labeled_bboxes_global",
        "counting_bbox_prediction": {
            "candidate_labels": resolved_labels,
            "recognized_boxes": parsed["recognized_boxes"],
            "ignored_boxes": parsed["ignored_boxes"],
            "unique_labels": parsed["unique_labels"],
            "unique_label_count": parsed["unique_label_count"],
            "count_decision": count_decision,
            "parse_error": "",
            "parse_warning": parsed.get("warning", ""),
        },
    }


def run_not_visible_any_frame_logprob(
    model: Any,
    processor: Any,
    image_paths: list[str],
    question: dict[str, Any],
    yes_token_ids: list[int],
    no_token_ids: list[int],
    frame_prefix_texts: list[str] | None = None,
) -> dict[str, Any]:
    """Pick the object with lowest P(appears in any frame)."""

    options = question.get("options", [])
    if len(options) != 4:
        raise ValueError(f"Expected 4 options, got {len(options)}.")
    if not image_paths:
        raise ValueError("Expected at least one effective frame.")
    if frame_prefix_texts is not None and len(frame_prefix_texts) != len(image_paths):
        raise ValueError("frame_prefix_texts must match image_paths length.")

    option_scores = []
    for option_index, option_text in enumerate(options):
        score = _verify_option(
            model=model,
            processor=processor,
            image_paths=image_paths,
            question=question,
            option_text=str(option_text),
            yes_token_ids=yes_token_ids,
            no_token_ids=no_token_ids,
            verify_prompt_builder=_build_any_frame_presence_prompt,
            frame_prefix_texts=frame_prefix_texts,
        )
        score["letter"] = _LETTERS[option_index]
        score["present_probability"] = score["yes_probability"]
        option_scores.append(score)

    best_score = min(option_scores, key=lambda item: item["present_probability"])
    return {
        "answer": best_score["letter"],
        "raw_answer": (
            f"{best_score['letter']} "
            f"(present_probability={best_score['present_probability']:.6f})"
        ),
        "decode_method": "not_visible_any_frame_option_guided_min_yes",
        "option_guided_verification": option_scores,
        "prompt_style": "any_frame_presence_option_guided",
    }


def run_not_visible_pairwise(
    model: Any,
    processor: Any,
    image_paths: list[str],
    question: dict[str, Any],
    choice_a_token_ids: list[int],
    choice_b_token_ids: list[int],
    frame_prefix_texts: list[str] | None = None,
) -> dict[str, Any]:
    """Compare every option pair and pick the least clearly visible object."""

    options = question.get("options", [])
    if len(options) != 4:
        raise ValueError(f"Expected 4 options, got {len(options)}.")
    if not image_paths:
        raise ValueError("Expected at least one effective frame.")
    if frame_prefix_texts is not None and len(frame_prefix_texts) != len(image_paths):
        raise ValueError("frame_prefix_texts must match image_paths length.")

    option_bodies = [extract_option_body(str(option_text)) for option_text in options]
    pairwise_scores = [0.0, 0.0, 0.0, 0.0]
    pairwise_comparisons = []
    for option_i, option_j in itertools.combinations(range(4), 2):
        detail = _score_not_visible_pair(
            model=model,
            processor=processor,
            image_paths=image_paths,
            question=question,
            option_bodies=option_bodies,
            option_i=option_i,
            option_j=option_j,
            choice_a_token_ids=choice_a_token_ids,
            choice_b_token_ids=choice_b_token_ids,
            frame_prefix_texts=frame_prefix_texts,
        )
        margin = float(detail["margin_i_over_j"])
        pairwise_scores[option_i] += margin
        pairwise_scores[option_j] -= margin
        pairwise_comparisons.append(detail)

    best_index = max(range(4), key=lambda idx: pairwise_scores[idx])
    return {
        "answer": _LETTERS[best_index],
        "raw_answer": (
            f"{_LETTERS[best_index]} "
            f"(pairwise_score={pairwise_scores[best_index]:.6f})"
        ),
        "decode_method": "not_visible_pairwise_margin_sum",
        "pairwise_verification": {
            "option_bodies": {
                _LETTERS[idx]: option_bodies[idx] for idx in range(len(option_bodies))
            },
            "scores": {
                _LETTERS[idx]: pairwise_scores[idx] for idx in range(len(pairwise_scores))
            },
            "pairs": pairwise_comparisons,
        },
        "prompt_style": "not_visible_pairwise_relative_visibility",
    }


def run_not_visible_mcq_logprob(
    model: Any,
    processor: Any,
    image_paths: list[str],
    question: dict[str, Any],
    mcq_token_ids: dict[str, list[int]],
    frame_prefix_texts: list[str] | None = None,
) -> dict[str, Any]:
    """Ask directly which option is not visible, then compare A/B/C/D logprobs."""

    options = question.get("options", [])
    if len(options) != 4:
        raise ValueError(f"Expected 4 options, got {len(options)}.")
    if not image_paths:
        raise ValueError("Expected at least one effective frame.")
    if frame_prefix_texts is not None and len(frame_prefix_texts) != len(image_paths):
        raise ValueError("frame_prefix_texts must match image_paths length.")
    for letter in _LETTERS:
        if not mcq_token_ids.get(letter):
            raise ValueError(f"Tokenizer lacks token candidates for {letter}.")

    prompt = _build_not_visible_mcq_prompt(question)
    letter_logprobs = _next_mcq_option_logprobs(
        model=model,
        processor=processor,
        image_paths=image_paths,
        prompt=prompt,
        mcq_token_ids=mcq_token_ids,
        frame_prefix_texts=frame_prefix_texts,
    )
    best_letter = max(_LETTERS, key=lambda letter: letter_logprobs[letter])
    return {
        "answer": best_letter,
        "raw_answer": (
            f"{best_letter} "
            f"(choice_logprob={letter_logprobs[best_letter]:.6f})"
        ),
        "decode_method": "not_visible_mcq_letter_logprob",
        "mcq_option_logprobs": letter_logprobs,
        "prompt_style": "direct_not_visible_mcq",
    }


def run_visibility_not_visible_logprob(
    model: Any,
    processor: Any,
    image_paths: list[str],
    question: dict[str, Any],
    yes_token_ids: list[int],
    no_token_ids: list[int],
    frame_prefix_texts: list[str] | None = None,
    max_frames_per_call: int = 10,
) -> dict[str, Any]:
    """Pick the option with lowest chunk-max P(yes | visible)."""

    options = question.get("options", [])
    if len(options) != 4:
        raise ValueError(f"Expected 4 options, got {len(options)}.")
    if not image_paths:
        raise ValueError("Expected at least one effective frame.")
    if frame_prefix_texts is not None and len(frame_prefix_texts) != len(image_paths):
        raise ValueError("frame_prefix_texts must match image_paths length.")

    frame_chunks, chunk_meta = chunk_frames(image_paths, max_frames_per_call)
    option_scores = []
    for option_index, option_text in enumerate(options):
        score = _verify_option_chunks(
            model=model,
            processor=processor,
            question=question,
            option_text=str(option_text),
            yes_token_ids=yes_token_ids,
            no_token_ids=no_token_ids,
            verify_prompt_builder=_build_visibility_prompt,
            frame_prefix_texts=frame_prefix_texts,
            frame_chunks=frame_chunks,
        )
        score["letter"] = _LETTERS[option_index]
        option_scores.append(score)

    best_score = min(option_scores, key=lambda item: item["visible_probability"])
    return {
        "answer": best_score["letter"],
        "raw_answer": (
            f"{best_score['letter']} "
            f"(visible_probability={best_score['visible_probability']:.6f}; "
            "chunk_max_then_option_min)"
        ),
        "decode_method": "visibility_yes_logprob_chunk_max_option_min",
        "option_guided_verification": option_scores,
        "frame_chunks": chunk_meta,
    }


def run_dominant_held_object_logprob(
    model: Any,
    processor: Any,
    image_paths: list[str],
    question: dict[str, Any],
    yes_token_ids: list[int],
    no_token_ids: list[int],
    frame_prefix_texts: list[str] | None = None,
    max_frames_per_call: int = 10,
) -> dict[str, Any]:
    """Pick the object with strongest cross-chunk evidence for dominant interaction."""

    options = question.get("options", [])
    if len(options) != 4:
        raise ValueError(f"Expected 4 options, got {len(options)}.")
    if not image_paths:
        raise ValueError("Expected at least one effective frame.")
    if frame_prefix_texts is not None and len(frame_prefix_texts) != len(image_paths):
        raise ValueError("frame_prefix_texts must match image_paths length.")

    frame_chunks, chunk_meta = chunk_frames(image_paths, max_frames_per_call)
    option_scores = []
    for option_index, option_text in enumerate(options):
        score = _verify_option_chunks(
            model=model,
            processor=processor,
            question=question,
            option_text=str(option_text),
            yes_token_ids=yes_token_ids,
            no_token_ids=no_token_ids,
            verify_prompt_builder=_build_dominant_held_object_prompt,
            frame_prefix_texts=frame_prefix_texts,
            frame_chunks=frame_chunks,
            aggregate_mode="frame_weighted_mean_margin",
        )
        score["letter"] = _LETTERS[option_index]
        option_scores.append(score)

    best_score = max(option_scores, key=lambda item: item["predominant_score"])
    return {
        "answer": best_score["letter"],
        "raw_answer": (
            f"{best_score['letter']} "
            f"(predominant_score={best_score['predominant_score']:.6f}; "
            f"mean_yes_probability={best_score['mean_yes_probability']:.6f}; "
            "frame_weighted_mean_margin)"
        ),
        "decode_method": "dominant_held_object_frame_weighted_mean_margin",
        "option_guided_verification": option_scores,
        "frame_chunks": chunk_meta,
    }


def run_spatial_point_regression(
    model: Any,
    processor: Any,
    image_paths: list[str],
    question: dict[str, Any],
    max_new_tokens: int,
    point_output_count: int = 1,
    frame_prefix_texts: list[str] | None = None,
) -> dict[str, Any]:
    """Generate one or more permille points, then map the ensemble to a region."""

    options = question.get("options", [])
    if len(options) != 4:
        raise ValueError(f"Expected 4 options, got {len(options)}.")
    if not image_paths:
        raise ValueError("Expected at least one effective frame.")
    if frame_prefix_texts is not None and len(frame_prefix_texts) != len(image_paths):
        raise ValueError("frame_prefix_texts must match image_paths length.")
    point_output_count = max(1, int(point_output_count))

    prompt = _build_spatial_point_prompt(question, point_output_count=point_output_count)
    raw_answer = _generate_text_response(
        model=model,
        processor=processor,
        image_paths=image_paths,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        frame_prefix_texts=frame_prefix_texts,
    )
    parsed_point = _parse_permille_points(raw_answer, expected_count=point_output_count)
    option_regions = _resolve_option_region_centers(options)
    if parsed_point["status"] != "ok":
        return {
            "answer": "",
            "raw_answer": raw_answer,
            "parse_status": parsed_point["status"],
            "decode_method": "spatial_point_regression_nearest_region",
            "prompt_style": (
                "single_point_permille"
                if point_output_count == 1
                else "point_list_permille"
            ),
            "spatial_point_prediction": {
                "requested_point_count": point_output_count,
                "raw_points": [],
                "aggregated_point": None,
                "predicted_point": None,
                "parse_error": parsed_point.get("error", ""),
                "parse_warning": parsed_point.get("warning", ""),
                "option_centers": option_regions,
                "distances": [],
            },
        }

    raw_points = [
        {"x": int(point["x"]), "y": int(point["y"])} for point in parsed_point["points"]
    ]
    pred_x = int(statistics.median(point["x"] for point in raw_points))
    pred_y = int(statistics.median(point["y"] for point in raw_points))
    distances = []
    for option_region in option_regions:
        distance_sq = (
            (pred_x - int(option_region["center_x"])) ** 2
            + (pred_y - int(option_region["center_y"])) ** 2
        )
        distances.append(
            {
                "letter": option_region["letter"],
                "option": option_region["option"],
                "canonical_region": option_region["canonical_region"],
                "center_x": option_region["center_x"],
                "center_y": option_region["center_y"],
                "distance_sq": distance_sq,
            }
        )
    best = min(distances, key=lambda item: (item["distance_sq"], item["letter"]))
    return {
        "answer": str(best["letter"]),
        "raw_answer": raw_answer,
        "parse_status": "ok",
        "decode_method": "spatial_point_regression_nearest_region",
        "prompt_style": (
            "single_point_permille" if point_output_count == 1 else "point_list_permille"
        ),
        "spatial_point_prediction": {
            "requested_point_count": point_output_count,
            "raw_points": raw_points,
            "aggregated_point": {"x": pred_x, "y": pred_y},
            "predicted_point": {"x": pred_x, "y": pred_y},
            "parse_error": "",
            "parse_warning": parsed_point.get("warning", ""),
            "option_centers": option_regions,
            "distances": distances,
        },
    }


def _build_verify_prompt(
    question: dict[str, Any],
    option_text: str,
) -> str:
    option_body = extract_option_body(option_text)
    return "\n".join(
        [
            "You are verifying one answer option for an egocentric video.",
            "Use only visible evidence in the video frames.",
            "Answer with exactly one word: yes or no.",
            "",
            "Question:",
            str(question.get("question_text", "")).strip(),
            "",
            "Candidate option:",
            option_body,
            "",
            (
                "Does the video visually support this candidate option as "
                "the best answer to the question?"
            ),
            "Answer:",
        ]
    )


def _build_visibility_prompt(
    question: dict[str, Any],
    option_text: str,
) -> str:
    """Yes/no: is the option object clearly visible."""

    option_body = extract_option_body(option_text)
    question_text = _normalize_visibility_question_text(
        str(question.get("question_text", "")).strip(),
    )
    button_hint = _visibility_button_hint(question)
    return "\n".join(
        [
            "You are answering about an egocentric industrial task video.",
            "Use only visual evidence. Answer with exactly one word: yes or no.",
            "Use the specific object categories defined by the answer choices, not broader umbrella categories.",
            button_hint,
            "",
            "Task context:",
            question_text,
            "",
            f"Candidate object: {option_body}",
            "",
            "Is this object clearly visible in this video segment?",
            "Answer:",
        ]
    )


def _build_dominant_held_object_prompt(
    question: dict[str, Any],
    option_text: str,
) -> str:
    option_body = extract_option_body(option_text)
    hand_context = _dominant_hand_context(str(question.get("question_text", "")))
    interaction_target = hand_context or "the operator hand described in the question"
    return "\n".join(
        [
            "You are answering about an egocentric industrial task video.",
            "Use only visual evidence. Answer with exactly one word: yes or no.",
            "",
            "Task context:",
            str(question.get("question_text", "")).strip(),
            "",
            f"Candidate object: {option_body}",
            f"Target hand: {interaction_target}",
            "",
            (
                f"Across the provided video segment, is {option_body} the object that "
                f"{interaction_target} predominantly interacts with?"
            ),
            "Answer:",
        ]
    )


def _build_next_interaction_prompt(
    question: dict[str, Any],
    option_text: str,
) -> str:
    option_body = extract_option_body(option_text)
    question_text = _normalize_next_interaction_question(
        str(question.get("question_text", "")).strip(),
    )
    return "\n".join(
        [
            "You are verifying one candidate next interaction for an egocentric industrial task video.",
            "The provided frames are the final frames of the observed clip and are in chronological order.",
            "Use only visible evidence in these frames.",
            "Answer with exactly one word: yes or no.",
            "",
            "Question:",
            question_text,
            "",
            "Candidate option:",
            option_body,
            "",
            "Is this the next interaction immediately after the observed clip ends?",
            "Answer:",
        ]
    )


def _normalize_next_interaction_question(question_text: str) -> str:
    """Drop redundant observed-up-to wording but keep the task-specific target."""

    normalized = _NEXT_INTERACTION_PREFIX_RE.sub("", question_text).strip()
    if not normalized:
        return question_text.strip()
    if normalized[:1].islower():
        return normalized[:1].upper() + normalized[1:]
    return normalized


def _normalize_visibility_question_text(question_text: str) -> str:
    """Drop the dataset-specific refined-category aside from Q5 stems."""

    normalized = _REFINED_CATEGORY_HINT_RE.sub("", question_text).strip()
    normalized = re.sub(r"\s{2,}", " ", normalized)
    normalized = re.sub(r"\s+([?.!,;:])", r"\1", normalized)
    return normalized or question_text.strip()


def _visibility_button_hint(question: dict[str, Any]) -> str:
    options = question.get("options", [])
    for option in options:
        option_body = extract_option_body(str(option)).strip().lower()
        if "button" in option_body:
            return "Here, button refers to a button on the electrical box or equipment panel."
    return ""


def _build_next_interaction_pair_prompt(
    question: dict[str, Any],
    candidate_a: str,
    candidate_b: str,
) -> str:
    question_text = _normalize_next_interaction_question(
        str(question.get("question_text", "")).strip(),
    )
    return "\n".join(
        [
            "You are comparing two candidate next interactions for an egocentric industrial task video.",
            "The provided frames are the final frames of the observed clip and are in chronological order.",
            "Use only visible evidence in these frames.",
            "Choose the candidate that is more likely to happen immediately after the clip ends.",
            "Answer with exactly one letter: A or B.",
            "",
            "Question:",
            question_text,
            "",
            f"Candidate A: {candidate_a}",
            f"Candidate B: {candidate_b}",
            "",
            "Which candidate is better supported as the next interaction?",
            "Answer:",
        ]
    )


def _build_next_interaction_video_mcq_prompt(question: dict[str, Any]) -> str:
    options = question.get("options", [])
    if len(options) != 4:
        raise ValueError(f"Expected 4 options, got {len(options)}.")
    question_text = _normalize_next_interaction_question(
        str(question.get("question_text", "")).strip(),
    )
    option_block = "\n".join(normalize_option(str(option)) for option in options)
    return "\n".join(
        [
            "You are answering a next-interaction multiple-choice question about an egocentric industrial task video.",
            "The provided input is one short video formed from chronologically ordered tail frames of the observed clip.",
            "Use only visible evidence from the video.",
            "Choose the interaction that is most likely to happen immediately after the observed clip ends.",
            "",
            "Question:",
            question_text,
            "",
            "Options:",
            option_block,
            "",
            "Answer with only the single letter: A, B, C, or D.",
        ]
    )


def _build_object_counting_prompt(
    question: dict[str, Any],
    option_text: str,
) -> str:
    option_body = extract_option_body(option_text)
    return "\n".join(
        [
            "You are verifying one candidate answer for an object-counting question in an egocentric industrial task video.",
            "The provided images are in chronological order and may show the same objects across multiple frames.",
            "Count only the relevant objects supported by visible evidence in the images.",
            "Do not double-count the same object across frames.",
            "Answer with exactly one word: yes or no.",
            "",
            "Question:",
            str(question.get("question_text", "")).strip(),
            "",
            "Candidate count answer:",
            option_body,
            "",
            "Is this candidate count the best supported answer to the question?",
            "Answer:",
        ]
    )


def _build_object_counting_bbox_prompt(
    question: dict[str, Any],
    candidate_labels: list[str],
) -> str:
    label_block = ", ".join(candidate_labels)
    return "\n".join(
        [
            "You are identifying distinct visible object types in an egocentric industrial task video.",
            "The provided images are in chronological order.",
            "Use only visible evidence from the images.",
            "Use labels exactly from the candidate object list below.",
            "Output at most one coarse bounding box per label.",
            (
                "If multiple instances of the same label are visible, output only the single "
                "most confident bbox for that label."
            ),
            "Bounding boxes must use permille coordinates: x/y in [0, 1000].",
            "Answer with one item per line in exactly this format:",
            "<label>: [x1, y1, x2, y2]",
            "If none of the candidate object types are visible, answer NONE.",
            "",
            "Candidate object labels:",
            label_block,
            "",
            "Question:",
            str(question.get("question_text", "")).strip(),
        ]
    )


def _build_spatial_point_prompt(
    question: dict[str, Any],
    point_output_count: int,
) -> str:
    question_text = str(question.get("question_text", "")).strip()
    lines = [
        "You are localizing a held object in an egocentric industrial task video.",
        "The provided images are in chronological order.",
        "Use only visible evidence from the images.",
        "Coordinates must be in permille of the image size:",
        "- x=0 is the left edge and x=1000 is the right edge.",
        "- y=0 is the top edge and y=1000 is the bottom edge.",
    ]
    if point_output_count <= 1:
        lines.extend(
            [
                "Return one approximate point on the referenced object at the queried moment.",
                "If the object spans an area, choose a point near its visible center.",
                "Answer with exactly this format and nothing else: x=<integer>, y=<integer>",
            ]
        )
    else:
        lines.extend(
            [
                (
                    f"Return exactly {point_output_count} approximate points on the "
                    "referenced object at the queried moment."
                ),
                "Spread the points across the visible extent of the object when possible.",
                (
                    "Answer with exactly this format and nothing else: "
                    "[(x1,y1), (x2,y2), ...]"
                ),
            ]
        )
    lines.extend(["", "Question:", question_text])
    return "\n".join(lines)


def _build_not_visible_mcq_prompt(question: dict[str, Any]) -> str:
    options = question.get("options", [])
    if len(options) != 4:
        raise ValueError(f"Expected 4 options, got {len(options)}.")
    option_block = "\n".join(normalize_option(str(option)) for option in options)
    question_text = _normalize_visibility_question_text(
        str(question.get("question_text", "")).strip(),
    )
    button_hint = _visibility_button_hint(question)
    return "\n".join(
        [
            "You are answering an object-visibility multiple-choice question about an egocentric industrial task video.",
            "The provided images are in chronological order.",
            "Use only visible evidence from the images.",
            # "Use the specific object categories defined by the answer choices, not broader umbrella categories.",
            "Use the specific object categories defined by the answer choices.",
            button_hint,
            "Exactly one option should be chosen as the object that is not visible in this video segment.",
            "",
            "Question:",
            question_text,
            "",
            "Options:",
            option_block,
            "",
            "Which option corresponds to the object that is not visible in the provided images?",
            "Answer with only the single letter: A, B, C, or D.",
        ]
    )


def _build_any_frame_presence_prompt(
    question: dict[str, Any],
    option_text: str,
) -> str:
    option_body = extract_option_body(option_text)
    question_text = _normalize_visibility_question_text(
        str(question.get("question_text", "")).strip(),
    )
    button_hint = _visibility_button_hint(question)
    return "\n".join(
        [
            "You are verifying one candidate object for an object-visibility question in an egocentric industrial task video.",
            "The provided images are in chronological order.",
            "Use only visible evidence from the images.",
            "Use the specific object categories defined by the answer choices, not broader umbrella categories.",
            button_hint,
            "Count an object as visible only if it is shown clearly enough to be confidently identified.",
            "A tiny, heavily occluded, cropped, blurry, or ambiguous partial view does not count as visible.",
            "Answer with exactly one word: yes or no.",
            "",
            "Question:",
            question_text,
            "",
            f"Candidate object: {option_body}",
            "",
            "Is this object clearly visible in the provided images?",
            "Answer:",
        ]
    )


def _build_not_visible_pair_prompt(
    question: dict[str, Any],
    candidate_a: str,
    candidate_b: str,
) -> str:
    question_text = _normalize_visibility_question_text(
        str(question.get("question_text", "")).strip(),
    )
    button_hint = _visibility_button_hint(question)
    return "\n".join(
        [
            "You are comparing two candidate objects for an object-visibility question in an egocentric industrial task video.",
            "The provided images are in chronological order.",
            "Use only visible evidence from the images.",
            "Use the specific object categories defined by the answer choices, not broader umbrella categories.",
            button_hint,
            "Treat an object as visible only if it is clearly shown and can be confidently identified.",
            "A tiny, heavily occluded, cropped, blurry, or ambiguous partial view does not count as clearly visible.",
            "Choose the object that is less clearly visible in the provided images.",
            "Answer with exactly one letter: A or B.",
            "",
            "Question:",
            question_text,
            "",
            f"Candidate A: {candidate_a}",
            f"Candidate B: {candidate_b}",
            "",
            "Which candidate object is less clearly visible in the provided images?",
            "Answer:",
        ]
    )


def _candidate_token_ids(processor: Any, words: tuple[str, ...]) -> list[int]:
    tokenizer = processor.tokenizer
    token_ids = []
    for word in words:
        variants = (word, f" {word}", word.capitalize())
        variants += (f" {word.capitalize()}",)
        for variant in variants:
            ids = tokenizer.encode(variant, add_special_tokens=False)
            if ids:
                token_ids.append(ids[0])
    return sorted(set(token_ids))


def _verify_option(
    model: Any,
    processor: Any,
    image_paths: list[str],
    question: dict[str, Any],
    option_text: str,
    yes_token_ids: list[int],
    no_token_ids: list[int],
    verify_prompt_builder: Callable[[dict[str, Any], str], str],
    frame_prefix_texts: list[str] | None = None,
) -> dict[str, Any]:
    prompt = verify_prompt_builder(question, option_text)
    yes_logprob, no_logprob = _next_token_logprobs(
        model=model,
        processor=processor,
        image_paths=image_paths,
        frame_prefix_texts=frame_prefix_texts,
        prompt=prompt,
        yes_token_ids=yes_token_ids,
        no_token_ids=no_token_ids,
    )
    yes_prob = _binary_probability(yes_logprob, no_logprob)
    return {
        "option": normalize_option(option_text),
        "yes_logprob": yes_logprob,
        "no_logprob": no_logprob,
        "yes_probability": yes_prob,
    }


def _verify_option_chunks(
    model: Any,
    processor: Any,
    question: dict[str, Any],
    option_text: str,
    yes_token_ids: list[int],
    no_token_ids: list[int],
    verify_prompt_builder: Callable[[dict[str, Any], str], str],
    frame_prefix_texts: list[str] | None,
    frame_chunks: list[dict[str, Any]],
    aggregate_mode: str = "max_visible_probability",
) -> dict[str, Any]:
    prompt = verify_prompt_builder(question, option_text)
    chunk_scores = []
    for chunk in frame_chunks:
        start = int(chunk["start_offset"])
        end = int(chunk["end_offset_exclusive"])
        chunk_prefixes = (
            frame_prefix_texts[start:end] if frame_prefix_texts is not None else None
        )
        yes_logprob, no_logprob = _next_token_logprobs(
            model=model,
            processor=processor,
            image_paths=list(chunk["image_paths"]),
            frame_prefix_texts=chunk_prefixes,
            prompt=prompt,
            yes_token_ids=yes_token_ids,
            no_token_ids=no_token_ids,
        )
        visible_prob = _binary_probability(yes_logprob, no_logprob)
        chunk_scores.append(
            {
                "chunk_index": chunk["chunk_index"],
                "original_indices": chunk["original_indices"],
                "num_frames": len(chunk["image_paths"]),
                "yes_logprob": yes_logprob,
                "no_logprob": no_logprob,
                "evidence_margin": yes_logprob - no_logprob,
                "visible_probability": visible_prob,
            }
        )

    if aggregate_mode == "max_visible_probability":
        best_chunk = max(chunk_scores, key=lambda item: item["visible_probability"])
        return {
            "option": normalize_option(option_text),
            "visible_probability": best_chunk["visible_probability"],
            "yes_logprob": best_chunk["yes_logprob"],
            "no_logprob": best_chunk["no_logprob"],
            "best_visible_chunk_index": best_chunk["chunk_index"],
            "chunk_scores": chunk_scores,
            "chunk_aggregation": aggregate_mode,
        }

    if aggregate_mode == "frame_weighted_mean_margin":
        total_frames = sum(max(1, int(item["num_frames"])) for item in chunk_scores)
        weighted_margin = sum(
            float(item["evidence_margin"]) * max(1, int(item["num_frames"]))
            for item in chunk_scores
        ) / total_frames
        weighted_yes_prob = sum(
            float(item["visible_probability"]) * max(1, int(item["num_frames"]))
            for item in chunk_scores
        ) / total_frames
        best_chunk = max(chunk_scores, key=lambda item: item["evidence_margin"])
        return {
            "option": normalize_option(option_text),
            "predominant_score": weighted_margin,
            "mean_yes_probability": weighted_yes_prob,
            "yes_logprob": best_chunk["yes_logprob"],
            "no_logprob": best_chunk["no_logprob"],
            "best_supporting_chunk_index": best_chunk["chunk_index"],
            "chunk_scores": chunk_scores,
            "chunk_aggregation": aggregate_mode,
        }

    raise ValueError(f"Unknown aggregate_mode: {aggregate_mode}")


def _score_next_interaction_pair(
    model: Any,
    processor: Any,
    image_paths: list[str],
    question: dict[str, Any],
    option_bodies: list[str],
    option_i: int,
    option_j: int,
    choice_a_token_ids: list[int],
    choice_b_token_ids: list[int],
    frame_prefix_texts: list[str] | None = None,
) -> dict[str, Any]:
    candidate_i = option_bodies[option_i]
    candidate_j = option_bodies[option_j]

    prompt_ij = _build_next_interaction_pair_prompt(question, candidate_i, candidate_j)
    logp_a_ij, logp_b_ij = _next_choice_logprobs(
        model=model,
        processor=processor,
        image_paths=image_paths,
        prompt=prompt_ij,
        choice_a_token_ids=choice_a_token_ids,
        choice_b_token_ids=choice_b_token_ids,
        frame_prefix_texts=frame_prefix_texts,
    )

    prompt_ji = _build_next_interaction_pair_prompt(question, candidate_j, candidate_i)
    logp_a_ji, logp_b_ji = _next_choice_logprobs(
        model=model,
        processor=processor,
        image_paths=image_paths,
        prompt=prompt_ji,
        choice_a_token_ids=choice_a_token_ids,
        choice_b_token_ids=choice_b_token_ids,
        frame_prefix_texts=frame_prefix_texts,
    )

    margin_i_over_j = ((logp_a_ij - logp_b_ij) + (logp_b_ji - logp_a_ji)) / 2.0
    return {
        "letter_i": _LETTERS[option_i],
        "letter_j": _LETTERS[option_j],
        "option_i": candidate_i,
        "option_j": candidate_j,
        "margin_i_over_j": margin_i_over_j,
        "runs": [
            {
                "A": _LETTERS[option_i],
                "B": _LETTERS[option_j],
                "logp_A": logp_a_ij,
                "logp_B": logp_b_ij,
            },
            {
                "A": _LETTERS[option_j],
                "B": _LETTERS[option_i],
                "logp_A": logp_a_ji,
                "logp_B": logp_b_ji,
            },
        ],
    }


def _score_not_visible_pair(
    model: Any,
    processor: Any,
    image_paths: list[str],
    question: dict[str, Any],
    option_bodies: list[str],
    option_i: int,
    option_j: int,
    choice_a_token_ids: list[int],
    choice_b_token_ids: list[int],
    frame_prefix_texts: list[str] | None = None,
) -> dict[str, Any]:
    candidate_i = option_bodies[option_i]
    candidate_j = option_bodies[option_j]

    prompt_ij = _build_not_visible_pair_prompt(question, candidate_i, candidate_j)
    logp_a_ij, logp_b_ij = _next_choice_logprobs(
        model=model,
        processor=processor,
        image_paths=image_paths,
        prompt=prompt_ij,
        choice_a_token_ids=choice_a_token_ids,
        choice_b_token_ids=choice_b_token_ids,
        frame_prefix_texts=frame_prefix_texts,
    )

    prompt_ji = _build_not_visible_pair_prompt(question, candidate_j, candidate_i)
    logp_a_ji, logp_b_ji = _next_choice_logprobs(
        model=model,
        processor=processor,
        image_paths=image_paths,
        prompt=prompt_ji,
        choice_a_token_ids=choice_a_token_ids,
        choice_b_token_ids=choice_b_token_ids,
        frame_prefix_texts=frame_prefix_texts,
    )

    margin_i_over_j = ((logp_a_ij - logp_b_ij) + (logp_b_ji - logp_a_ji)) / 2.0
    return {
        "letter_i": _LETTERS[option_i],
        "letter_j": _LETTERS[option_j],
        "option_i": candidate_i,
        "option_j": candidate_j,
        "margin_i_over_j": margin_i_over_j,
        "runs": [
            {
                "A": _LETTERS[option_i],
                "B": _LETTERS[option_j],
                "logp_A": logp_a_ij,
                "logp_B": logp_b_ij,
            },
            {
                "A": _LETTERS[option_j],
                "B": _LETTERS[option_i],
                "logp_A": logp_a_ji,
                "logp_B": logp_b_ji,
            },
        ],
    }


def _dominant_hand_context(question_text: str) -> str | None:
    lowered = question_text.lower()
    for phrase in (
        "operator's left hand",
        "operator's right hand",
        "operator left hand",
        "operator right hand",
    ):
        if phrase in lowered:
            return phrase
    return None


def _parse_permille_points(raw_answer: str | None, expected_count: int = 1) -> dict[str, Any]:
    if raw_answer is None:
        return {"status": "empty", "error": "empty_output"}
    text = str(raw_answer).strip()
    if not text:
        return {"status": "empty", "error": "empty_output"}

    expected_count = max(1, int(expected_count))
    labeled_matches = list(_POINT_WITH_LABEL_RE.finditer(text))
    plain_matches = list(_POINT_PLAIN_RE.finditer(text)) if not labeled_matches else []
    matches = labeled_matches or plain_matches
    if not matches:
        return {"status": "invalid_point_format", "error": "unsupported_point_format"}

    points = []
    for match in matches:
        x = int(match.group(1))
        y = int(match.group(2))
        if not (0 <= x <= 1000 and 0 <= y <= 1000):
            return {
                "status": "invalid_point_range",
                "error": f"point_out_of_range({x},{y})",
            }
        points.append({"x": x, "y": y})
    warning = ""
    if len(points) != expected_count:
        warning = f"expected_{expected_count}_points_got_{len(points)}"
    return {"status": "ok", "points": points, "warning": warning}


def _parse_counting_labeled_bboxes(
    raw_answer: str | None,
    candidate_labels: list[str],
) -> dict[str, Any]:
    if raw_answer is None:
        return {"status": "empty", "error": "empty_output"}
    text = str(raw_answer).strip()
    if not text:
        return {"status": "empty", "error": "empty_output"}

    candidate_map = {
        _normalize_candidate_label(label): str(label).strip()
        for label in candidate_labels
        if str(label).strip()
    }
    if text.upper() == "NONE":
        return {
            "status": "ok",
            "recognized_boxes": [],
            "ignored_boxes": [],
            "unique_labels": [],
            "unique_label_count": 0,
            "warning": "",
        }

    matches = list(_COUNTING_BBOX_JSON_RE.finditer(text))
    if not matches:
        matches = list(_COUNTING_BBOX_LINE_RE.finditer(text))
    if not matches:
        return {"status": "invalid_bbox_format", "error": "unsupported_bbox_format"}

    recognized_boxes = []
    ignored_boxes = []
    geometry_warnings = []
    seen_labels: set[str] = set()
    unique_labels: list[str] = []
    for match in matches:
        label_text = str(match.group("label")).strip()
        label_key = _normalize_candidate_label(label_text)
        try:
            x1 = int(match.group("x1"))
            y1 = int(match.group("y1"))
            x2 = int(match.group("x2"))
            y2 = int(match.group("y2"))
        except (TypeError, ValueError):
            ignored_boxes.append({"label": label_text, "reason": "invalid_coordinate_cast"})
            continue
        if not all(0 <= value <= 1000 for value in (x1, y1, x2, y2)):
            ignored_boxes.append(
                {
                    "label": label_text,
                    "bbox": [x1, y1, x2, y2],
                    "reason": "coordinate_out_of_range",
                }
            )
            continue

        ordered_x1, ordered_x2 = sorted((x1, x2))
        ordered_y1, ordered_y2 = sorted((y1, y2))
        if ordered_x1 == ordered_x2 or ordered_y1 == ordered_y2:
            ignored_boxes.append(
                {
                    "label": label_text,
                    "bbox": [x1, y1, x2, y2],
                    "reason": "degenerate_bbox",
                }
            )
            continue
        if (ordered_x1, ordered_y1, ordered_x2, ordered_y2) != (x1, y1, x2, y2):
            geometry_warnings.append(f"reordered_bbox_for_{label_text}")

        canonical_label = candidate_map.get(label_key)
        if canonical_label is None:
            ignored_boxes.append(
                {
                    "label": label_text,
                    "bbox": [ordered_x1, ordered_y1, ordered_x2, ordered_y2],
                    "reason": "label_not_in_candidate_list",
                }
            )
            continue

        recognized_boxes.append(
            {
                "label": canonical_label,
                "raw_label": label_text,
                "bbox": [ordered_x1, ordered_y1, ordered_x2, ordered_y2],
            }
        )
        if canonical_label not in seen_labels:
            seen_labels.add(canonical_label)
            unique_labels.append(canonical_label)

    if not recognized_boxes and not ignored_boxes:
        return {"status": "invalid_bbox_format", "error": "unsupported_bbox_format"}

    warning_parts = []
    if ignored_boxes:
        warning_parts.append(f"ignored_{len(ignored_boxes)}_boxes")
    if geometry_warnings:
        warning_parts.extend(geometry_warnings)
    return {
        "status": "ok",
        "recognized_boxes": recognized_boxes,
        "ignored_boxes": ignored_boxes,
        "unique_labels": unique_labels,
        "unique_label_count": len(unique_labels),
        "warning": ";".join(warning_parts),
    }


def _normalize_candidate_label(label: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", " ", str(label).strip().lower())
    return " ".join(normalized.split())


def _numeric_option_counts(options: list[str]) -> list[dict[str, Any]]:
    counts = []
    for idx, option in enumerate(options):
        body = extract_option_body(str(option)).strip()
        try:
            count = int(body)
        except ValueError:
            continue
        counts.append(
            {
                "letter": _LETTERS[idx],
                "count": count,
                "option": normalize_option(str(option)),
            }
        )
    return counts


def _choose_count_option_from_value(
    raw_count: int,
    option_counts: list[dict[str, Any]],
) -> dict[str, Any]:
    selected = min(
        option_counts,
        key=lambda item: (
            abs(int(item["count"]) - int(raw_count)),
            -int(item["count"]),
        ),
    )
    return {
        "answer_letter": str(selected["letter"]),
        "selected_count": int(selected["count"]),
        "raw_detected_count": int(raw_count),
        "option_counts": option_counts,
    }


def _resolve_option_region_centers(options: list[str]) -> list[dict[str, Any]]:
    centers = []
    for idx, option_text in enumerate(options):
        option_body = extract_option_body(str(option_text))
        canonical_region = _canonicalize_region_label(option_body)
        center_x, center_y = _center_for_region_label(canonical_region)
        centers.append(
            {
                "letter": _LETTERS[idx],
                "option": normalize_option(str(option_text)),
                "option_body": option_body,
                "canonical_region": canonical_region,
                "center_x": center_x,
                "center_y": center_y,
            }
        )
    return centers


def _canonicalize_region_label(region_text: str) -> str:
    normalized = re.sub(r"[^a-z]+", "-", region_text.strip().lower()).strip("-")
    normalized = normalized.replace("middle", "center")
    if not normalized:
        raise ValueError(f"Unsupported empty region label from option: {region_text!r}")
    if normalized == "center":
        return "center"

    parts = [part for part in normalized.split("-") if part]
    if len(parts) != 2:
        raise ValueError(f"Unsupported region label: {region_text!r}")

    vertical = None
    horizontal = None
    for part in parts:
        if part in {"top", "center", "bottom"} and vertical is None:
            vertical = part
            continue
        if part in {"left", "center", "right"} and horizontal is None:
            horizontal = part
            continue
        raise ValueError(f"Unsupported region label: {region_text!r}")

    if vertical is None or horizontal is None:
        raise ValueError(f"Unsupported region label: {region_text!r}")
    if vertical == "center" and horizontal == "center":
        return "center"
    return f"{vertical}-{horizontal}"


def _center_for_region_label(region_label: str) -> tuple[int, int]:
    if region_label == "center":
        return (_GRID_AXIS_TO_PERMILLE["center"], _GRID_AXIS_TO_PERMILLE["center"])

    vertical, horizontal = region_label.split("-", maxsplit=1)
    if vertical not in {"top", "center", "bottom"}:
        raise ValueError(f"Unsupported vertical region token: {vertical!r}")
    if horizontal not in {"left", "center", "right"}:
        raise ValueError(f"Unsupported horizontal region token: {horizontal!r}")
    return (_GRID_AXIS_TO_PERMILLE[horizontal], _GRID_AXIS_TO_PERMILLE[vertical])


def _next_token_logprobs(
    model: Any,
    processor: Any,
    image_paths: list[str],
    prompt: str,
    yes_token_ids: list[int],
    no_token_ids: list[int],
    frame_prefix_texts: list[str] | None = None,
) -> tuple[float, float]:
    # pylint: disable=import-outside-toplevel
    import torch

    if frame_prefix_texts is not None and len(frame_prefix_texts) != len(image_paths):
        raise ValueError("frame_prefix_texts must match image_paths length.")

    content: list[dict[str, Any]] = []
    for idx, image_path in enumerate(image_paths):
        if frame_prefix_texts is not None:
            content.append({"type": "text", "text": frame_prefix_texts[idx]})
        content.append({"type": "image", "image": image_path})
    content.append({"type": "text", "text": prompt})
    messages = [{"role": "user", "content": content}]
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)
    outputs = model(**inputs)
    logits = outputs.logits[:, -1, :]
    log_probs = torch.log_softmax(logits, dim=-1)
    yes_scores = log_probs[0, yes_token_ids]
    no_scores = log_probs[0, no_token_ids]
    return (
        float(torch.max(yes_scores).detach().cpu()),
        float(torch.max(no_scores).detach().cpu()),
    )


def _next_choice_logprobs(
    model: Any,
    processor: Any,
    image_paths: list[str],
    prompt: str,
    choice_a_token_ids: list[int],
    choice_b_token_ids: list[int],
    frame_prefix_texts: list[str] | None = None,
) -> tuple[float, float]:
    # pylint: disable=import-outside-toplevel
    import torch

    if frame_prefix_texts is not None and len(frame_prefix_texts) != len(image_paths):
        raise ValueError("frame_prefix_texts must match image_paths length.")

    content: list[dict[str, Any]] = []
    for idx, image_path in enumerate(image_paths):
        if frame_prefix_texts is not None:
            content.append({"type": "text", "text": frame_prefix_texts[idx]})
        content.append({"type": "image", "image": image_path})
    content.append({"type": "text", "text": prompt})
    messages = [{"role": "user", "content": content}]
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)
    outputs = model(**inputs)
    logits = outputs.logits[:, -1, :]
    log_probs = torch.log_softmax(logits, dim=-1)
    choice_a_scores = log_probs[0, choice_a_token_ids]
    choice_b_scores = log_probs[0, choice_b_token_ids]
    return (
        float(torch.max(choice_a_scores).detach().cpu()),
        float(torch.max(choice_b_scores).detach().cpu()),
    )


def _next_mcq_option_logprobs(
    model: Any,
    processor: Any,
    image_paths: list[str],
    prompt: str,
    mcq_token_ids: dict[str, list[int]],
    frame_prefix_texts: list[str] | None = None,
) -> dict[str, float]:
    # pylint: disable=import-outside-toplevel
    import torch

    if frame_prefix_texts is not None and len(frame_prefix_texts) != len(image_paths):
        raise ValueError("frame_prefix_texts must match image_paths length.")

    content: list[dict[str, Any]] = []
    for idx, image_path in enumerate(image_paths):
        if frame_prefix_texts is not None:
            content.append({"type": "text", "text": frame_prefix_texts[idx]})
        content.append({"type": "image", "image": image_path})
    content.append({"type": "text", "text": prompt})
    messages = [{"role": "user", "content": content}]
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)
    outputs = model(**inputs)
    logits = outputs.logits[:, -1, :]
    log_probs = torch.log_softmax(logits, dim=-1)
    return {
        letter: float(torch.max(log_probs[0, token_ids]).detach().cpu())
        for letter, token_ids in mcq_token_ids.items()
    }


def _generate_text_response(
    model: Any,
    processor: Any,
    image_paths: list[str],
    prompt: str,
    max_new_tokens: int,
    frame_prefix_texts: list[str] | None = None,
) -> str:
    # pylint: disable=import-outside-toplevel
    import torch

    if frame_prefix_texts is not None and len(frame_prefix_texts) != len(image_paths):
        raise ValueError("frame_prefix_texts must match image_paths length.")

    content: list[dict[str, Any]] = []
    for idx, image_path in enumerate(image_paths):
        if frame_prefix_texts is not None:
            content.append({"type": "text", "text": frame_prefix_texts[idx]})
        content.append({"type": "image", "image": image_path})
    content.append({"type": "text", "text": prompt})
    messages = [{"role": "user", "content": content}]
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )
    generated_ids_trimmed = [
        out_ids[len(in_ids) :]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return output_text[0].strip()


def _generate_video_text_response(
    model: Any,
    processor: Any,
    image_paths: list[str],
    prompt: str,
    max_new_tokens: int,
    sample_fps: float = 1.0,
    video_max_pixels: int | None = None,
) -> str:
    # pylint: disable=import-outside-toplevel
    import torch

    try:
        from qwen_vl_utils import process_vision_info
    except ImportError as err:
        raise ImportError(
            "Video-frame input requires qwen-vl-utils. Install it with "
            "`pip install qwen-vl-utils`.",
        ) from err

    video_content: dict[str, Any] = {
        "type": "video",
        "video": [_as_file_uri(image_path) for image_path in image_paths],
        "sample_fps": float(sample_fps),
    }
    if video_max_pixels is not None:
        video_content["max_pixels"] = int(video_max_pixels)

    messages = [
        {
            "role": "user",
            "content": [
                video_content,
                {"type": "text", "text": prompt},
            ],
        },
    ]
    text = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )
    generated_ids_trimmed = [
        out_ids[len(in_ids) :]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return output_text[0].strip()


def _as_file_uri(path: str) -> str:
    return Path(path).resolve().as_uri()


def _binary_probability(yes_logprob: float, no_logprob: float) -> float:
    max_logprob = max(yes_logprob, no_logprob)
    yes_score = math.exp(yes_logprob - max_logprob)
    no_score = math.exp(no_logprob - max_logprob)
    return yes_score / (yes_score + no_score)
