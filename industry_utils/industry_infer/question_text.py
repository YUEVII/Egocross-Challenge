"""Question text time-window handling and effective-frame filtering."""

from __future__ import annotations

import math
import re
from typing import Any

from industry_infer.frame_time import first_slot_starts_at_zero
from industry_infer.frame_time import seconds_per_slot
from industry_infer.parsing import extract_option_body

TIME_FROM_TO_RE = re.compile(
    r"from\s+([\d.]+)\s*s\s+to\s+([\d.]+)\s*s",
    re.IGNORECASE,
)
WITHIN_SEGMENT_RE = re.compile(
    r"\(\s*within\s+segment\s+([\d.]+)\s*s\s*-\s*([\d.]+)\s*s\s*\)",
    re.IGNORECASE,
)
QUESTION_POINT_TIMESTAMP_RE = re.compile(
    r"(?:around\s+timestamp|around|at)\s+([\d.]+)\s*s\b",
    re.IGNORECASE,
)
OPTION_POINT_TIMESTAMP_RE = re.compile(r"^\s*([\d.]+)\s*s?\s*$", re.IGNORECASE)


def _find_time_window_match(
    question_text: str,
) -> tuple[re.Match[str], float, float, str] | None:
    for regex, source in (
        (TIME_FROM_TO_RE, "from_to"),
        (WITHIN_SEGMENT_RE, "within_segment"),
    ):
        match = regex.search(question_text)
        if not match:
            continue
        try:
            start_sec = float(match.group(1))
            end_sec = float(match.group(2))
        except ValueError:
            return None
        return match, start_sec, end_sec, source
    return None


def _cleanup_question_text(text: str) -> str:
    text = re.sub(r"\(\s*\)", "", text)
    text = re.sub(r"\s{2,}", " ", text)
    text = re.sub(r"\s+([?.!,;:])", r"\1", text)
    return text.strip()


def extract_time_window_sec(question_text: str) -> tuple[float, float] | None:
    found = _find_time_window_match(question_text)
    if found is None:
        return None
    _, start_sec, end_sec, _ = found
    return (start_sec, end_sec)


def is_whole_clip_redundant(
    num_frames: int,
    start_sec: float,
    end_sec: float,
    effective_fps: float,
) -> bool:
    duration = abs(end_sec - start_sec)
    if duration <= 0:
        return False
    step = 1.0 / effective_fps
    lo = max(1, math.floor(duration / step))
    hi = max(1, math.ceil(duration / step) + 1)
    if lo > hi:
        lo, hi = hi, lo
    return lo <= num_frames <= hi


def strip_redundant_time_window(
    question_text: str,
    num_frames: int,
    effective_fps: float,
) -> tuple[str, dict[str, Any]]:
    """Remove the first matching time span if it is redundant with ``num_frames``."""

    found = _find_time_window_match(question_text)
    if found is None:
        return question_text, {"stripped": False, "reason": "no_time_span"}

    match, start_sec, end_sec, source = found
    if not is_whole_clip_redundant(num_frames, start_sec, end_sec, effective_fps):
        return question_text, {
            "stripped": False,
            "reason": "not_redundant_whole_clip",
            "span": [start_sec, end_sec],
            "match_source": source,
        }

    cleaned = question_text[: match.start()] + question_text[match.end() :]
    cleaned = _cleanup_question_text(cleaned)
    return cleaned, {
        "stripped": True,
        "span": [start_sec, end_sec],
        "effective_fps": effective_fps,
        "num_frames": num_frames,
        "match_source": source,
    }


def extract_question_point_timestamp_sec(question_text: str) -> float | None:
    """Extract an explicit single timestamp like ``around timestamp 2.76s``."""

    match = QUESTION_POINT_TIMESTAMP_RE.search(question_text)
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def extract_option_point_timestamps_sec(options: list[str]) -> list[float]:
    """Extract numeric timestamp options like ``A: 5.7s``."""

    timestamps: list[float] = []
    for option in options:
        body = extract_option_body(str(option))
        match = OPTION_POINT_TIMESTAMP_RE.match(body)
        if not match:
            continue
        try:
            timestamps.append(float(match.group(1)))
        except ValueError:
            continue
    return timestamps


def frame_timestamps_for_question(
    num_frames: int,
    effective_fps: float,
    *,
    question_text: str,
    options: list[str],
) -> list[float]:
    """Return the wall-clock timestamp attached to each released frame."""

    slot = seconds_per_slot(effective_fps)
    timeline_start = 0.0 if first_slot_starts_at_zero(question_text, options) else slot
    return [timeline_start + idx * slot for idx in range(num_frames)]


def select_option_timepoint_neighbor_frames(
    question_text: str,
    options: list[str],
    image_paths: list[str],
    effective_fps: float,
) -> tuple[list[str], dict[str, Any]]:
    """Select the frame before/after each option timestamp and merge them."""

    target_timestamps = extract_option_point_timestamps_sec(options)
    if not image_paths:
        return [], {
            "applied": False,
            "reason": "no_frames",
            "selection_mode": "option_timepoint_neighbors",
        }
    if not target_timestamps:
        return list(image_paths), {
            "applied": False,
            "reason": "no_option_timestamps",
            "selection_mode": "option_timepoint_neighbors",
        }

    frame_timestamps = frame_timestamps_for_question(
        len(image_paths),
        effective_fps,
        question_text=question_text,
        options=options,
    )
    selected_indices: set[int] = set()
    targets = []
    for target_sec in target_timestamps:
        before_idx = 0
        for idx, timestamp_sec in enumerate(frame_timestamps):
            if timestamp_sec <= target_sec:
                before_idx = idx
            else:
                break

        after_idx = len(frame_timestamps) - 1
        for idx, timestamp_sec in enumerate(frame_timestamps):
            if timestamp_sec >= target_sec:
                after_idx = idx
                break

        selected_indices.add(before_idx)
        selected_indices.add(after_idx)
        targets.append(
            {
                "target_timestamp_sec": target_sec,
                "before_index": before_idx,
                "before_timestamp_sec": frame_timestamps[before_idx],
                "after_index": after_idx,
                "after_timestamp_sec": frame_timestamps[after_idx],
            }
        )

    ordered_indices = sorted(selected_indices)
    selected_paths = [image_paths[idx] for idx in ordered_indices]
    return selected_paths, {
        "applied": True,
        "selection_mode": "option_timepoint_neighbors",
        "target_timestamps_sec": target_timestamps,
        "targets": targets,
        "selected_frame_indices": ordered_indices,
        "selected_frame_timestamps_sec": [
            frame_timestamps[idx] for idx in ordered_indices
        ],
        "frame_timestamps_sec": frame_timestamps,
    }


def select_question_timepoint_neighborhood_frames(
    question_text: str,
    options: list[str],
    image_paths: list[str],
    effective_fps: float,
    *,
    radius: int,
) -> tuple[list[str], dict[str, Any]]:
    """Select a symmetric frame neighborhood around one explicit timestamp."""

    if not image_paths:
        return [], {
            "applied": False,
            "reason": "no_frames",
            "selection_mode": "question_timepoint_neighborhood",
        }

    target_sec = extract_question_point_timestamp_sec(question_text)
    if target_sec is None:
        return list(image_paths), {
            "applied": False,
            "reason": "no_question_timestamp",
            "selection_mode": "question_timepoint_neighborhood",
        }

    frame_timestamps = frame_timestamps_for_question(
        len(image_paths),
        effective_fps,
        question_text=question_text,
        options=options,
    )
    nearest_idx = min(
        range(len(frame_timestamps)),
        key=lambda idx: abs(frame_timestamps[idx] - target_sec),
    )
    radius = max(0, int(radius))
    start = max(0, nearest_idx - radius)
    end = min(len(frame_timestamps), nearest_idx + radius + 1)
    ordered_indices = list(range(start, end))
    selected_paths = [image_paths[idx] for idx in ordered_indices]
    return selected_paths, {
        "applied": True,
        "selection_mode": "question_timepoint_neighborhood",
        "target_timestamp_sec": target_sec,
        "center_index": nearest_idx,
        "center_timestamp_sec": frame_timestamps[nearest_idx],
        "radius": radius,
        "selected_frame_indices": ordered_indices,
        "selected_frame_timestamps_sec": [
            frame_timestamps[idx] for idx in ordered_indices
        ],
        "frame_timestamps_sec": frame_timestamps,
    }


def resolve_question_window(
    question_text: str,
    options: list[str],
    image_paths: list[str],
    effective_fps: float,
) -> tuple[str, list[str], dict[str, Any]]:
    """Return cleaned text plus frames valid for the question's time window."""

    found = _find_time_window_match(question_text)
    num_frames = len(image_paths)
    if found is None:
        return question_text, list(image_paths), {
            "has_time_window": False,
            "window_is_whole_clip": False,
            "text_modified": False,
            "effective_range_sec": None,
            "effective_frame_indices": list(range(num_frames)),
            "effective_num_frames": num_frames,
            "reason": "no_time_span",
        }

    match, time0, time1, source = found
    start_sec, end_sec = sorted((time0, time1))
    if is_whole_clip_redundant(num_frames, start_sec, end_sec, effective_fps):
        cleaned = question_text[: match.start()] + question_text[match.end() :]
        cleaned = _cleanup_question_text(cleaned)
        return cleaned, list(image_paths), {
            "has_time_window": True,
            "window_is_whole_clip": True,
            "text_modified": cleaned != question_text,
            "span": [start_sec, end_sec],
            "match_source": source,
            "effective_range_sec": None,
            "effective_frame_indices": list(range(num_frames)),
            "effective_num_frames": num_frames,
            "reason": "whole_clip_time_span_stripped",
        }

    effective, eff_indices, timestamps, fallback = _filter_frames_by_window(
        image_paths,
        start_sec,
        end_sec,
        effective_fps,
        question_text=question_text,
        options=options,
    )
    return question_text, effective, {
        "has_time_window": True,
        "window_is_whole_clip": False,
        "text_modified": False,
        "span": [start_sec, end_sec],
        "match_source": source,
        "effective_range_sec": [start_sec, end_sec],
        "effective_frame_indices": eff_indices,
        "effective_num_frames": len(effective),
        "frame_timestamps_sec": timestamps,
        "fallback": fallback,
        "reason": "filtered_to_time_window",
    }


def _filter_frames_by_window(
    image_paths: list[str],
    start_sec: float,
    end_sec: float,
    effective_fps: float,
    *,
    question_text: str,
    options: list[str],
) -> tuple[list[str], list[int], list[float], str | None]:
    slot = seconds_per_slot(effective_fps)
    timeline_start = 0.0 if first_slot_starts_at_zero(question_text, options) else slot

    timestamps = [timeline_start + idx * slot for idx in range(len(image_paths))]
    if not timestamps:
        return [], [], [], None

    first_idx = 0
    for idx, timestamp_sec in enumerate(timestamps):
        if timestamp_sec <= start_sec:
            first_idx = idx
        else:
            break

    last_idx = len(timestamps) - 1
    for idx, timestamp_sec in enumerate(timestamps):
        if timestamp_sec >= end_sec:
            last_idx = idx
            break

    if first_idx > last_idx:
        center = (start_sec + end_sec) / 2.0
        best_idx = min(
            range(len(timestamps)),
            key=lambda idx: abs(timestamps[idx] - center),
        )
        return [image_paths[best_idx]], [best_idx], timestamps, "nearest_frame"

    indices = list(range(first_idx, last_idx + 1))
    selected = [image_paths[idx] for idx in indices]
    return selected, indices, timestamps, "expanded_to_nearest_frame_points"
