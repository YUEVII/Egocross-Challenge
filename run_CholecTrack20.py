#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import tempfile
from pathlib import Path
from typing import Any, Callable

from PIL import Image, ImageEnhance


VALID_ANSWERS = frozenset({"A", "B", "C", "D"})
_LETTERS = ("A", "B", "C", "D")
_ZERO_START_HINT = re.compile(
    r"(?:from\s+0(?:\.0+)?s\b|segment\s+from\s+0s\b|\b0\.00s\b|"
    r"\bfrom\s+0s\b|\bat\s+0(?:\.0+)?s\b)",
    re.IGNORECASE,
)
TIME_FROM_TO_RE = re.compile(
    r"from\s+([\d.]+)\s*s\s+to\s+([\d.]+)\s*s",
    re.IGNORECASE,
)
QUESTION_POINT_TIMESTAMP_RE = re.compile(
    r"(?:around\s+timestamp|around|at)\s+([\d.]+)\s*s\b",
    re.IGNORECASE,
)
OPTION_POINT_TIMESTAMP_RE = re.compile(r"^\s*([\d.]+)\s*s?\s*$", re.IGNORECASE)
_POINT_WITH_LABEL_RE = re.compile(
    r"x\s*[:=]\s*(\d{1,4})\s*[,;]\s*y\s*[:=]\s*(\d{1,4})",
    re.IGNORECASE,
)
_POINT_PLAIN_RE = re.compile(
    r"\(?\s*(\d{1,4})\s*[,;]\s*(\d{1,4})\s*\)?",
    re.IGNORECASE,
)
_GRID_AXIS_TO_PERMILLE = {
    "left": 167,
    "center": 500,
    "right": 833,
    "top": 167,
    "bottom": 833,
}
_EARLIEST_Q_RE = re.compile(
    r"first\s+start|first\s+begins?|\bstart\b|\bbegin|\bfirst\b",
    re.IGNORECASE,
)

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET_JSON = SCRIPT_DIR / "testset" / "egocross_testbed_imgs.json"
DEFAULT_DATA_ROOT = SCRIPT_DIR / "testset"
DEFAULT_MODEL_PATH = SCRIPT_DIR / "ckpts" / "base_model"
if not DEFAULT_MODEL_PATH.exists():
    DEFAULT_MODEL_PATH = SCRIPT_DIR.parent / "ckpts" / "base_model"
DEFAULT_SUBMISSION_JSON = SCRIPT_DIR / "submission.json"

DATASET = "CholecTrack20"
ATL_QT = "action temporal localization"
Q3_QT = "next action prediction"
Q4_QT = "next phase prediction"
COUNT_QT = "object counting"
DHOI_QT = "dominant held-object identification"
NOT_VISIBLE_QT = "object not visible identification"
SPATIAL_QT = "object spatial localization"
DEFAULT_FPS = 0.5
Q3_Q4_MAX_FRAMES = 10
Q3_Q4_MAX_NEW_TOKENS = 32
BASE_MAX_FRAMES = 8
BASE_MAX_NEW_TOKENS = 64
SPATIAL_RADIUS = 1
SPATIAL_POINT_COUNT = 7
ATL2_YES_THRESHOLD = 0.6
ATL2_MARGIN_FB = 0.10
_PATCH_DIR: Path | None = None


def normalize_option(option_text: str) -> str:
    option_text = option_text.strip()
    if re.match(r"^[A-D]\s*[:.]", option_text):
        return f"{option_text[0]}. {option_text[2:].strip()}"
    return option_text


def parse_letter(raw_answer: str | None) -> tuple[str | None, str]:
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
    text = normalize_option(option_text)
    if len(text) >= 2 and text[0] in VALID_ANSWERS and text[1] == ".":
        return text[2:].strip()
    return text


def _load_json_list(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a JSON list.")
    return [item for item in data if isinstance(item, dict)]


def resolve_image_path(raw_path: str, data_root: Path) -> str:
    image_path = Path(raw_path)
    if image_path.is_absolute() and image_path.is_file():
        return str(image_path)
    relative_path = raw_path.lstrip("/")
    candidate = data_root / relative_path
    if candidate.is_file():
        return str(candidate)
    path_parts = Path(relative_path).parts
    if path_parts and path_parts[0] == data_root.name:
        deduped = data_root / Path(*path_parts[1:])
        if deduped.is_file():
            return str(deduped)
    return str(candidate)


def image_paths_for_question(question: dict[str, Any], data_root: Path) -> list[str]:
    paths: list[str] = []
    for raw_path in question.get("video_path", []):
        if isinstance(raw_path, str):
            paths.append(resolve_image_path(raw_path, data_root))
    return paths


def seconds_per_slot(effective_sampling_fps: float) -> float:
    if effective_sampling_fps <= 0:
        raise ValueError("effective_sampling_fps must be positive.")
    return 1.0 / effective_sampling_fps


def first_slot_starts_at_zero(question_text: str, options: list[str]) -> bool:
    blob = question_text + "\n" + "\n".join(str(o) for o in options)
    return bool(_ZERO_START_HINT.search(blob))


def build_frame_prefix_texts(
    num_frames: int,
    effective_sampling_fps: float,
    *,
    question_text: str,
    options: list[str],
) -> tuple[list[str], dict[str, Any]]:
    slot = seconds_per_slot(effective_sampling_fps)
    start_zero = first_slot_starts_at_zero(question_text, options)
    t0 = 0.0 if start_zero else slot
    prefixes: list[str] = []
    timestamps: list[float] = []
    for index in range(num_frames):
        timestamp_sec = t0 + index * slot
        timestamps.append(timestamp_sec)
        prefixes.append(
            f"Image {index + 1}/{num_frames} - approximately at {timestamp_sec:.1f} s in the clip."
        )
    return prefixes, {
        "effective_sampling_fps": effective_sampling_fps,
        "seconds_per_step": slot,
        "first_slot_starts_at_zero": start_zero,
        "first_frame_timestamp_sec": t0,
        "frame_timestamps_sec": timestamps,
    }


def build_timing_legend(num_frames: int, effective_sampling_fps: float) -> str:
    slot = seconds_per_slot(effective_sampling_fps)
    return "\n".join(
        [
            "Time reference: The images below are in chronological order (earliest to latest).",
            (
                f"Assume they were sampled at {effective_sampling_fps:g} FPS "
                f"along the clip: adjacent images are about {slot:.1f} s apart."
            ),
            f"You are given {num_frames} image(s); use the per-image timestamps to answer.",
        ]
    )


def build_direct_mcq_prompt(question_text: str, options: list[str]) -> str:
    lines = [
        question_text.strip(),
        "",
        *[normalize_option(str(option)) for option in options],
        "",
        "Answer with only the single letter: A, B, C, or D.",
    ]
    return "\n".join(lines)


_DEFAULT_SURGICAL_CONTEXT = """This is a laparoscopic cholecystectomy video. The procedure
follows seven phases in this canonical order:

1. Preparation
   - retract/grasp gallbladder; clear surrounding tissue
2. Calot's triangle dissection
   - dissect around gallbladder neck; expose cystic duct
     and artery; achieve critical view of safety
3. Clipping and cutting
   - clip cystic duct and cystic artery; cut between clips
4. Gallbladder dissection
   - dissect/coagulate gallbladder free from liver bed
5. Gallbladder packaging
   - place gallbladder into specimen retrieval bag
6. Cleaning and coagulation
   - aspirate fluid; coagulate bleeding; irrigate field
7. Gallbladder extraction
   - pull specimen bag out through trocar; close ports"""


def build_surgical_context_mcq_prompt(
    question_text: str,
    options: list[str],
    context_text: str | None = None,
) -> str:
    context = context_text.strip() if context_text is not None and context_text.strip() else _DEFAULT_SURGICAL_CONTEXT
    lines = [
        context,
        "",
        "Refer to the procedure context above and choose the option that best matches the problem.",
        "",
        question_text.strip(),
        "",
        *[normalize_option(str(o)) for o in options],
        "",
        "Answer with only the single letter: A, B, C, or D.",
    ]
    return "\n".join(lines)


def run_direct_mcq(
    model: Any,
    processor: Any,
    image_paths: list[str],
    prompt: str,
    max_new_tokens: int,
    frame_prefix_texts: list[str] | None = None,
) -> str:
    import torch
    if frame_prefix_texts is not None and len(frame_prefix_texts) != len(image_paths):
        raise ValueError("frame_prefix_texts must match image_paths length.")
    query_content: list[dict[str, Any]] = []
    for idx, image_path in enumerate(image_paths):
        if frame_prefix_texts is not None:
            query_content.append({"type": "text", "text": frame_prefix_texts[idx]})
        query_content.append({"type": "image", "image": image_path})
    query_content.append({"type": "text", "text": prompt})
    messages = [{"role": "user", "content": query_content}]
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
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return output_text[0].strip()


def uniform_sample_frames(
    paths: list[str],
    max_frames_per_call: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not paths:
        empty = {
            "image_paths": [],
            "original_indices": [],
            "sampled_offsets": [],
            "start_offset": 0,
            "end_offset_exclusive": 0,
        }
        return empty, {
            "strategy": "no_effective_frames",
            "max_frames_per_call": max_frames_per_call,
            "num_frames": 0,
            "num_sampled_frames": 0,
        }
    if max_frames_per_call <= 0 or len(paths) <= max_frames_per_call:
        full = {
            "image_paths": list(paths),
            "original_indices": list(range(len(paths))),
            "sampled_offsets": list(range(len(paths))),
            "start_offset": 0,
            "end_offset_exclusive": len(paths),
        }
        return full, {
            "strategy": "all_effective_frames",
            "max_frames_per_call": max_frames_per_call,
            "num_frames": len(paths),
            "num_sampled_frames": len(paths),
            "sampled_offsets": list(range(len(paths))),
        }
    if max_frames_per_call == 1:
        sampled_offsets = [len(paths) // 2]
    else:
        sampled_offsets = [
            int(round(i * (len(paths) - 1) / (max_frames_per_call - 1)))
            for i in range(max_frames_per_call)
        ]
    sampled_offsets = sorted(dict.fromkeys(sampled_offsets))
    sampled_paths = [paths[idx] for idx in sampled_offsets]
    sampled = {
        "image_paths": sampled_paths,
        "original_indices": sampled_offsets,
        "sampled_offsets": sampled_offsets,
        "start_offset": sampled_offsets[0],
        "end_offset_exclusive": sampled_offsets[-1] + 1,
    }
    return sampled, {
        "strategy": "uniform_sample",
        "max_frames_per_call": max_frames_per_call,
        "num_frames": len(paths),
        "num_sampled_frames": len(sampled_paths),
        "sampled_offsets": sampled_offsets,
    }


def chunk_frames(
    paths: list[str],
    max_frames_per_call: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not paths:
        return [], {
            "strategy": "no_effective_frames",
            "max_frames_per_call": max_frames_per_call,
            "num_chunks": 0,
            "num_frames": 0,
        }
    chunk_size = len(paths) if max_frames_per_call <= 0 else max_frames_per_call
    chunks: list[dict[str, Any]] = []
    for start in range(0, len(paths), chunk_size):
        end = min(start + chunk_size, len(paths))
        chunks.append(
            {
                "chunk_index": len(chunks),
                "image_paths": paths[start:end],
                "original_indices": list(range(start, end)),
                "start_offset": start,
                "end_offset_exclusive": end,
            }
        )
    return chunks, {
        "strategy": "chronological_chunks",
        "max_frames_per_call": max_frames_per_call,
        "num_chunks": len(chunks),
        "num_frames": len(paths),
        "chunk_sizes": [len(chunk["image_paths"]) for chunk in chunks],
    }


def extract_time_window_sec(question_text: str) -> tuple[float, float] | None:
    match = TIME_FROM_TO_RE.search(question_text)
    if not match:
        return None
    try:
        return float(match.group(1)), float(match.group(2))
    except ValueError:
        return None


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


def extract_question_point_timestamp_sec(question_text: str) -> float | None:
    match = QUESTION_POINT_TIMESTAMP_RE.search(question_text)
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def extract_option_point_timestamps_sec(options: list[str]) -> list[float]:
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
    slot = seconds_per_slot(effective_fps)
    timeline_start = 0.0 if first_slot_starts_at_zero(question_text, options) else slot
    return [timeline_start + idx * slot for idx in range(num_frames)]


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
    for idx, ts in enumerate(timestamps):
        if ts <= start_sec:
            first_idx = idx
        else:
            break
    last_idx = len(timestamps) - 1
    for idx, ts in enumerate(timestamps):
        if ts >= end_sec:
            last_idx = idx
            break
    if first_idx > last_idx:
        center = (start_sec + end_sec) / 2.0
        best_idx = min(range(len(timestamps)), key=lambda i: abs(timestamps[i] - center))
        return [image_paths[best_idx]], [best_idx], timestamps, "nearest_frame"
    indices = list(range(first_idx, last_idx + 1))
    selected = [image_paths[idx] for idx in indices]
    return selected, indices, timestamps, "expanded_to_nearest_frame_points"


def resolve_question_window(
    question_text: str,
    options: list[str],
    image_paths: list[str],
    effective_fps: float,
) -> tuple[str, list[str], dict[str, Any]]:
    match = TIME_FROM_TO_RE.search(question_text)
    num_frames = len(image_paths)
    if not match:
        return question_text, list(image_paths), {
            "has_time_window": False,
            "window_is_whole_clip": False,
            "text_modified": False,
            "effective_range_sec": None,
            "effective_frame_indices": list(range(num_frames)),
            "effective_num_frames": num_frames,
            "reason": "no_time_span",
        }
    try:
        t0, t1 = float(match.group(1)), float(match.group(2))
    except ValueError:
        return question_text, list(image_paths), {
            "has_time_window": True,
            "window_is_whole_clip": False,
            "text_modified": False,
            "effective_range_sec": None,
            "effective_frame_indices": list(range(num_frames)),
            "effective_num_frames": num_frames,
            "reason": "parse_error",
        }
    start_sec, end_sec = sorted((t0, t1))
    if is_whole_clip_redundant(num_frames, start_sec, end_sec, effective_fps):
        cleaned = question_text[: match.start()] + question_text[match.end():]
        cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()
        return cleaned, list(image_paths), {
            "has_time_window": True,
            "window_is_whole_clip": True,
            "text_modified": cleaned != question_text,
            "span": [start_sec, end_sec],
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
        "effective_range_sec": [start_sec, end_sec],
        "effective_frame_indices": eff_indices,
        "effective_num_frames": len(effective),
        "frame_timestamps_sec": timestamps,
        "fallback": fallback,
        "reason": "filtered_to_time_window",
    }


def select_option_timepoint_neighbor_frames(
    question_text: str,
    options: list[str],
    image_paths: list[str],
    effective_fps: float,
) -> tuple[list[str], dict[str, Any]]:
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
        "selected_frame_timestamps_sec": [frame_timestamps[idx] for idx in ordered_indices],
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
    nearest_idx = min(range(len(frame_timestamps)), key=lambda idx: abs(frame_timestamps[idx] - target_sec))
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
        "selected_frame_timestamps_sec": [frame_timestamps[idx] for idx in ordered_indices],
        "frame_timestamps_sec": frame_timestamps,
    }


def build_yes_no_token_ids(processor: Any) -> tuple[list[int], list[int]]:
    return _candidate_token_ids(processor, ("yes", "YES")), _candidate_token_ids(processor, ("no", "NO"))


def _candidate_token_ids(processor: Any, words: tuple[str, ...]) -> list[int]:
    tokenizer = processor.tokenizer
    token_ids: list[int] = []
    for word in words:
        variants = (word, f" {word}", word.capitalize(), f" {word.capitalize()}")
        for variant in variants:
            ids = tokenizer.encode(variant, add_special_tokens=False)
            if ids:
                token_ids.append(ids[0])
    return sorted(set(token_ids))


def _next_token_logprobs(
    model: Any,
    processor: Any,
    image_paths: list[str],
    prompt: str,
    yes_token_ids: list[int],
    no_token_ids: list[int],
    frame_prefix_texts: list[str] | None = None,
) -> tuple[float, float]:
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


def _generate_text_response(
    model: Any,
    processor: Any,
    image_paths: list[str],
    prompt: str,
    max_new_tokens: int,
    frame_prefix_texts: list[str] | None = None,
) -> str:
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
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return output_text[0].strip()


def _binary_probability(yes_logprob: float, no_logprob: float) -> float:
    max_logprob = max(yes_logprob, no_logprob)
    yes_score = math.exp(yes_logprob - max_logprob)
    no_score = math.exp(no_logprob - max_logprob)
    return yes_score / (yes_score + no_score)


def _build_visibility_prompt(question: dict[str, Any], option_text: str) -> str:
    option_body = extract_option_body(option_text)
    return "\n".join(
        [
            "You are answering about an egocentric surgery video (frames in time order).",
            "Use only visual evidence. Answer with exactly one word: yes or no.",
            "",
            "Task context:",
            str(question.get("question_text", "")).strip(),
            "",
            f"Candidate object / instrument: {option_body}",
            "",
            "Is this object or instrument clearly visible in this video segment?",
            "Answer:",
        ]
    )


def _build_dominant_held_object_prompt(question: dict[str, Any], option_text: str) -> str:
    option_body = extract_option_body(option_text)
    hand_context = _dominant_hand_context(str(question.get("question_text", "")))
    interaction_target = hand_context or "the operator hand described in the question"
    return "\n".join(
        [
            "You are answering about an egocentric surgery video (frames in time order).",
            "Use only visual evidence. Answer with exactly one word: yes or no.",
            "",
            "Task context:",
            str(question.get("question_text", "")).strip(),
            "",
            f"Candidate tool: {option_body}",
            f"Target hand: {interaction_target}",
            "",
            "Choose the tool that the specified hand interacts with for the largest portion of the segment, judged across all frames.",
            (
                f"Across the entire segment, is {option_body} the tool that "
                f"{interaction_target} predominantly interacts with for the largest share of time? "
                "Do not answer yes if the tool only appears briefly or in the background."
            ),
            "Answer:",
        ]
    )


def _build_spatial_point_prompt(question: dict[str, Any], point_output_count: int) -> str:
    question_text = str(question.get("question_text", "")).strip()
    lines = [
        "You are localizing a surgical instrument in an egocentric surgery video.",
        "The provided images are in chronological order.",
        "Use only visible evidence from the images.",
        "Coordinates must be in permille of the image size:",
        "- x=0 is the left edge and x=1000 is the right edge.",
        "- y=0 is the top edge and y=1000 is the bottom edge.",
    ]
    if point_output_count <= 1:
        lines.extend(
            [
                "Return one approximate point on the referenced instrument at the queried moment.",
                "If the instrument spans an area, choose a point near its visible center.",
                "Answer with exactly this format and nothing else: x=<integer>, y=<integer>",
            ]
        )
    else:
        lines.extend(
            [
                f"Return exactly {point_output_count} approximate points on the referenced instrument at the queried moment.",
                "Spread the points across the visible extent of the instrument when possible.",
                "Answer with exactly this format and nothing else: [(x1,y1), (x2,y2), ...]",
            ]
        )
    lines.extend(["", "Question:", question_text])
    return "\n".join(lines)


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
        chunk_prefixes = frame_prefix_texts[start:end] if frame_prefix_texts is not None else None
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
        "raw_answer": best_score["letter"],
        "decode_method": "visibility_yes_logprob_chunk_max_option_min",
        "option_guided_verification": option_scores,
        "frame_chunks": chunk_meta,
    }


def _dominant_hand_context(question_text: str) -> str | None:
    lowered = question_text.lower()
    for surgeon in ("main surgeon", "assistant surgeon"):
        for hand in ("left hand", "right hand"):
            phrase = f"{surgeon} {hand}"
            if phrase in lowered:
                return phrase
    for phrase in (
        "operator's left hand",
        "operator's right hand",
        "operator left hand",
        "operator right hand",
    ):
        if phrase in lowered:
            return phrase
    return None


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
        "raw_answer": best_score["letter"],
        "decode_method": "dominant_held_object_frame_weighted_mean_margin",
        "option_guided_verification": option_scores,
        "frame_chunks": chunk_meta,
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
        "raw_answer": best_score["letter"],
        "decode_method": "object_counting_option_guided_max_yes",
        "option_guided_verification": option_scores,
    }


def _build_object_counting_prompt(question: dict[str, Any], option_text: str) -> str:
    option_body = extract_option_body(option_text)
    return "\n".join(
        [
            "You are answering about an egocentric surgery video (frames in time order).",
            "Use only visual evidence. Answer with exactly one word: yes or no.",
            "",
            "Task context:",
            str(question.get("question_text", "")).strip(),
            "",
            f"Candidate count: {option_body}",
            "",
            f"Is the correct count {option_body}?",
            "Answer:",
        ]
    )


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
            return {"status": "invalid_point_range", "error": f"point_out_of_range({x},{y})"}
        points.append({"x": x, "y": y})
    warning = ""
    if len(points) != expected_count:
        warning = f"expected_{expected_count}_points_got_{len(points)}"
    return {"status": "ok", "points": points, "warning": warning}


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


def run_spatial_point_regression(
    model: Any,
    processor: Any,
    image_paths: list[str],
    question: dict[str, Any],
    max_new_tokens: int,
    point_output_count: int = 1,
    frame_prefix_texts: list[str] | None = None,
) -> dict[str, Any]:
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
        }
    raw_points = [{"x": int(point["x"]), "y": int(point["y"])} for point in parsed_point["points"]]
    pred_x = int(statistics.median(point["x"] for point in raw_points))
    pred_y = int(statistics.median(point["y"] for point in raw_points))
    distances = []
    for option_region in option_regions:
        distance_sq = (
            (pred_x - int(option_region["center_x"])) ** 2
            + (pred_y - int(option_region["center_y"])) ** 2
        )
        distances.append({"letter": option_region["letter"], "distance_sq": distance_sq})
    best_letter = min(distances, key=lambda item: (item["distance_sq"], item["letter"]))["letter"]
    return {
        "answer": best_letter,
        "raw_answer": raw_answer,
        "parse_status": "ok",
        "decode_method": "spatial_point_regression_nearest_region",
    }


def _patch_root() -> Path:
    global _PATCH_DIR
    if _PATCH_DIR is None:
        _PATCH_DIR = Path(tempfile.mkdtemp(prefix="cholectrack_atl6_"))
    return _PATCH_DIR


def preprocess_frame_path(
    src_path: str,
    variant: str,
    *,
    question_id: str,
    frame_idx: int,
) -> str:
    if variant in ("none", "full"):
        return src_path
    with Image.open(src_path) as im:
        rgb = im.convert("RGB")
        w, h = rgb.size
        if variant == "sharpen":
            rgb = ImageEnhance.Sharpness(rgb).enhance(2.0)
        else:
            raise ValueError(f"Unsupported ATL6 variant: {variant}")
        out_dir = _patch_root() / question_id
        out_dir.mkdir(parents=True, exist_ok=True)
        out_p = out_dir / f"f{frame_idx}_{variant}.png"
        rgb.save(out_p, format="PNG")
        return str(out_p.resolve())


def preprocess_paths(paths: list[str], variant: str, question_id: str) -> list[str]:
    if variant in ("none", "full"):
        return list(paths)
    return [
        preprocess_frame_path(path, variant, question_id=question_id, frame_idx=i)
        for i, path in enumerate(paths)
    ]


def effective_sampling_fps_for_atl(question: dict[str, Any], base_fps: float = 0.5) -> float:
    for raw in question.get("video_path", []):
        if not isinstance(raw, str):
            continue
        match = re.search(r"/VID(\d+)/", raw, re.IGNORECASE)
        if match and match.group(1) in ("25", "111"):
            return 1.0
    return base_fps


def extract_raw_action_phrase(question_text: str) -> str:
    text = question_text.strip().rstrip("?").strip()
    patterns = (
        r"approximately\s+at\s+what\s+timestamp\s+does\s+the\s+(.+)",
        r"at\s+what\s+timestamp\s+does\s+the\s+(.+)",
        r"what\s+timestamp\s+does\s+the\s+(.+)",
    )
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(1).strip()
    return text


def normalize_action_phrase(raw: str) -> str:
    phrase = raw.strip()
    phrase = re.sub(r"\s+", " ", phrase)
    phrase = re.sub(r"\bfirst\s+start\b", "starts", phrase, flags=re.IGNORECASE)
    phrase = re.sub(r"\bfirst\s+begins?\b", "starts", phrase, flags=re.IGNORECASE)
    phrase = re.sub(r"\bstart\s+grasp\b", "start grasping", phrase, flags=re.IGNORECASE)
    phrase = re.sub(r"\bgrasp\s+([^u][^\s]*)\s+using\s+", r"grasping \1 with a ", phrase, flags=re.IGNORECASE)
    phrase = re.sub(r"\busing\s+grasper\b", "with a grasper", phrase, flags=re.IGNORECASE)
    phrase = re.sub(r"\busing\s+hook\b", "with a hook", phrase, flags=re.IGNORECASE)
    phrase = re.sub(r"\busing\s+clipper\b", "with a clipper", phrase, flags=re.IGNORECASE)
    phrase = re.sub(r"\busing\s+scissors\b", "with scissors", phrase, flags=re.IGNORECASE)
    phrase = re.sub(r"\busing\s+irrigator\b", "with an irrigator", phrase, flags=re.IGNORECASE)
    phrase = re.sub(r"\bretract\s+", "retracting ", phrase, flags=re.IGNORECASE)
    phrase = re.sub(r"\bdissect\s+", "dissecting ", phrase, flags=re.IGNORECASE)
    phrase = phrase.strip()
    if phrase and phrase[0].islower():
        phrase = phrase[0].upper() + phrase[1:]
    return phrase


def prompt_atl2_earliest_scan(*, action_phrase: str, option_timestamp: float) -> str:
    return (
        "At this timestamp, has the target action started?\n\n"
        f"Target action:\n{action_phrase}\n\n"
        f"Candidate timestamp:\n{option_timestamp} s\n\n"
        "Use the ordered images around this timestamp.\n"
        "Answer Yes only if the target action is visible at or just after this timestamp.\n"
        "Answer No if it has not started yet.\n\n"
        "Answer only Yes or No."
    )


def next_token_yes_no_logprobs(
    model: Any,
    processor: Any,
    image_paths: list[str],
    frame_prefix_texts: list[str] | None,
    prompt: str,
    yes_token_ids: list[int],
    no_token_ids: list[int],
) -> tuple[float, float]:
    return _next_token_logprobs(
        model=model,
        processor=processor,
        image_paths=image_paths,
        prompt=prompt,
        yes_token_ids=yes_token_ids,
        no_token_ids=no_token_ids,
        frame_prefix_texts=frame_prefix_texts,
    )


def binary_yes_probability(yes_logprob: float, no_logprob: float) -> float:
    return _binary_probability(yes_logprob, no_logprob)


def nearest_frame_index(timestamps: list[float], target_sec: float) -> int:
    if not timestamps:
        return 0
    return min(range(len(timestamps)), key=lambda i: abs(timestamps[i] - target_sec))


def unique_ordered_indices(indices: list[int]) -> list[int]:
    seen: set[int] = set()
    out: list[int] = []
    for idx in indices:
        if idx not in seen:
            seen.add(idx)
            out.append(idx)
    return out


def _frame_pack_atl1(
    candidate_ts: float,
    offsets: list[float],
    frame_timestamps: list[float],
    image_paths: list[str],
) -> tuple[list[str], list[str]]:
    raw_indices: list[int] = []
    for offset in offsets:
        target = candidate_ts + offset
        idx = nearest_frame_index(frame_timestamps, target)
        raw_indices.append(idx)
    indices = unique_ordered_indices(raw_indices)
    paths = [image_paths[i] for i in indices]
    prefixes: list[str] = []
    for idx in indices:
        t = frame_timestamps[idx]
        delta = t - candidate_ts
        if abs(delta) < 1e-3:
            label = "CANDIDATE"
        elif delta < 0:
            label = f"BEFORE: ~{t:.2f}s"
        else:
            label = f"AFTER: ~{t:.2f}s"
        prefixes.append(f"{label} (candidate {candidate_ts:.2f}s) - frame ~{t:.2f}s.")
    return paths, prefixes


def _apply_atl6(paths: list[str], atl6: str, qid: str) -> tuple[list[str], dict[str, Any]]:
    meta: dict[str, Any] = {"atl6": atl6}
    if atl6 == "ATL6_A":
        return paths, meta
    if atl6 == "ATL6_D":
        return preprocess_paths(paths, "sharpen", qid), meta
    raise ValueError(f"Unsupported ATL6: {atl6!r}")


def build_frame_prefix_texts_with_phase(num_frames: int, effective_sampling_fps: float) -> list[str]:
    slot = seconds_per_slot(effective_sampling_fps)
    prefixes = []
    for index in range(num_frames):
        start_sec = index * slot
        end_sec = (index + 1) * slot
        if num_frames <= 1:
            phase = "single frame"
        elif index == 0:
            phase = "earliest part"
        elif index == num_frames - 1:
            phase = "latest part"
        else:
            quarter = max(1, (num_frames - 1) // 4)
            if index <= quarter:
                phase = "early part"
            elif index >= (num_frames - 1) - quarter:
                phase = "late part"
            else:
                phase = "middle part"
        prefixes.append(f"Image {index + 1}/{num_frames}: {start_sec:.1f}-{end_sec:.1f}s, {phase}")
    return prefixes


def build_timing_legend_phase(num_frames: int, effective_sampling_fps: float) -> str:
    slot = seconds_per_slot(effective_sampling_fps)
    return "\n".join(
        [
            "Time reference: Frames are in chronological order (earliest to latest).",
            (
                f"Each line uses {effective_sampling_fps:g} FPS stepping "
                f"({slot:.1f} s per image): time window plus a coarse region "
                "(earliest / early / middle / late / latest)."
            ),
            f"You see {num_frames} frame(s); align events with the options.",
        ]
    )


def _build_fps_mapping_mcq_prompt(question: dict[str, Any], num_frames_shown: int) -> str:
    options = question.get("options", [])
    if len(options) != 4:
        raise ValueError(f"Expected 4 options, got {len(options)}.")
    q_text = str(question.get("question_text", "")).strip()
    opt_lines = [normalize_option(str(opt)) for opt in options]
    chronicle = (
        "Images are shown in chronological order. "
        f"Image 1 is the earliest shown frame and image {num_frames_shown} "
        f"is the latest shown frame ({num_frames_shown} frames in this message)."
    )
    ask_line = (
        "Return the earliest image number (a single integer from 1 to "
        f"{num_frames_shown}) where the event in the question first clearly occurs or begins."
    )
    return "\n".join([q_text, "", *opt_lines, "", chronicle, ask_line, "Answer only with that integer and nothing else."])


def _parse_option_temporal_span(option_body: str) -> tuple[float, float] | None:
    range_re = re.compile(r"(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)")
    point_re = re.compile(r"(\d+(?:\.\d+)?)\s*s(?:ec(?:ond)?s?)?\b", re.IGNORECASE)
    text = option_body.strip()
    rng = range_re.search(text)
    if rng:
        return float(rng.group(1)), float(rng.group(2))
    pt = point_re.search(text)
    if pt:
        t = float(pt.group(1))
        return t, t
    return None


def _point_to_interval_distance(t: float, low: float, high: float) -> float:
    if low > high:
        low, high = high, low
    if t < low:
        return low - t
    if t > high:
        return t - high
    return 0.0


def _extract_first_integer(raw: str) -> int | None:
    match = re.search(r"\b(\d+)\b", raw.strip())
    if not match:
        return None
    return int(match.group(1))


def _classify_fps_mapping_image_index(raw: str, num_frames_shown: int) -> tuple[int | None, str, int | None]:
    if num_frames_shown < 1:
        return None, "invalid", _extract_first_integer(raw)
    first = _extract_first_integer(raw)
    if first is None:
        return None, "no_integer", None
    if 1 <= first <= num_frames_shown:
        return first, "ok", first
    return None, "out_of_range", first


def _letter_by_latest_interval_end(options: list[str]) -> str | None:
    per_letter: dict[str, tuple[float, float]] = {}
    for index, opt in enumerate(options):
        letter = chr(ord("A") + index)
        span = _parse_option_temporal_span(extract_option_body(str(opt)))
        if span is None:
            return None
        low, high = span
        if low > high:
            low, high = high, low
        per_letter[letter] = (low, high)
    return max(per_letter, key=lambda letter: (per_letter[letter][1], letter))


def _map_fps_slot_to_mcq_letter(options: list[str], image_index_one_based: int, fps: float) -> str | None:
    slot_sec = 1.0 / float(fps)
    t_rep = (float(image_index_one_based) - 0.5) * slot_sec
    per_letter: dict[str, float] = {}
    for index, opt in enumerate(options):
        letter = chr(ord("A") + index)
        span = _parse_option_temporal_span(extract_option_body(str(opt)))
        if span is None:
            return None
        per_letter[letter] = _point_to_interval_distance(t_rep, span[0], span[1])
    return min(per_letter, key=lambda letter: (per_letter[letter], letter))


def baseline_fps_mapping_answer(
    model: Any,
    processor: Any,
    question: dict[str, Any],
    data_root: Path,
    *,
    frame_time_fps: float,
    max_frames: int = 0,
    max_new_tokens: int = 16,
) -> dict[str, Any]:
    paths = image_paths_for_question(question, data_root)
    if max_frames > 0 and len(paths) > max_frames:
        if max_frames == 1:
            paths = [paths[len(paths) // 2]]
        else:
            last = len(paths) - 1
            paths = [paths[round(i * last / (max_frames - 1))] for i in range(max_frames)]
    prefixes = build_frame_prefix_texts_with_phase(len(paths), frame_time_fps)
    legend = build_timing_legend_phase(len(paths), frame_time_fps)
    body = _build_fps_mapping_mcq_prompt(question, len(paths))
    raw = _generate_text_response(model, processor, paths, f"{legend}\n\n{body}", max_new_tokens, prefixes)
    img_idx, _status, _first_int = _classify_fps_mapping_image_index(raw, len(paths))
    options = list(question.get("options", []))
    answer = ""
    if img_idx is not None:
        answer = _map_fps_slot_to_mcq_letter(options, img_idx, frame_time_fps) or ""
    if not answer:
        answer = _letter_by_latest_interval_end(options) or ""
    return {
        "answer": answer,
        "raw_answer": raw,
        "decode_method": "baseline_fps_mapping",
    }


def run_atl2(
    model: Any,
    processor: Any,
    question: dict[str, Any],
    data_root: Path,
    *,
    action_phrase: str,
    fps: float,
    baseline_letter: str,
    yes_threshold: float,
    margin_fb: float = 0.10,
    atl6: str = "ATL6_A",
) -> dict[str, Any]:
    qtext = str(question.get("question_text", ""))
    if not _EARLIEST_Q_RE.search(qtext):
        return {"answer": baseline_letter, "method": "ATL2", "fallback": "question_keywords"}
    options = list(question.get("options", []))
    paths = image_paths_for_question(question, data_root)
    fts = frame_timestamps_for_question(len(paths), fps, question_text=qtext, options=options)
    ts_list = extract_option_point_timestamps_sec(options)
    ranked: list[tuple[float, str, float]] = []
    yes_ids, no_ids = build_yes_no_token_ids(processor)
    qid = str(question.get("id", ""))
    for i, opt in enumerate(options):
        letter = _LETTERS[i]
        ct = ts_list[i]
        offsets = [-1.0, 0.0, 1.0]
        p0, pr = _frame_pack_atl1(float(ct), offsets, fts, paths)
        imgs, _ = _apply_atl6(p0, atl6, qid)
        prompt = prompt_atl2_earliest_scan(action_phrase=action_phrase, option_timestamp=float(ct))
        yl, nl = next_token_yes_no_logprobs(
            model,
            processor,
            imgs,
            pr if len(pr) == len(imgs) else None,
            prompt,
            yes_ids,
            no_ids,
        )
        yp = binary_yes_probability(yl, nl)
        ranked.append((float(ct), letter, yp))
    ranked.sort(key=lambda x: x[0])
    passing = [(t, letter, p) for t, letter, p in ranked if p >= yes_threshold]
    if not passing:
        return {"answer": baseline_letter, "method": "ATL2", "fallback": "threshold", "yes_threshold": yes_threshold}
    earliest = passing[0]
    if len(passing) > 1:
        second = passing[1]
        if earliest[2] - second[2] < margin_fb:
            return {"answer": baseline_letter, "method": "ATL2", "fallback": "earliest_margin", "yes_threshold": yes_threshold}
    return {"answer": earliest[1], "method": "ATL2", "fallback": None, "yes_threshold": yes_threshold}


def _device_map_from_string(device_str: str) -> Any:
    if device_str.strip().lower() == "cpu":
        return None
    if device_str.strip().lower().startswith("cuda"):
        import torch
        dev = torch.device(device_str)
        idx = dev.index if dev.index is not None else 0
        return {"": idx}
    return "auto"


def _resolve_torch_dtype(dtype_str: str) -> Any:
    import torch
    s = (dtype_str or "auto").strip().lower()
    if s in ("auto", "none", ""):
        return "auto"
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "half": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
        "float": torch.float32,
    }
    if s not in mapping:
        raise ValueError(f"Unknown dtype: {dtype_str!r}")
    return mapping[s]


def _load_model_and_processor(
    model_path: Path,
    device: str,
    dtype: str,
    allow_remote_model: bool,
) -> tuple[Any, Any]:
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor
    device_map = _device_map_from_string(device)
    model = AutoModelForImageTextToText.from_pretrained(
        str(model_path),
        torch_dtype=_resolve_torch_dtype(dtype),
        device_map=device_map,
        local_files_only=not allow_remote_model,
    )
    processor = AutoProcessor.from_pretrained(
        str(model_path),
        local_files_only=not allow_remote_model,
    )
    if device_map is None:
        model = model.to(torch.device(device))
    return model, processor


def _effective_sampling_fps(question: dict[str, Any]) -> float:
    for raw in question.get("video_path", []):
        if not isinstance(raw, str):
            continue
        match = re.search(r"/VID(\d+)/", raw, re.IGNORECASE)
        if match and match.group(1) in {"25", "111"}:
            return 1.0
    return DEFAULT_FPS


def _prefixes_for_effective_frames(
    num_frames: int,
    effective_fps: float,
    cleaned_question_text: str,
    options: list[str],
    window_meta: dict[str, Any],
) -> list[str]:
    selected_timestamps = window_meta.get("selected_frame_timestamps_sec")
    if isinstance(selected_timestamps, list) and len(selected_timestamps) == num_frames:
        return [
            f"Image {idx + 1}/{num_frames} - approximately at {float(timestamp_sec):.1f} s in the clip."
            for idx, timestamp_sec in enumerate(selected_timestamps)
        ]
    timestamps = window_meta.get("frame_timestamps_sec")
    indices = window_meta.get("effective_frame_indices")
    if isinstance(timestamps, list) and isinstance(indices, list):
        prefixes = []
        for out_idx, src_idx in enumerate(indices):
            if isinstance(src_idx, int) and 0 <= src_idx < len(timestamps):
                prefixes.append(
                    f"Image {out_idx + 1}/{len(indices)} - approximately at {float(timestamps[src_idx]):.1f} s in the clip."
                )
            else:
                prefixes.append(f"Image {out_idx + 1}/{len(indices)}.")
        return prefixes
    prefixes, _ = build_frame_prefix_texts(
        num_frames,
        effective_fps,
        question_text=cleaned_question_text,
        options=options,
    )
    return prefixes


def _sampled_prefixes(prefixes: list[str], sampled_offsets: list[int]) -> list[str]:
    return [prefixes[idx] for idx in sampled_offsets] if sampled_offsets else []


def _prepare_effective_frames(question: dict[str, Any], data_root: Path) -> tuple[str, list[str], float, dict[str, Any], list[str], list[str]]:
    original_paths = image_paths_for_question(question, data_root)
    eff_fps = _effective_sampling_fps(question)
    qtext = str(question.get("question_text", ""))
    options = list(question.get("options", []))
    cleaned, frames, window_meta = resolve_question_window(qtext, options, original_paths, eff_fps)
    prefixes = _prefixes_for_effective_frames(len(frames), eff_fps, cleaned, options, window_meta)
    return cleaned, options, eff_fps, window_meta, frames, prefixes


def _run_direct_strategy(
    model: Any,
    processor: Any,
    frames: list[str],
    cleaned: str,
    options: list[str],
    effective_fps: float,
    max_frames: int,
    max_new_tokens: int,
    prefixes: list[str],
    context_text: str | None = None,
) -> str:
    active_sample, _ = uniform_sample_frames(frames, max_frames)
    sampled_offsets = list(active_sample.get("sampled_offsets", []))
    active_prefixes = _sampled_prefixes(prefixes, sampled_offsets)
    body = (
        build_surgical_context_mcq_prompt(cleaned, options, context_text=context_text)
        if context_text is not None
        else build_direct_mcq_prompt(cleaned, options)
    )
    prompt = f"{build_timing_legend(len(frames), effective_fps)}\n\n{body}"
    raw = run_direct_mcq(
        model=model,
        processor=processor,
        image_paths=list(active_sample["image_paths"]),
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        frame_prefix_texts=active_prefixes,
    )
    letter, _ = parse_letter(raw)
    return letter or ""


def _submission_answer_by_question_id(predictions: list[dict[str, Any]]) -> dict[str, str]:
    answers: dict[str, str] = {}
    for prediction in predictions:
        question_id = str(prediction.get("question_id", "")).strip()
        answer = str(prediction.get("answer", "")).strip()
        if question_id and answer:
            answers[question_id] = answer
    return answers


def _submission_answer_by_id(predictions: list[dict[str, Any]]) -> dict[int, str]:
    answers: dict[int, str] = {}
    for prediction in predictions:
        item_id = prediction.get("id")
        answer = str(prediction.get("answer", "")).strip()
        if isinstance(item_id, int) and answer:
            answers[item_id] = answer
    return answers


def _merge_submission_answers(submission_path: Path, predictions: list[dict[str, Any]]) -> int:
    answers_by_question_id = _submission_answer_by_question_id(predictions)
    answers_by_id = _submission_answer_by_id(predictions)
    submission_rows = _load_json_list(submission_path)
    num_updated = 0
    for row in submission_rows:
        question_id = str(row.get("question_id", "")).strip()
        item_id = row.get("id")
        existing_answer = str(row.get("answer", "")).strip()
        new_answer = answers_by_question_id.get(question_id, "")
        if not new_answer and isinstance(item_id, int):
            new_answer = answers_by_id.get(item_id, "")
        if existing_answer or not new_answer:
            continue
        row["answer"] = new_answer
        num_updated += 1
    with submission_path.open("w", encoding="utf-8") as file_obj:
        json.dump(submission_rows, file_obj, ensure_ascii=False, indent=4)
        file_obj.write("\n")
    return num_updated


def _predict_atl2(question: dict[str, Any], model: Any, processor: Any, data_root: Path) -> str:
    fps = effective_sampling_fps_for_atl(question, DEFAULT_FPS)
    base = baseline_fps_mapping_answer(
        model,
        processor,
        question,
        data_root,
        frame_time_fps=fps,
        max_frames=0,
        max_new_tokens=16,
    )
    baseline = base.get("answer") or ""
    raw_act = extract_raw_action_phrase(str(question.get("question_text", "")))
    action = normalize_action_phrase(raw_act)
    pack = run_atl2(
        model,
        processor,
        question,
        data_root,
        action_phrase=action,
        fps=fps,
        baseline_letter=baseline,
        yes_threshold=ATL2_YES_THRESHOLD,
        margin_fb=ATL2_MARGIN_FB,
        atl6="ATL6_A",
    )
    return pack.get("answer") or baseline


def _predict_q_overlay(question: dict[str, Any], model: Any, processor: Any, data_root: Path) -> str:
    cleaned, options, eff_fps, _wm, frames, prefixes = _prepare_effective_frames(question, data_root)
    return _run_direct_strategy(
        model,
        processor,
        frames,
        cleaned,
        options,
        eff_fps,
        Q3_Q4_MAX_FRAMES,
        Q3_Q4_MAX_NEW_TOKENS,
        prefixes,
        context_text="",
    )


def _predict_base(
    question: dict[str, Any],
    model: Any,
    processor: Any,
    yes_ids: list[int],
    no_ids: list[int],
    data_root: Path,
) -> str:
    cleaned, options, eff_fps, window_meta, frames, prefixes = _prepare_effective_frames(question, data_root)
    qt = str(question.get("question_type", "")).strip()
    work = dict(question)
    work["question_text"] = cleaned
    if qt == ATL_QT:
        frames2, meta = select_option_timepoint_neighbor_frames(cleaned, options, frames, eff_fps)
        prefixes2 = prefixes
        if meta.get("applied"):
            wm = dict(window_meta)
            wm["effective_frame_indices"] = meta["selected_frame_indices"]
            wm["frame_timestamps_sec"] = meta["frame_timestamps_sec"]
            wm["selected_frame_timestamps_sec"] = meta["selected_frame_timestamps_sec"]
            prefixes2 = _prefixes_for_effective_frames(len(frames2), eff_fps, cleaned, options, wm)
        return _run_direct_strategy(
            model,
            processor,
            frames2,
            cleaned,
            options,
            eff_fps,
            BASE_MAX_FRAMES,
            BASE_MAX_NEW_TOKENS,
            prefixes2,
        )
    if qt == DHOI_QT:
        return run_dominant_held_object_logprob(
            model=model,
            processor=processor,
            image_paths=frames,
            question=work,
            yes_token_ids=yes_ids,
            no_token_ids=no_ids,
            frame_prefix_texts=prefixes,
            max_frames_per_call=BASE_MAX_FRAMES,
        ).get("answer") or ""
    if qt in (Q3_QT, Q4_QT):
        return _run_direct_strategy(
            model,
            processor,
            frames,
            cleaned,
            options,
            eff_fps,
            BASE_MAX_FRAMES,
            BASE_MAX_NEW_TOKENS,
            prefixes,
            context_text="",
        )
    if qt == COUNT_QT:
        active_sample, _ = uniform_sample_frames(frames, BASE_MAX_FRAMES)
        sampled_offsets = list(active_sample.get("sampled_offsets", []))
        dec = run_object_counting_logprob(
            model,
            processor,
            list(active_sample["image_paths"]),
            work,
            yes_ids,
            no_ids,
            _sampled_prefixes(prefixes, sampled_offsets),
        )
        return dec.get("answer") or ""
    if qt == NOT_VISIBLE_QT:
        return run_visibility_not_visible_logprob(
            model=model,
            processor=processor,
            image_paths=frames,
            question=work,
            yes_token_ids=yes_ids,
            no_token_ids=no_ids,
            frame_prefix_texts=prefixes,
            max_frames_per_call=BASE_MAX_FRAMES,
        ).get("answer") or ""
    if qt == SPATIAL_QT:
        frames2, meta = select_question_timepoint_neighborhood_frames(
            cleaned,
            options,
            frames,
            eff_fps,
            radius=SPATIAL_RADIUS,
        )
        prefixes2 = prefixes
        if meta.get("applied"):
            wm = dict(window_meta)
            wm["effective_frame_indices"] = meta["selected_frame_indices"]
            wm["frame_timestamps_sec"] = meta["frame_timestamps_sec"]
            wm["selected_frame_timestamps_sec"] = meta["selected_frame_timestamps_sec"]
            prefixes2 = _prefixes_for_effective_frames(len(frames2), eff_fps, cleaned, options, wm)
        active_sample, _ = uniform_sample_frames(frames2, BASE_MAX_FRAMES)
        sampled_offsets = list(active_sample.get("sampled_offsets", []))
        dec = run_spatial_point_regression(
            model=model,
            processor=processor,
            image_paths=list(active_sample["image_paths"]),
            question=work,
            max_new_tokens=BASE_MAX_NEW_TOKENS,
            point_output_count=SPATIAL_POINT_COUNT,
            frame_prefix_texts=_sampled_prefixes(prefixes2, sampled_offsets),
        )
        return dec.get("answer") or ""
    raise ValueError(f"Unsupported CholecTrack20 question_type: {qt}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Readable single-file CholecTrack20 submission runner.")
    parser.add_argument("--dataset-json", type=Path, default=DEFAULT_DATASET_JSON)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--submission-json", type=Path, default=DEFAULT_SUBMISSION_JSON)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--dtype", type=str, default="auto")
    parser.add_argument("--allow-remote-model", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    rows = [
        row
        for row in _load_json_list(args.dataset_json)
        if str(row.get("dataset", "")).strip() == DATASET
    ]
    rows = sorted(rows, key=lambda row: row.get("id") if isinstance(row.get("id"), int) else -1)
    if args.limit is not None:
        rows = rows[: max(0, args.limit)]

    model, processor = _load_model_and_processor(
        args.model_path.resolve(),
        args.device,
        args.dtype,
        args.allow_remote_model,
    )
    yes_ids, no_ids = build_yes_no_token_ids(processor)

    import torch

    num_filled = 0
    for idx, question in enumerate(rows, start=1):
        qt = str(question.get("question_type", "")).strip()
        with torch.inference_mode():
            if qt == ATL_QT:
                answer = _predict_atl2(question, model, processor, args.data_root.resolve())
            elif qt == Q3_QT:
                answer = _predict_q_overlay(question, model, processor, args.data_root.resolve())
            elif qt == Q4_QT:
                answer = _predict_q_overlay(question, model, processor, args.data_root.resolve())
            else:
                answer = _predict_base(
                    question,
                    model,
                    processor,
                    yes_ids,
                    no_ids,
                    args.data_root.resolve(),
                )
        row = {
            "id": question.get("id"),
            "question_id": str(question.get("question_id", "")),
            "dataset": DATASET,
            "answer": answer,
        }
        num_filled += _merge_submission_answers(args.submission_json.resolve(), [row])
        print(f"[{idx}/{len(rows)}] {question.get('id')} {qt} -> {answer}", flush=True)

    print(f"Filled {num_filled} empty answer(s) in {args.submission_json.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
