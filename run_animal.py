#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any

from parsing import extract_option_body
from parsing import normalize_option
from parsing import parse_letter


# ---------------------------------------------------------------------------
# Dataset / path defaults
# ---------------------------------------------------------------------------

_ANIMAL_DATASET = "EgoPet"
_ANIMAL_IDENTIFICATION_QT = "animal identification"
_INTERACTION_IDENTIFICATION_QT = "interaction identification"
_INTERACTION_TEMPORAL_QT = "interaction temporal localization"
_TEMPORAL_SUBSET_LABEL = "temporal localization"

_SCRIPT_DIR = Path(__file__).parent
_DEFAULT_DATASET_JSON = _SCRIPT_DIR / "testset" / "egocross_testbed_imgs.json"
_DEFAULT_DATA_ROOT = _SCRIPT_DIR / "testset"
_DEFAULT_SUBMISSION_JSON = _SCRIPT_DIR / "submission.json"

_DEFAULT_EFFECTIVE_SAMPLING_FPS = 0.5
_FINE_GRAINED_OBJECTS_PROMPT_LABEL = "Fine-grained objects"
_PIPELINE_MAX_FRAMES = 0
_VIDEO_SAMPLE_FPS = 1.0
_FPS_MAPPING_LABEL = "fps-mapping"

_RANGE_RE = re.compile(r"(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)")
_POINT_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*s(?:ec(?:ond)?s?)?\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Prompting helpers
# ---------------------------------------------------------------------------

_FINE_GRAINED_OBJECTS_INSTRUCTION_LINES = (
    "Focus on the object the animal is directly interacting with.",
    "",
    "When choosing the object category, use the most visually specific "
    "evidence:",
    (
        "- shape: thin, string-like, flat, round, bulky, container-like, "
        "etc."
    ),
    (
        "- function: food, toy, tool, surface, obstacle, another animal, "
        "etc."
    ),
    (
        "- material only when the option clearly refers to a material or a "
        "broad object made of that material."
    ),
    "",
    (
        "Do not choose a broad material category if a more specific visible "
        "object shape or object type better matches the interaction target."
    ),
    "Ignore background objects that are not directly involved.",
)


def _fine_grained_objects_instruction_block() -> str:
    """Returns the fine-grained object-selection preamble."""

    return "\n".join(_FINE_GRAINED_OBJECTS_INSTRUCTION_LINES)


_INTERACTION_IDENTIFICATION_HINT_LINES_DEFAULT = (
    "Identify the main target of the animal's action.",
    (
        "An object counts as the answer only if the animal is actively "
        "engaging with it, such as touching, approaching, chasing, eating, "
        "sniffing, or playing with it."
    ),
    "Do not choose objects that only appear in the background.",
)

_INTERACTION_IDENTIFICATION_HINT_LINES_NEGATIVE = (
    "Choose the object the animal is actively engaging with.",
    (
        "Do not choose a large background object unless the animal directly "
        "interacts with it."
    ),
    (
        "Do not choose a nearby object unless the animal touches, chases, "
        "eats, sniffs, watches closely, or plays with it."
    ),
)


def _interaction_identification_hint_lines(hint_style: str) -> tuple[str, ...]:
    """II instruction lines for ``default`` or ``negative`` style."""

    if hint_style == "default":
        return _INTERACTION_IDENTIFICATION_HINT_LINES_DEFAULT
    if hint_style == "negative":
        return _INTERACTION_IDENTIFICATION_HINT_LINES_NEGATIVE
    raise ValueError(
        f"Unknown interaction_identification hint_style: {hint_style!r}",
    )


def _temporal_phase_phrase(frame_index: int, num_frames: int) -> str:
    """Coarse clip region label for phase-style captions."""

    if num_frames <= 1:
        return "single frame"
    last = num_frames - 1
    if frame_index == 0:
        return "earliest part"
    if frame_index == last:
        return "latest part"
    quarter = max(1, last // 4)
    if frame_index <= quarter:
        return "early part"
    if frame_index >= last - quarter:
        return "late part"
    return "middle part"


def _seconds_per_slot(effective_sampling_fps: float) -> float:
    """Seconds represented by each frame slot at the given effective FPS."""

    if effective_sampling_fps <= 0:
        raise ValueError("effective_sampling_fps must be positive.")
    return 1.0 / effective_sampling_fps


def _build_timing_legend(
    num_frames: int,
    effective_sampling_fps: float,
) -> str:
    """Legend for simple (dash) time-labeled frames."""

    slot = _seconds_per_slot(effective_sampling_fps)
    return "\n".join(
        [
            (
                "Time reference: The images below are in chronological order "
                "(earliest to latest)."
            ),
            (
                f"Assume they were sampled at {effective_sampling_fps:g} FPS "
                f"along the clip: image 1 spans about 0-{slot:.1f} s, image 2 "
                f"about {slot:.1f}-{2 * slot:.1f} s, and so on ({slot:.1f} s "
                "per step)."
            ),
            f"You are given {num_frames} image(s); use these spans to answer.",
        ],
    )


def _build_frame_prefix_texts(
    num_frames: int,
    effective_sampling_fps: float,
) -> list[str]:
    """Per-frame prefixes for simple time-label style."""

    slot = _seconds_per_slot(effective_sampling_fps)
    prefixes: list[str] = []
    for index in range(num_frames):
        start_sec = index * slot
        end_sec = (index + 1) * slot
        prefixes.append(
            (
                f"Image {index + 1}/{num_frames} — approximately "
                f"{start_sec:.1f}-{end_sec:.1f} s in the clip."
            )
        )
    return prefixes


def _build_timing_legend_phase(
    num_frames: int,
    effective_sampling_fps: float,
) -> str:
    """Legend for phase-style per-frame labels."""

    slot = _seconds_per_slot(effective_sampling_fps)
    return "\n".join(
        [
            (
                "Time reference: Frames are in chronological order (earliest "
                "to latest)."
            ),
            (
                f"Each line uses {effective_sampling_fps:g} FPS stepping "
                f"({slot:.1f} s per image): time window plus a coarse region "
                "(earliest / early / middle / late / latest)."
            ),
            f"You see {num_frames} frame(s); align events with the options.",
        ],
    )


def _build_frame_prefix_texts_with_phase(
    num_frames: int,
    effective_sampling_fps: float,
) -> list[str]:
    """Per-frame ``Image i/N: a-bs, <phase>`` labels."""

    slot = _seconds_per_slot(effective_sampling_fps)
    prefixes: list[str] = []
    for index in range(num_frames):
        start_sec = index * slot
        end_sec = (index + 1) * slot
        phase = _temporal_phase_phrase(index, num_frames)
        prefixes.append(
            (
                f"Image {index + 1}/{num_frames}: "
                f"{start_sec:.1f}-{end_sec:.1f}s, {phase}"
            )
        )
    return prefixes


def _build_interaction_identification_mcq_prompt(
    question: dict[str, Any],
    hint_style: str = "default",
    fine_grained_objects: bool = False,
) -> str:
    """MCQ block for interaction-identification tasks."""

    options = question.get("options", [])
    if len(options) != 4:
        raise ValueError(f"Expected 4 options, got {len(options)}.")
    question_text = str(question.get("question_text", "")).strip()
    option_block = "\n".join(
        normalize_option(str(option)) for option in options
    )
    head_chunks: list[str] = []
    if fine_grained_objects:
        head_chunks.append(_fine_grained_objects_instruction_block())
    head_chunks.append(
        "\n".join(_interaction_identification_hint_lines(hint_style)),
    )
    head_chunks.append("Answer only A, B, C, or D.")
    mcq_head = "\n\n".join(head_chunks)
    body = "\n".join(
        [
            mcq_head,
            "",
            "Question:",
            question_text,
            "",
            "Options:",
            option_block,
        ],
    )
    return body


# ---------------------------------------------------------------------------
# FPS mapping helpers
# ---------------------------------------------------------------------------


def _build_fps_mapping_mcq_prompt(
    question: dict[str, Any],
    num_frames_shown: int,
    fine_grained_objects: bool = False,
    prompt_index_upper: str = "default",
    original_num_frames: int | None = None,
) -> str:
    """MCQ stem plus instruction to answer one image index."""

    options = question.get("options", [])
    if len(options) != 4:
        raise ValueError(f"Expected 4 options, got {len(options)}.")
    if prompt_index_upper == "strict_original":
        if original_num_frames is None or original_num_frames < 1:
            raise ValueError(
                "strict_original requires original_num_frames >= 1.",
            )
        upper_ask = original_num_frames
    else:
        upper_ask = num_frames_shown
    q_text = str(question.get("question_text", "")).strip()
    opt_lines = [normalize_option(str(opt)) for opt in options]
    chronicle = (
        f"Images are shown in chronological order. Image 1 is the earliest "
        f"shown frame and image {num_frames_shown} is the latest shown frame "
        f"({num_frames_shown} frames in this message)."
    )
    ask_line = (
        "Return the earliest image number (a single integer from 1 to "
        f"{upper_ask}) where the event in the question first clearly "
        "occurs or begins."
    )
    tail_lines = ["", chronicle, ask_line]
    if prompt_index_upper == "strict_shown":
        tail_lines.append(
            "Hard requirement: output only an integer from 1 to "
            f"{num_frames_shown} inclusive; do not output any other number.",
        )
    elif prompt_index_upper == "strict_original" and original_num_frames:
        tail_lines.append(
            "Hard requirement: output only an integer from 1 to "
            f"{original_num_frames} inclusive; do not output any other "
            "number.",
        )
    tail_lines.append("Answer only with that integer and nothing else.")
    blocks: list[str] = []
    if fine_grained_objects:
        blocks.append(_fine_grained_objects_instruction_block())
    blocks.append(q_text)
    blocks.append("")
    blocks.extend(opt_lines)
    blocks.extend(tail_lines)
    return "\n".join(blocks)


def _parse_option_temporal_span(
    option_body: str,
) -> tuple[float, float] | None:
    """Parse ``a-b`` range or single ``Xs`` timestamp from option text."""

    text = option_body.strip()
    rng = _RANGE_RE.search(text)
    if rng:
        return float(rng.group(1)), float(rng.group(2))
    pt = _POINT_RE.search(text)
    if pt:
        t = float(pt.group(1))
        return t, t
    return None


def _point_to_interval_distance(t: float, low: float, high: float) -> float:
    """Distance from ``t`` to ``[low, high]``."""

    if low > high:
        low, high = high, low
    if t < low:
        return low - t
    if t > high:
        return t - high
    return 0.0


def _extract_first_integer(raw: str) -> int | None:
    """First decimal integer token in ``raw``."""

    match = re.search(r"\b(\d+)\b", raw.strip())
    if not match:
        return None
    return int(match.group(1))


def _classify_fps_mapping_image_index(
    raw: str,
    num_frames_shown: int,
) -> tuple[int | None, str, int | None]:
    """Parse first integer vs ``1..num_frames_shown``."""

    if num_frames_shown < 1:
        return None, "invalid", _extract_first_integer(raw)
    first = _extract_first_integer(raw)
    if first is None:
        return None, "no_integer", None
    if 1 <= first <= num_frames_shown:
        return first, "ok", first
    return None, "out_of_range", first


def _letter_by_latest_interval_end(
    options: list[str],
) -> tuple[str | None, dict[str, Any]]:
    """Pick the letter whose interval ends latest."""

    if len(options) != 4:
        raise ValueError(f"Expected 4 options, got {len(options)}.")
    per_letter: dict[str, dict[str, float]] = {}
    for index, opt in enumerate(options):
        letter = chr(ord("A") + index)
        body = extract_option_body(str(opt))
        span = _parse_option_temporal_span(body)
        if span is None:
            debug: dict[str, Any] = {
                "per_letter": per_letter,
                "error": f"unparsed_interval_{letter}",
            }
            return None, debug
        low, high = span
        if low > high:
            low, high = high, low
        per_letter[letter] = {"low_sec": low, "high_sec": high}
    best = max(
        per_letter,
        key=lambda letter_key: (
            per_letter[letter_key]["high_sec"],
            letter_key,
        ),
    )
    debug = {
        "per_letter": per_letter,
        "chosen_letter": best,
        "rule": "latest_interval_end",
    }
    return best, debug


def _map_fps_slot_to_mcq_letter(
    options: list[str],
    image_index_one_based: int,
    frame_time_fps: float,
) -> tuple[str | None, dict[str, Any]]:
    """Pick A–D by closest time match to the image slot midpoint."""

    if len(options) != 4:
        raise ValueError(f"Expected 4 options, got {len(options)}.")
    slot_sec = 1.0 / float(frame_time_fps)
    t_rep = (float(image_index_one_based) - 0.5) * slot_sec
    per_letter: dict[str, dict[str, Any]] = {}
    for index, opt in enumerate(options):
        letter = chr(ord("A") + index)
        body = extract_option_body(str(opt))
        span = _parse_option_temporal_span(body)
        if span is None:
            debug_bad: dict[str, Any] = {
                "t_rep_sec": t_rep,
                "slot_sec": slot_sec,
                "image_index": image_index_one_based,
                "per_letter": per_letter,
                "error": f"unparsed_interval_{letter}",
            }
            return None, debug_bad
        low, high = span
        dist = _point_to_interval_distance(t_rep, low, high)
        per_letter[letter] = {
            "low_sec": low,
            "high_sec": high,
            "distance": dist,
        }
    best = min(
        per_letter,
        key=lambda letter_key: (
            per_letter[letter_key]["distance"],
            letter_key,
        ),
    )
    debug = {
        "t_rep_sec": t_rep,
        "slot_sec": slot_sec,
        "image_index": image_index_one_based,
        "per_letter": per_letter,
        "chosen_letter": best,
    }
    return best, debug


# ---------------------------------------------------------------------------
# I/O and sampling
# ---------------------------------------------------------------------------


def _load_json_list(json_path: Path) -> list[dict[str, Any]]:
    """Load a JSON list of objects."""

    with json_path.open("r", encoding="utf-8") as file_obj:
        payload = json.load(file_obj)
    if not isinstance(payload, list):
        raise ValueError(f"{json_path} must contain a JSON list.")
    for index, item in enumerate(payload):
        if not isinstance(item, dict):
            raise ValueError(f"{json_path} item {index} must be an object.")
    return payload


def _filter_animal_questions(
    questions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """EgoPet rows only."""

    return [
        item
        for item in questions
        if str(item.get("dataset", "")).strip() == _ANIMAL_DATASET
    ]


def _resolve_image_path(raw_path: str, data_root: Path) -> str:
    """Resolve testbed relative paths against ``data_root``."""

    image_path = Path(raw_path)
    if image_path.is_absolute() and image_path.is_file():
        return str(image_path)

    relative_path = raw_path.lstrip("/")
    candidate = data_root / relative_path
    if candidate.is_file():
        return str(candidate)

    path_parts = Path(relative_path).parts
    if path_parts and path_parts[0] == data_root.name:
        return str(data_root / Path(*path_parts[1:]))
    return str(candidate)


def _image_paths(question: dict[str, Any], data_root: Path) -> list[str]:
    """Resolved frame paths for one question."""

    paths: list[str] = []
    for raw_path in question.get("video_path", []):
        if isinstance(raw_path, str):
            paths.append(_resolve_image_path(raw_path, data_root))
    return paths


def _uniform_sample(items: list[str], max_items: int) -> list[str]:
    """Uniform temporal subsampling; non-positive ``max_items`` keeps all."""

    if max_items <= 0 or len(items) <= max_items:
        return items
    if max_items == 1:
        return [items[len(items) // 2]]
    last_index = len(items) - 1
    return [
        items[round(index * last_index / (max_items - 1))]
        for index in range(max_items)
    ]


def _as_file_uri(path: str) -> str:
    """Local path as ``file://`` URI for video content."""

    return Path(path).resolve().as_uri()


def _build_plain_mcq_prompt(question: dict[str, Any]) -> str:
    """Plain MCQ (animal identification / video path)."""

    options = question.get("options", [])
    if len(options) != 4:
        raise ValueError(f"Expected 4 options, got {len(options)}.")
    prompt_lines = [
        str(question.get("question_text", "")).strip(),
        "",
    ]
    prompt_lines.extend(normalize_option(str(option)) for option in options)
    prompt_lines.extend(
        [
            "",
            "Answer with only the single letter: A, B, C, or D.",
        ],
    )
    return "\n".join(prompt_lines)


def _resolve_frame_time_labeling(
    use_frame_time_labels: bool,
    label_style: str,
    num_frames: int,
    frame_time_fps: float,
) -> tuple[list[str] | None, str | None]:
    """Per-image prefixes and legend for time-labeled runs."""

    if not use_frame_time_labels:
        return None, None
    if label_style == "phase":
        return (
            _build_frame_prefix_texts_with_phase(num_frames, frame_time_fps),
            _build_timing_legend_phase(num_frames, frame_time_fps),
        )
    if label_style == "simple":
        return (
            _build_frame_prefix_texts(num_frames, frame_time_fps),
            _build_timing_legend(num_frames, frame_time_fps),
        )
    raise ValueError(f"Unknown label_style: {label_style!r}")


# ---------------------------------------------------------------------------
# Model inference
# ---------------------------------------------------------------------------


def _run_direct_inference_multi_image(
    model: Any,
    processor: Any,
    selected_image_paths: list[str],
    prompt: str,
    max_new_tokens: int,
    frame_prefix_texts: list[str] | None,
) -> str:
    """Direct generation with one chat message and multiple images."""

    if frame_prefix_texts is not None and len(frame_prefix_texts) != len(
        selected_image_paths,
    ):
        raise ValueError(
            "frame_prefix_texts must match selected_image_paths length.",
        )

    query_content: list[dict[str, Any]] = []
    for idx, image_path in enumerate(selected_image_paths):
        if frame_prefix_texts is not None:
            query_content.append(
                {"type": "text", "text": frame_prefix_texts[idx]},
            )
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
        out_ids[len(in_ids) :]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return output_text[0].strip()


def _run_direct_inference_video(
    model: Any,
    processor: Any,
    selected_frame_paths: list[str],
    prompt: str,
    max_new_tokens: int,
    sample_fps: float,
    video_max_pixels: int | None,
) -> str:
    """Single ``type: video`` message from an ordered frame list."""

    video_content: dict[str, Any] = {
        "type": "video",
        "video": [_as_file_uri(p) for p in selected_frame_paths],
        "sample_fps": sample_fps,
    }
    if video_max_pixels is not None:
        video_content["max_pixels"] = video_max_pixels

    messages = [
        {
            "role": "user",
            "content": [
                video_content,
                {"type": "text", "text": prompt},
            ],
        }
    ]

    try:
        from qwen_vl_utils import process_vision_info
    except ImportError as err:
        raise ImportError(
            "Video input requires qwen-vl-utils. "
            "Install with `pip install qwen-vl-utils`.",
        ) from err

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


def _assemble_image_prediction_row(
    question: dict[str, Any],
    answer: str,
    raw_answer: str,
    parse_status: str,
    original_image_paths: list[str],
    selected_image_paths: list[str],
    max_frames: int,
    frame_time_fps: float,
    use_frame_time_labels: bool,
    frame_time_label_style: str,
    output_format_status: str,
    interaction_mcq_hint_style: str | None,
    fine_grained_objects: bool,
    subset_label: str | None,
    fps_mapping_debug: dict[str, Any] | None = None,
    fps_mapping_label: str | None = None,
) -> dict[str, Any]:
    """Builds one prediction row for image tasks."""

    row: dict[str, Any] = {
        "id": question.get("id"),
        "question_id": question.get("question_id", ""),
        "dataset": question.get("dataset", ""),
        "question_type": question.get("question_type", ""),
        "answer": answer,
        "raw_answer": raw_answer,
        "parse_status": parse_status,
        "decode_method": "direct_mcq",
        "output_format_status": output_format_status,
        "option_guided_verification": None,
        "baseline": "animal_direct_no_tricks",
        "input_media_type": "image",
        "frame_time_fps": frame_time_fps if use_frame_time_labels else None,
        "frame_time_labels": use_frame_time_labels,
        "frame_time_label_style": frame_time_label_style,
        "original_num_frames": len(original_image_paths),
        "num_frames": len(selected_image_paths),
        "max_frames": max_frames,
    }
    if interaction_mcq_hint_style is not None:
        row["interaction_mcq_hint_style"] = interaction_mcq_hint_style
    row["fine_grained_objects"] = fine_grained_objects
    if fine_grained_objects:
        row["fine_grained_objects_label"] = _FINE_GRAINED_OBJECTS_PROMPT_LABEL
    row["cfg"] = False
    if subset_label is not None:
        row["subset_label"] = subset_label
    if fps_mapping_debug is not None:
        row["fps_mapping"] = fps_mapping_debug
    if fps_mapping_label is not None:
        row["fps_mapping_label"] = fps_mapping_label
    return row


def _run_interaction_identification_image(
    model: Any,
    processor: Any,
    question: dict[str, Any],
    data_root: Path,
    max_frames: int,
    max_new_tokens: int,
) -> dict[str, Any]:
    """Multi-image II with simple time labels and negative hints."""

    frame_time_fps = _DEFAULT_EFFECTIVE_SAMPLING_FPS
    use_labels = True
    style = "simple"
    original_image_paths = _image_paths(question, data_root)
    selected_image_paths = _uniform_sample(original_image_paths, max_frames)
    frame_prefix_texts, mcq_legend = _resolve_frame_time_labeling(
        use_labels,
        style,
        len(selected_image_paths),
        frame_time_fps,
    )
    mcq_body = _build_interaction_identification_mcq_prompt(
        question,
        hint_style="negative",
        fine_grained_objects=False,
    )
    if use_labels and mcq_legend:
        full_prompt = f"{mcq_legend}\n\n{mcq_body}"
    else:
        full_prompt = mcq_body
    raw_answer = _run_direct_inference_multi_image(
        model=model,
        processor=processor,
        selected_image_paths=selected_image_paths,
        prompt=full_prompt,
        max_new_tokens=max_new_tokens,
        frame_prefix_texts=frame_prefix_texts,
    )
    letter, parse_status = parse_letter(raw_answer)
    row_style = "off" if not use_labels else style
    return _assemble_image_prediction_row(
        question=question,
        answer=letter or "",
        raw_answer=raw_answer,
        parse_status=parse_status,
        original_image_paths=original_image_paths,
        selected_image_paths=selected_image_paths,
        max_frames=max_frames,
        frame_time_fps=frame_time_fps,
        use_frame_time_labels=use_labels,
        frame_time_label_style=row_style,
        output_format_status="mcq_generation",
        interaction_mcq_hint_style="negative",
        fine_grained_objects=False,
        subset_label=_INTERACTION_IDENTIFICATION_QT,
    )


def _run_temporal_fps_mapping_image(
    model: Any,
    processor: Any,
    question: dict[str, Any],
    data_root: Path,
    max_frames: int,
    max_new_tokens: int,
) -> dict[str, Any]:
    """Phase labels + fps-mapping + OOB latest-interval."""

    frame_time_fps = _DEFAULT_EFFECTIVE_SAMPLING_FPS
    use_labels = True
    style = "phase"
    oob_latest = True
    prompt_upper = "default"

    original_image_paths = _image_paths(question, data_root)
    selected_image_paths = _uniform_sample(original_image_paths, max_frames)
    frame_prefix_texts, mcq_legend = _resolve_frame_time_labeling(
        use_labels,
        style,
        len(selected_image_paths),
        frame_time_fps,
    )
    num_frames_local = len(selected_image_paths)
    num_orig = len(original_image_paths)
    fps_body = _build_fps_mapping_mcq_prompt(
        question,
        num_frames_local,
        fine_grained_objects=False,
        prompt_index_upper=prompt_upper,
        original_num_frames=num_orig,
    )
    if use_labels and mcq_legend:
        full_prompt = f"{mcq_legend}\n\n{fps_body}"
    else:
        full_prompt = fps_body
    raw_answer = _run_direct_inference_multi_image(
        model=model,
        processor=processor,
        selected_image_paths=selected_image_paths,
        prompt=full_prompt,
        max_new_tokens=max_new_tokens,
        frame_prefix_texts=frame_prefix_texts,
    )
    img_idx, idx_status, first_int = _classify_fps_mapping_image_index(
        raw_answer,
        num_frames_local,
    )
    options_list = question.get("options", [])
    fps_debug: dict[str, Any] = {
        "raw_answer": raw_answer,
        "image_index": img_idx,
        "first_integer_in_output": first_int,
        "index_parse_status": idx_status,
        "frame_time_fps": frame_time_fps,
        "original_num_frames": num_orig,
        "num_frames_shown": num_frames_local,
        "prompt_index_upper": prompt_upper,
        "oob_pick_latest_enabled": oob_latest,
    }
    answer = ""
    parse_status = "invalid"
    if img_idx is not None:
        letter, map_debug = _map_fps_slot_to_mcq_letter(
            options_list,
            img_idx,
            frame_time_fps,
        )
        fps_debug.update(map_debug)
        if letter is not None:
            answer = letter
            parse_status = "ok"
    elif oob_latest:
        letter, latest_debug = _letter_by_latest_interval_end(options_list)
        fps_debug.update(latest_debug)
        fps_debug["oob_fallback"] = "latest_interval_end"
        if letter is not None:
            answer = letter
            parse_status = "ok"
    row_style = "off" if not use_labels else style
    return _assemble_image_prediction_row(
        question=question,
        answer=answer or "",
        raw_answer=raw_answer,
        parse_status=parse_status,
        original_image_paths=original_image_paths,
        selected_image_paths=selected_image_paths,
        max_frames=max_frames,
        frame_time_fps=frame_time_fps,
        use_frame_time_labels=use_labels,
        frame_time_label_style=row_style,
        output_format_status="fps_mapping",
        interaction_mcq_hint_style=None,
        fine_grained_objects=False,
        subset_label=_TEMPORAL_SUBSET_LABEL,
        fps_mapping_debug=fps_debug,
        fps_mapping_label=_FPS_MAPPING_LABEL,
    )


def _run_animal_identification_video(
    model: Any,
    processor: Any,
    question: dict[str, Any],
    data_root: Path,
    max_frames: int,
    max_new_tokens: int,
) -> dict[str, Any]:
    """Plain MCQ on frames as one video tensor."""

    original_image_paths = _image_paths(question, data_root)
    selected_image_paths = _uniform_sample(original_image_paths, max_frames)
    prompt = _build_plain_mcq_prompt(question)
    raw_answer = _run_direct_inference_video(
        model=model,
        processor=processor,
        selected_frame_paths=selected_image_paths,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        sample_fps=_VIDEO_SAMPLE_FPS,
        video_max_pixels=None,
    )
    letter, parse_status = parse_letter(raw_answer)
    return {
        "id": question.get("id"),
        "question_id": question.get("question_id", ""),
        "dataset": question.get("dataset", ""),
        "question_type": question.get("question_type", ""),
        "answer": letter or "",
        "raw_answer": raw_answer,
        "parse_status": parse_status,
        "decode_method": "direct_mcq",
        "baseline": "animal_direct_no_tricks",
        "input_media_type": "video",
        "sample_fps": _VIDEO_SAMPLE_FPS,
        "video_max_pixels": None,
        "original_num_frames": len(original_image_paths),
        "num_frames": len(selected_image_paths),
        "max_frames": max_frames,
        "subset_label": _ANIMAL_IDENTIFICATION_QT,
    }


# ---------------------------------------------------------------------------
# CLI, eval, main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parses command-line arguments.

    Returns:
        Parsed namespace with model path, I/O paths, and runtime options.
    """

    parser = argparse.ArgumentParser(
        description="Run full EgoPet animal benchmark (183 questions).",
    )
    parser.add_argument(
        "model_path",
        type=Path,
        help="Model or checkpoint path.",
    )
    parser.add_argument(
        "--dataset-json",
        type=Path,
        default=_DEFAULT_DATASET_JSON,
        help="EgoCross testbed JSON.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=_DEFAULT_DATA_ROOT,
        help="Root used to resolve media paths.",
    )
    parser.add_argument(
        "--submission-json",
        type=Path,
        default=_DEFAULT_SUBMISSION_JSON,
        help="Submission JSON to update.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=16,
        help="Generation cap per question.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Smoke limit.")
    parser.add_argument("--dtype", default="auto", help="Model dtype.")
    parser.add_argument("--device-map", default="auto", help="Device map.")
    parser.add_argument(
        "--allow-remote-model",
        action="store_true",
        help="Allow remote model download/cache.",
    )
    return parser.parse_args()


def _submission_answer_by_question_id(
    predictions: list[dict[str, Any]],
) -> dict[str, str]:
    """Builds a non-empty answer map keyed by question id.

    Args:
        predictions: Prediction rows from this runner.

    Returns:
        Question id to answer letter.
    """

    answers: dict[str, str] = {}
    for prediction in predictions:
        question_id = str(prediction.get("question_id", "")).strip()
        answer = str(prediction.get("answer", "")).strip()
        if question_id and answer:
            answers[question_id] = answer
    return answers


def _submission_answer_by_id(
    predictions: list[dict[str, Any]],
) -> dict[int, str]:
    """Builds a non-empty answer map keyed by numeric id.

    Args:
        predictions: Prediction rows from this runner.

    Returns:
        Numeric id to answer letter.
    """

    answers: dict[int, str] = {}
    for prediction in predictions:
        item_id = prediction.get("id")
        answer = str(prediction.get("answer", "")).strip()
        if isinstance(item_id, int) and answer:
            answers[item_id] = answer
    return answers


def _merge_submission_answers(
    submission_path: Path,
    predictions: list[dict[str, Any]],
) -> int:
    """Fills empty answers in the submission file.

    Args:
        submission_path: JSON file following the official submission format.
        predictions: Prediction rows from this runner.

    Returns:
        Number of answers newly filled.
    """

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


def _question_bucket(question: dict[str, Any]) -> str:
    """Returns an internal dispatch label for one EgoPet row.

    Args:
        question: Testbed row.

    Returns:
        One of ``animal_identification``, ``interaction_identification``,
        ``interaction_temporal``.

    Raises:
        ValueError: If ``question_type`` is not one of the three EgoPet tasks.
    """

    qt = str(question.get("question_type", "")).strip()
    if qt == _ANIMAL_IDENTIFICATION_QT:
        return "animal_identification"
    if qt == _INTERACTION_IDENTIFICATION_QT:
        return "interaction_identification"
    if qt == _INTERACTION_TEMPORAL_QT:
        return "interaction_temporal"
    raise ValueError(f"Unexpected EgoPet question_type: {qt!r}")


def main() -> None:
    """Loads the model once and fills empty EgoPet answers."""

    args = parse_args()

    all_rows = _load_json_list(args.dataset_json)
    ego_pet = _filter_animal_questions(all_rows)
    ego_pet = sorted(
        ego_pet,
        key=lambda q: q.get("id") if isinstance(q.get("id"), int) else -1,
    )
    if args.limit is not None:
        ego_pet = ego_pet[: args.limit]

    # pylint: disable=import-outside-toplevel
    import torch
    from transformers import AutoModelForImageTextToText
    from transformers import AutoProcessor

    print(f"Loading model from {args.model_path}...")
    model = AutoModelForImageTextToText.from_pretrained(
        str(args.model_path),
        dtype=args.dtype,
        device_map=args.device_map,
        local_files_only=not args.allow_remote_model,
    )
    processor = AutoProcessor.from_pretrained(
        str(args.model_path),
        local_files_only=not args.allow_remote_model,
    )
    processor.video_processor.size = {
        "longest_edge": 16384 * 32 * 32 * 2,
        "shortest_edge": 256 * 32 * 32 * 2,
    }

    print(f"Running {len(ego_pet)} EgoPet questions.")
    start_time = time.perf_counter()
    num_filled = 0

    for index, question in enumerate(ego_pet, start=1):
        item_id = question.get("id")
        bucket = _question_bucket(question)
        print(f"Question {item_id} ({index}/{len(ego_pet)}) [{bucket}]...")
        with torch.inference_mode():
            if bucket == "animal_identification":
                row = _run_animal_identification_video(
                    model=model,
                    processor=processor,
                    question=question,
                    data_root=args.data_root,
                    max_frames=_PIPELINE_MAX_FRAMES,
                    max_new_tokens=args.max_new_tokens,
                )
            elif bucket == "interaction_identification":
                row = _run_interaction_identification_image(
                    model=model,
                    processor=processor,
                    question=question,
                    data_root=args.data_root,
                    max_frames=_PIPELINE_MAX_FRAMES,
                    max_new_tokens=args.max_new_tokens,
                )
            else:
                row = _run_temporal_fps_mapping_image(
                    model=model,
                    processor=processor,
                    question=question,
                    data_root=args.data_root,
                    max_frames=_PIPELINE_MAX_FRAMES,
                    max_new_tokens=args.max_new_tokens,
                )

        num_filled += _merge_submission_answers(args.submission_json, [row])
        print(f"  answer: {row.get('answer') or 'INVALID'}")

    elapsed_sec = round(time.perf_counter() - start_time, 3)
    print(f"Filled {num_filled} empty answer(s) in {args.submission_json}.")
    print(f"Elapsed: {elapsed_sec} sec.")


if __name__ == "__main__":
    main()
