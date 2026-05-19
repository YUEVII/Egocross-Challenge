#!/usr/bin/env python3
"""Simplified default Xsports inference runner."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# pylint: disable=wrong-import-position
from egocross.inference import run_inference
from direct_decode import build_yes_no_token_ids
from direct_decode import run_option_guided_verification
from direct_decode import run_transition_guided_verification
from frame_sampling import build_experiment_paths
from frame_sampling import default_experiment_config
from frame_sampling import experiment_config_to_dict
from frame_sampling import sample_frame_pack
from in_context import default_retrieval_config
from in_context import retrieval_config_to_dict
from prompt_aug import build_prompt
from prompt_aug import build_prompt_metrics
from prompt_aug import count_text_tokens
from prompt_aug import extract_answer
from prompt_aug import get_prompt_config
from prompt_aug import prompt_config_to_dict
from shuffle_option import build_shuffle_metrics
from shuffle_option import default_shuffle_config
from shuffle_option import shuffle_config_to_dict

from atl_tricks import build_atl_timestamp_instruction
from atl_tricks import estimate_video_duration_seconds
from atl_tricks import is_action_temporal_localization_question

_VALID_ANSWERS = frozenset({"A", "B", "C", "D"})
_DEFAULT_EXP_ROOT = Path(__file__).resolve().parent / "outputs"
_DEFAULT_DOMAIN = "xsports"
_DEFAULT_DECODE_METHOD = "option_guided"
_DIRECT_PROMPT_CONFIG_NAME = "P0"


def _resolve_model_checkpoint(args: argparse.Namespace) -> Path:
    """HF checkpoint directory: ``--model-path`` overrides ``--model-kind``."""

    if args.model_path is not None:
        return args.model_path.resolve()
    subdir = "base_model" if args.model_kind == "base" else "xsports"
    return (_REPO_ROOT / "ckpts" / subdir).resolve()




def _load_json_list(json_path: Path) -> list[dict[str, Any]]:
    with json_path.open("r", encoding="utf-8") as file_obj:
        payload = json.load(file_obj)
    if not isinstance(payload, list):
        raise ValueError(f"{json_path} must contain a JSON list.")
    for index, item in enumerate(payload):
        if not isinstance(item, dict):
            raise ValueError(f"{json_path} item {index} must be an object.")
    return payload


def _resolve_image_path(raw_path: str, data_root: Path) -> str:
    image_path = Path(raw_path)
    if image_path.is_absolute() and image_path.is_file():
        return str(image_path)

    relative_path = raw_path.lstrip("/")
    candidate = data_root / relative_path
    if candidate.is_file():
        return str(candidate)

    path_parts = Path(relative_path).parts
    if path_parts and path_parts[0] == data_root.name:
        deduped_candidate = data_root / Path(*path_parts[1:])
        if deduped_candidate.is_file():
            return str(deduped_candidate)
        return str(deduped_candidate)

    return str(candidate)


def _image_paths(question: dict[str, Any], data_root: Path) -> list[str]:
    paths = []
    for raw_path in question.get("video_path", []):
        if isinstance(raw_path, str):
            paths.append(_resolve_image_path(raw_path, data_root))
    return paths


def _normalize_answer(raw_answer: Any) -> str | None:
    if raw_answer is None:
        return None
    answer_text = str(raw_answer).strip().upper()
    if answer_text in _VALID_ANSWERS:
        return answer_text
    for char in answer_text:
        if char in _VALID_ANSWERS:
            return char
    return None


def _domain_name(dataset_name: str) -> str:
    normalized = dataset_name.strip().lower().replace("_", "").replace("-", "")
    if normalized == "egopet":
        return "animal"
    if normalized == "enigma":
        return "industry"
    if normalized in {"extramesportfpv", "extremesportfpv"}:
        return "xsports"
    if normalized in {"cholectrack20", "egosurgery"}:
        return "surgery"
    return normalized


def _filter_by_domain(
    items: list[dict[str, Any]],
    domain_name: str,
) -> list[dict[str, Any]]:
    if domain_name == "all":
        return items
    return [
        item
        for item in items
        if _domain_name(str(item.get("dataset", ""))) == domain_name
    ]




def _question_type_from_id(question_id: str) -> str:
    if "_q" not in question_id:
        return "unknown"
    _, _, suffix = question_id.partition("_q")
    _, separator, question_type = suffix.partition("_")
    if not separator or not question_type:
        return "unknown"
    return question_type











def _build_experiment_name(args: argparse.Namespace) -> str:
    experiment_name = f"F0_P0_R0_S0_{args.decode_method.upper()}_XSPORTS"
    if args.question_types:
        question_type_suffix = "_".join(
            question_type.upper() for question_type in sorted(args.question_types)
        )
        experiment_name += "_QT_" + question_type_suffix
    if args.atl_temporal_anchors:
        experiment_name += "_ATLANCHORS"
    if args.atl_duration_seconds is not None:
        duration_label = str(args.atl_duration_seconds).replace(".", "P")
        experiment_name += f"_D{duration_label}"
    if args.atl_frame_timestamps:
        experiment_name += "_FRAMETS"
    if args.decode_method == "transition_guided":
        experiment_name += f"_{args.atl_transition_score.upper()}SCORE"
    return experiment_name


def _atl_prompt_prefix(
    question: dict[str, Any],
    image_paths: list[str],
    *,
    atl_temporal_anchors: bool,
    duration_seconds: float | None,
) -> tuple[str, bool]:
    if (
        atl_temporal_anchors
        and is_action_temporal_localization_question(question)
    ):
        return (
            build_atl_timestamp_instruction(
                question,
                len(image_paths),
                duration_seconds=duration_seconds,
            ),
            True,
        )
    return "", False


def _frame_timestamps(
    question: dict[str, Any],
    sampling_metadata: dict[str, Any],
    *,
    duration_seconds: float | None,
) -> list[float] | None:
    duration = estimate_video_duration_seconds(
        question,
        override_seconds=duration_seconds,
    )
    if duration is None:
        return None
    original_count = int(sampling_metadata.get("original_count", 0))
    selected_indices = sampling_metadata.get("selected_indices", [])
    if original_count <= 0 or not isinstance(selected_indices, list):
        return None
    return [
        duration * (int(index) + 0.5) / original_count
        for index in selected_indices
    ]


def _frame_timestamp_texts(frame_times: list[float] | None) -> list[str] | None:
    if frame_times is None:
        return None
    return [
        f"Frame {index + 1}, timestamp ~= {frame_time:.2f}s."
        for index, frame_time in enumerate(frame_times)
    ]


def _timestamped_image_paths(
    image_paths: list[str],
    frame_times: list[float] | None,
    *,
    exp_dir: Path,
    item_id: Any,
    enabled: bool,
    mirror: bool = False,
) -> list[str]:
    if not enabled or frame_times is None:
        return image_paths
    try:
        # pylint: disable=import-outside-toplevel
        from PIL import Image, ImageDraw, ImageFont, ImageOps
    except ImportError:
        return image_paths

    out_dir = exp_dir / ("mirrored_timestamped_frames" if mirror else "timestamped_frames") / str(item_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamped_paths = []
    for index, (image_path, frame_time) in enumerate(zip(image_paths, frame_times)):
        out_path = out_dir / f"frame_{index + 1:02d}.jpg"
        if not out_path.exists():
            with Image.open(image_path) as image:
                image = image.convert("RGB")
                if mirror:
                    image = ImageOps.mirror(image)
                draw = ImageDraw.Draw(image)
                if index == 0:
                    label = f"START 0.00s | Frame {index + 1}  t={frame_time:.2f}s"
                elif index == len(image_paths) - 1:
                    label = f"END | Frame {index + 1}  t={frame_time:.2f}s"
                else:
                    label = f"Frame {index + 1}  t={frame_time:.2f}s"
                font = ImageFont.load_default()
                bbox = draw.textbbox((0, 0), label, font=font)
                pad = 6
                rect = (
                    0,
                    0,
                    bbox[2] - bbox[0] + pad * 2,
                    bbox[3] - bbox[1] + pad * 2,
                )
                draw.rectangle(rect, fill=(0, 0, 0))
                draw.text((pad, pad), label, fill=(255, 255, 255), font=font)
                image.save(out_path, quality=92)
        stamped_paths.append(str(out_path))
    return stamped_paths


def _build_direct_atl_prompt(question: dict[str, Any], prompt_prefix: str) -> str:
    prompt = build_prompt(question, get_prompt_config(_DIRECT_PROMPT_CONFIG_NAME))
    if is_action_temporal_localization_question(question):
        prompt = "\n".join(
            [
                "For this ATL question, compare all four candidate timestamps jointly.",
                "Choose the option whose timestamp best matches when the action begins.",
                prompt,
            ]
        )
    if prompt_prefix:
        prompt = f"{prompt_prefix}\n\n{prompt}"
    return prompt


def _default_in_context_metadata() -> dict[str, Any]:
    retrieval_config = default_retrieval_config()
    return {
        "experiment": retrieval_config.name,
        "strategy": retrieval_config.strategy,
        "requested_examples": retrieval_config.num_examples,
        "selected_examples": 0,
        "selected_support_ids": [],
    }


def _default_option_shuffle_metadata() -> dict[str, Any]:
    shuffle_config = default_shuffle_config()
    return {
        "experiment": shuffle_config.name,
        "num_orders": shuffle_config.num_orders,
        "aggregation": {},
        "runs": [],
    }


def _run_question(
    model: Any,
    processor: Any,
    question: dict[str, Any],
    data_root: Path,
    *,
    atl_temporal_anchors: bool,
    decode_method: str,
    max_frames: int,
    support_root: Path,
    exp_dir: Path,
    atl_duration_seconds: float | None,
    atl_frame_timestamps: bool,
    atl_transition_score: str,
) -> dict[str, Any]:
    item_id = question.get("id")
    options = question.get("options", [])
    original_image_paths = _image_paths(question, data_root)
    image_paths, sampling_metadata = sample_frame_pack(
        original_image_paths,
        max_frames=max_frames,
    )
    row = {
        "id": item_id,
        "question_id": question.get("question_id", ""),
        "dataset": question.get("dataset", ""),
        "original_num_frames": len(original_image_paths),
        "num_frames": len(image_paths),
        "frame_sampling": sampling_metadata,
        "in_context": _default_in_context_metadata(),
        "option_shuffle": _default_option_shuffle_metadata(),
        "decode_method": decode_method,
        "answer": "",
        "raw_answer": "",
        "parse_status": "invalid",
        "output_format_status": "invalid",
        "prompt_experiment": "P0",
        "prompt_template": "direct",
        "prompt_token_count": 0,
        "output_token_count": 0,
        "total_text_token_count": 0,
        "atl_timeline_applied": False,
        "atl_duration_seconds": estimate_video_duration_seconds(
            question,
            override_seconds=atl_duration_seconds,
        ),
        "atl_frame_timestamps_applied": False,
        "atl_onset_definition": "",
    }

    if not image_paths:
        row["error"] = "missing video_path"
        row["atl_temporal_anchors_applied"] = False
        return row
    if len(options) != 4:
        row["error"] = f"expected 4 options, got {len(options)}"
        row["atl_temporal_anchors_applied"] = False
        return row

    # Import lazily so argument parsing does not initialize heavy ML libraries.
    # pylint: disable=import-outside-toplevel
    import torch

    frame_times = _frame_timestamps(
        question,
        sampling_metadata,
        duration_seconds=atl_duration_seconds,
    )
    frame_prefix_texts = (
        _frame_timestamp_texts(frame_times) if atl_frame_timestamps else None
    )
    if atl_frame_timestamps and is_action_temporal_localization_question(question):
        image_paths = _timestamped_image_paths(
            image_paths,
            frame_times,
            exp_dir=exp_dir,
            item_id=item_id,
            enabled=True,
        )
        row["atl_frame_timestamps_applied"] = True


    prompt_prefix, anchors_applied = _atl_prompt_prefix(
        question,
        image_paths,
        atl_temporal_anchors=atl_temporal_anchors,
        duration_seconds=atl_duration_seconds,
    )
    row["atl_temporal_anchors_applied"] = anchors_applied

    if decode_method in {
        "option_guided",
        "transition_guided",
    }:
        yes_token_ids, no_token_ids = build_yes_no_token_ids(processor)

    if decode_method == "option_guided":
        with torch.inference_mode():
            decode_result = run_option_guided_verification(
                model=model,
                processor=processor,
                image_paths=image_paths,
                question=question,
                yes_token_ids=yes_token_ids,
                no_token_ids=no_token_ids,
                prompt_prefix=prompt_prefix,
                frame_prefix_texts=frame_prefix_texts,
            )
        row["answer"] = decode_result["answer"]
        row["raw_answer"] = decode_result["raw_answer"]
        row["parse_status"] = "ok"
        row["output_format_status"] = "yes_no_logprob"
        row["option_guided_verification"] = decode_result[
            "option_guided_verification"
        ]
        return row

    if decode_method == "transition_guided":
        with torch.inference_mode():
            decode_result = run_transition_guided_verification(
                model=model,
                processor=processor,
                image_paths=image_paths,
                question=question,
                yes_token_ids=yes_token_ids,
                no_token_ids=no_token_ids,
                prompt_prefix=prompt_prefix,
                frame_times=frame_times,
                score_mode=atl_transition_score,
            )
        row["answer"] = decode_result["answer"]
        row["raw_answer"] = decode_result["raw_answer"]
        row["parse_status"] = "ok"
        row["output_format_status"] = "transition_logprob"
        row["transition_guided_verification"] = decode_result[
            "transition_guided_verification"
        ]
        return row

    prompt = _build_direct_atl_prompt(question, prompt_prefix)
    with torch.inference_mode():
        raw_answer = run_inference(
            model=model,
            processor=processor,
            image_paths=image_paths,
            prompt=prompt,
            support_examples=[],
            support_root=str(support_root),
            images_per_example=0,
            max_new_tokens=16,
            frame_prefix_texts=frame_prefix_texts,
        )
    answer, output_format_status = extract_answer(raw_answer)
    row["answer"] = answer or ""
    row["raw_answer"] = raw_answer
    row["parse_status"] = "ok" if answer else "invalid"
    row["output_format_status"] = output_format_status
    row["prompt_token_count"] = count_text_tokens(processor, prompt)
    row["output_token_count"] = count_text_tokens(processor, raw_answer)
    row["total_text_token_count"] = (
        row["prompt_token_count"] + row["output_token_count"]
    )
    return row




