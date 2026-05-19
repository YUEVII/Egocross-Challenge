"""Industry inference loop: YAML config, outputs under ``output/``."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

_PKG_ROOT = Path(__file__).resolve().parents[1]
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

# pylint: disable=wrong-import-position
from direct_decode import build_ab_token_ids
from direct_decode import build_mcq_token_ids
from direct_decode import build_yes_no_token_ids
from direct_decode import run_dominant_held_object_logprob
from direct_decode import run_next_interaction_logprob
from direct_decode import run_next_interaction_pairwise
from direct_decode import run_next_interaction_video_mcq
from direct_decode import run_not_visible_any_frame_logprob
from direct_decode import run_object_counting_labeled_bbox_regression
from direct_decode import run_not_visible_pairwise
from direct_decode import run_not_visible_mcq_logprob
from direct_decode import run_object_counting_logprob
from direct_decode import run_spatial_point_regression
from direct_decode import run_visibility_not_visible_logprob
from industry_infer.config import RunConfig
from industry_infer.config import config_to_json_dict
from industry_infer.config import load_run_config
from industry_infer.frame_time import build_frame_prefix_texts
from industry_infer.frame_time import build_timing_legend
from industry_infer.inference_local import build_direct_mcq_prompt
from industry_infer.inference_local import run_direct_mcq
from industry_infer.parsing import parse_letter
from industry_infer.paths import image_paths_for_question
from industry_infer.question_text import resolve_question_window
from industry_infer.question_text import select_option_timepoint_neighbor_frames
from industry_infer.question_text import select_question_timepoint_neighborhood_frames
from industry_infer.sampling import uniform_sample_frames
from industry_infer.sampling_fps import resolve_effective_sampling_fps


def _device_map_from_string(device_str: str) -> Any:
    if device_str.strip().lower() == "cpu":
        return None
    if device_str.strip().lower().startswith("cuda"):
        import torch

        device = torch.device(device_str)
        idx = device.index if device.index is not None else 0
        return {"": idx}
    return "auto"


def _resolve_torch_dtype(dtype_str: str) -> Any:
    """Argument for ``AutoModelForImageTextToText.from_pretrained(torch_dtype=...)``."""

    import torch

    dtype = (dtype_str or "auto").strip().lower()
    if dtype in ("auto", "none", ""):
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
    if dtype not in mapping:
        raise ValueError(f"Unknown inference.dtype in YAML: {dtype_str!r}")
    return mapping[dtype]


def _load_json_list(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a JSON list.")
    return data


def _filter_questions(
    rows: list[dict[str, Any]],
    datasets: set[str],
    enabled: dict[str, bool],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        if str(row.get("dataset", "")).strip() not in datasets:
            continue
        question_type = str(row.get("question_type", "")).strip()
        if not enabled.get(question_type, False):
            continue
        out.append(row)
    return out


def _prefixes_for_effective_frames(
    num_frames: int,
    effective_fps: float,
    *,
    cleaned_question_text: str,
    options: list[str],
    window_meta: dict[str, Any],
) -> tuple[list[str], dict[str, Any]]:
    selected_timestamps = window_meta.get("selected_frame_timestamps_sec")
    if isinstance(selected_timestamps, list) and len(selected_timestamps) == num_frames:
        prefixes = [
            (
                f"Image {idx + 1}/{num_frames} — approximately at "
                f"{float(timestamp_sec):.1f} s in the clip."
            )
            for idx, timestamp_sec in enumerate(selected_timestamps)
        ]
        return prefixes, {
            "source": "selected_frame_timestamps",
            "effective_sampling_fps": effective_fps,
            "num_frames": len(prefixes),
        }

    timestamps = window_meta.get("frame_timestamps_sec")
    indices = window_meta.get("effective_frame_indices")
    if isinstance(timestamps, list) and isinstance(indices, list):
        prefixes = []
        for out_idx, src_idx in enumerate(indices):
            if isinstance(src_idx, int) and 0 <= src_idx < len(timestamps):
                timestamp_sec = timestamps[src_idx]
                prefixes.append(
                    (
                        f"Image {out_idx + 1}/{len(indices)} — approximately at "
                        f"{float(timestamp_sec):.1f} s in the clip."
                    )
                )
            else:
                prefixes.append(f"Image {out_idx + 1}/{len(indices)}.")
        return prefixes, {
            "source": "time_window_timestamps",
            "effective_sampling_fps": effective_fps,
            "num_frames": len(prefixes),
        }

    prefixes, meta = build_frame_prefix_texts(
        num_frames,
        effective_fps,
        question_text=cleaned_question_text,
        options=options,
    )
    meta["source"] = "sequential_effective_frames"
    return prefixes, meta


def _select_tail_frames(
    frames: list[str],
    prefixes: list[str],
    *,
    max_frames: int,
) -> tuple[list[str], list[str], dict[str, Any]]:
    """Take the final ``max_frames`` chronologically ordered frames."""

    if max_frames <= 0 or len(frames) <= max_frames:
        return list(frames), list(prefixes), {
            "applied": False,
            "selection_mode": "tail_frames",
            "reason": "all_effective_frames",
            "selected_frame_offsets": list(range(len(frames))),
        }

    start = len(frames) - max_frames
    return list(frames[start:]), list(prefixes[start:]), {
        "applied": True,
        "selection_mode": "tail_frames",
        "reason": "last_n_frames",
        "selected_frame_offsets": list(range(start, len(frames))),
        "num_selected_frames": max_frames,
    }


def run_inference(cfg: RunConfig, limit: int | None = None) -> None:
    datasets = set(cfg.datasets)
    questions = _filter_questions(
        _load_json_list(cfg.dataset_json),
        datasets,
        cfg.question_type_enabled,
    )
    if limit is not None:
        questions = questions[: max(0, limit)]


    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    pred_path = cfg.output_dir / "predictions.json"
    conf_path = cfg.output_dir / "config.json"
    prog_path = cfg.output_dir / "progress.json"

    payload_cfg = config_to_json_dict(cfg)
    payload_cfg["limit"] = limit
    conf_path.write_text(
        json.dumps(payload_cfg, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    # pylint: disable=import-outside-toplevel
    import torch
    from transformers import AutoModelForImageTextToText
    from transformers import AutoProcessor

    model_path = cfg.model_path_sft_local if cfg.use_sft else cfg.model_path_base
    device_map = _device_map_from_string(cfg.device)
    torch_dtype = _resolve_torch_dtype(cfg.dtype)

    print(f"Loading model from {model_path}...", flush=True)
    model = AutoModelForImageTextToText.from_pretrained(
        str(model_path),
        torch_dtype=torch_dtype,
        device_map=device_map,
        local_files_only=not cfg.allow_remote_model,
    )
    processor = AutoProcessor.from_pretrained(
        str(model_path),
        local_files_only=not cfg.allow_remote_model,
    )
    if device_map is None:
        model = model.to(torch.device(cfg.device))

    yes_ids, no_ids = build_yes_no_token_ids(processor)
    if not yes_ids or not no_ids:
        raise ValueError("Tokenizer lacks yes/no token candidates.")
    choice_a_ids, choice_b_ids = build_ab_token_ids(processor)
    mcq_token_ids = build_mcq_token_ids(processor)

    predictions: list[dict[str, Any]] = []
    start_time = time.perf_counter()

    for idx, question in enumerate(questions, start=1):
        item_id = question.get("id")
        dataset_name = str(question.get("dataset", "")).strip()
        question_type = str(question.get("question_type", "")).strip()
        print(
            f"Question {item_id} ({idx}/{len(questions)}) [{dataset_name} / {question_type}]...",
            flush=True,
        )

        original_paths = image_paths_for_question(question, cfg.data_root)
        effective_fps, fps_meta = resolve_effective_sampling_fps(question, cfg)

        question_text = str(question.get("question_text", ""))
        options = list(question.get("options", []))
        cleaned, frames, window_meta = resolve_question_window(
            question_text,
            options,
            original_paths,
            effective_fps,
        )
        qt_setting = cfg.question_type_settings.get(
            question_type,
            {"use_custom_strategy": False, "strategy": "direct_mcq"},
        )
        use_custom = bool(qt_setting.get("use_custom_strategy", False))
        strategy = str(qt_setting.get("strategy", "direct_mcq"))

        strategy_frame_selection = None
        if use_custom and strategy == "temporal_option_frame_pair_mcq":
            frames, strategy_frame_selection = select_option_timepoint_neighbor_frames(
                question_text,
                options,
                frames,
                effective_fps,
            )
        elif use_custom and strategy in (
            "question_timepoint_neighborhood_mcq",
            "question_timepoint_point_coord_mcq",
        ):
            frames, strategy_frame_selection = (
                select_question_timepoint_neighborhood_frames(
                    question_text,
                    options,
                    frames,
                    effective_fps,
                    radius=int(qt_setting.get("timepoint_neighbor_radius", 3)),
                )
            )

        if strategy_frame_selection and strategy_frame_selection.get("applied"):
            window_meta = dict(window_meta)
            window_meta["effective_frame_indices"] = strategy_frame_selection[
                "selected_frame_indices"
            ]
            window_meta["effective_num_frames"] = len(frames)
            window_meta["frame_timestamps_sec"] = strategy_frame_selection[
                "frame_timestamps_sec"
            ]
            window_meta["selected_frame_timestamps_sec"] = strategy_frame_selection[
                "selected_frame_timestamps_sec"
            ]
            window_meta["strategy_frame_selection"] = strategy_frame_selection

        prefixes, frame_time_meta = _prefixes_for_effective_frames(
            len(frames),
            effective_fps,
            cleaned_question_text=question_text,
            options=options,
            window_meta=window_meta,
        )
        legend = build_timing_legend(len(frames), effective_fps)

        work = dict(question)
        work["question_text"] = cleaned

        with torch.inference_mode():
            if use_custom and strategy == "visible_logprob_chunk_max_option_min":
                dec = run_visibility_not_visible_logprob(
                    model=model,
                    processor=processor,
                    image_paths=frames,
                    question=work,
                    yes_token_ids=yes_ids,
                    no_token_ids=no_ids,
                    frame_prefix_texts=prefixes,
                    max_frames_per_call=cfg.max_frames,
                )
                letter = dec["answer"]
                raw = dec["raw_answer"]
                parse_status = "ok"
                decode_method = dec.get(
                    "decode_method",
                    "visibility_yes_logprob_chunk_max_option_min",
                )
                extra = {
                    "option_guided_verification": dec["option_guided_verification"],
                    "frame_chunks": dec["frame_chunks"],
                }
            elif use_custom and strategy == "not_visible_mcq_logprob":
                dec = run_not_visible_mcq_logprob(
                    model=model,
                    processor=processor,
                    image_paths=frames,
                    question=work,
                    mcq_token_ids=mcq_token_ids,
                    frame_prefix_texts=prefixes,
                )
                letter = dec["answer"]
                raw = dec["raw_answer"]
                parse_status = "ok"
                decode_method = dec.get(
                    "decode_method",
                    "not_visible_mcq_letter_logprob",
                )
                extra = {
                    "mcq_option_logprobs": dec["mcq_option_logprobs"],
                    "prompt_style": dec["prompt_style"],
                }
            elif use_custom and strategy == "not_visible_any_frame_option_guided":
                dec = run_not_visible_any_frame_logprob(
                    model=model,
                    processor=processor,
                    image_paths=frames,
                    question=work,
                    yes_token_ids=yes_ids,
                    no_token_ids=no_ids,
                    frame_prefix_texts=prefixes,
                )
                letter = dec["answer"]
                raw = dec["raw_answer"]
                parse_status = "ok"
                decode_method = dec.get(
                    "decode_method",
                    "not_visible_any_frame_option_guided_min_yes",
                )
                extra = {
                    "option_guided_verification": dec["option_guided_verification"],
                    "prompt_style": dec["prompt_style"],
                }
            elif use_custom and strategy == "not_visible_pairwise":
                if not choice_a_ids or not choice_b_ids:
                    raise ValueError("Tokenizer lacks A/B token candidates.")
                dec = run_not_visible_pairwise(
                    model=model,
                    processor=processor,
                    image_paths=frames,
                    question=work,
                    choice_a_token_ids=choice_a_ids,
                    choice_b_token_ids=choice_b_ids,
                    frame_prefix_texts=prefixes,
                )
                letter = dec["answer"]
                raw = dec["raw_answer"]
                parse_status = "ok"
                decode_method = dec.get(
                    "decode_method",
                    "not_visible_pairwise_margin_sum",
                )
                extra = {
                    "pairwise_verification": dec["pairwise_verification"],
                    "prompt_style": dec["prompt_style"],
                }
            elif use_custom and strategy == "dominant_held_object_frame_weighted_mean":
                dec = run_dominant_held_object_logprob(
                    model=model,
                    processor=processor,
                    image_paths=frames,
                    question=work,
                    yes_token_ids=yes_ids,
                    no_token_ids=no_ids,
                    frame_prefix_texts=prefixes,
                    max_frames_per_call=cfg.max_frames,
                )
                letter = dec["answer"]
                raw = dec["raw_answer"]
                parse_status = "ok"
                decode_method = dec.get(
                    "decode_method",
                    "dominant_held_object_frame_weighted_mean_margin",
                )
                extra = {
                    "option_guided_verification": dec["option_guided_verification"],
                    "frame_chunks": dec["frame_chunks"],
                }
            elif use_custom and strategy == "next_interaction_tail_option_guided":
                tail_frames, tail_prefixes, tail_meta = _select_tail_frames(
                    frames,
                    prefixes,
                    max_frames=cfg.max_frames,
                )
                dec = run_next_interaction_logprob(
                    model=model,
                    processor=processor,
                    image_paths=tail_frames,
                    question=work,
                    yes_token_ids=yes_ids,
                    no_token_ids=no_ids,
                    frame_prefix_texts=tail_prefixes,
                )
                letter = dec["answer"]
                raw = dec["raw_answer"]
                parse_status = "ok"
                decode_method = dec.get(
                    "decode_method",
                    "next_interaction_option_guided_max_yes",
                )
                extra = {
                    "option_guided_verification": dec["option_guided_verification"],
                    "tail_frame_selection": tail_meta,
                }
            elif use_custom and strategy == "next_interaction_tail_pairwise":
                if not choice_a_ids or not choice_b_ids:
                    raise ValueError("Tokenizer lacks A/B token candidates.")
                tail_frames, tail_prefixes, tail_meta = _select_tail_frames(
                    frames,
                    prefixes,
                    max_frames=cfg.max_frames,
                )
                dec = run_next_interaction_pairwise(
                    model=model,
                    processor=processor,
                    image_paths=tail_frames,
                    question=work,
                    choice_a_token_ids=choice_a_ids,
                    choice_b_token_ids=choice_b_ids,
                    frame_prefix_texts=tail_prefixes,
                )
                letter = dec["answer"]
                raw = dec["raw_answer"]
                parse_status = "ok"
                decode_method = dec.get(
                    "decode_method",
                    "next_interaction_pairwise_margin_sum",
                )
                extra = {
                    "pairwise_verification": dec["pairwise_verification"],
                    "tail_frame_selection": tail_meta,
                }
            elif use_custom and strategy == "next_interaction_tail_video_mcq":
                tail_frames, _, tail_meta = _select_tail_frames(
                    frames,
                    prefixes,
                    max_frames=cfg.max_frames,
                )
                dec = run_next_interaction_video_mcq(
                    model=model,
                    processor=processor,
                    image_paths=tail_frames,
                    question=work,
                    max_new_tokens=cfg.max_new_tokens,
                    sample_fps=float(qt_setting.get("video_sample_fps", 1.0)),
                    video_max_pixels=(
                        int(qt_setting["video_max_pixels"])
                        if qt_setting.get("video_max_pixels") is not None
                        else None
                    ),
                )
                letter = dec.get("answer") or None
                raw = dec["raw_answer"]
                parse_status = dec.get("parse_status", "ok")
                decode_method = dec.get(
                    "decode_method",
                    "next_interaction_tail_video_mcq",
                )
                extra = {
                    "tail_frame_selection": tail_meta,
                    "prompt_style": dec["prompt_style"],
                    "input_media_type": dec["input_media_type"],
                    "sample_fps": dec["sample_fps"],
                    "video_max_pixels": dec["video_max_pixels"],
                }
            elif use_custom and strategy == "object_counting_option_guided":
                active_sample, chunk_meta = uniform_sample_frames(frames, cfg.max_frames)
                sampled_offsets = list(active_sample.get("sampled_offsets", []))
                active_prefixes = (
                    [prefixes[idx] for idx in sampled_offsets] if sampled_offsets else []
                )
                dec = run_object_counting_logprob(
                    model=model,
                    processor=processor,
                    image_paths=list(active_sample["image_paths"]),
                    question=work,
                    yes_token_ids=yes_ids,
                    no_token_ids=no_ids,
                    frame_prefix_texts=active_prefixes,
                )
                letter = dec["answer"]
                raw = dec["raw_answer"]
                parse_status = "ok"
                decode_method = dec.get(
                    "decode_method",
                    "object_counting_option_guided_max_yes",
                )
                extra = {
                    "option_guided_verification": dec["option_guided_verification"],
                    "frame_chunks": chunk_meta,
                    "counting_option_guided_frame_policy": chunk_meta.get(
                        "strategy",
                        "uniform_sample",
                    ),
                }
            elif use_custom and strategy == "object_counting_labeled_bboxes":
                active_sample, chunk_meta = uniform_sample_frames(frames, cfg.max_frames)
                sampled_offsets = list(active_sample.get("sampled_offsets", []))
                active_prefixes = (
                    [prefixes[idx] for idx in sampled_offsets] if sampled_offsets else []
                )
                dec = run_object_counting_labeled_bbox_regression(
                    model=model,
                    processor=processor,
                    image_paths=list(active_sample["image_paths"]),
                    question=work,
                    candidate_labels=list(qt_setting.get("counting_object_labels", [])),
                    max_new_tokens=cfg.max_new_tokens,
                    frame_prefix_texts=active_prefixes,
                )
                letter = dec.get("answer") or None
                raw = dec["raw_answer"]
                parse_status = dec.get("parse_status", "ok")
                decode_method = dec.get(
                    "decode_method",
                    "object_counting_labeled_bbox_unique_labels",
                )
                extra = {
                    "counting_bbox_prediction": dec["counting_bbox_prediction"],
                    "frame_chunks": chunk_meta,
                    "prompt_style": dec["prompt_style"],
                }
            elif use_custom and strategy == "question_timepoint_point_coord_mcq":
                active_sample, chunk_meta = uniform_sample_frames(frames, cfg.max_frames)
                sampled_offsets = list(active_sample.get("sampled_offsets", []))
                active_prefixes = (
                    [prefixes[idx] for idx in sampled_offsets] if sampled_offsets else []
                )
                dec = run_spatial_point_regression(
                    model=model,
                    processor=processor,
                    image_paths=list(active_sample["image_paths"]),
                    question=work,
                    max_new_tokens=cfg.max_new_tokens,
                    point_output_count=int(qt_setting.get("point_output_count", 1)),
                    frame_prefix_texts=active_prefixes,
                )
                letter = dec.get("answer") or None
                raw = dec["raw_answer"]
                parse_status = dec.get("parse_status", "ok")
                decode_method = dec.get(
                    "decode_method",
                    "spatial_point_regression_nearest_region",
                )
                extra = {
                    "frame_chunks": chunk_meta,
                    "spatial_point_prediction": dec["spatial_point_prediction"],
                    "prompt_style": dec["prompt_style"],
                }
            elif (not use_custom) or strategy in (
                "direct_mcq",
                "temporal_option_frame_pair_mcq",
                "question_timepoint_neighborhood_mcq",
            ):
                active_sample, chunk_meta = uniform_sample_frames(frames, cfg.max_frames)
                sampled_offsets = list(active_sample.get("sampled_offsets", []))
                active_prefixes = (
                    [prefixes[idx] for idx in sampled_offsets] if sampled_offsets else []
                )
                body = build_direct_mcq_prompt(cleaned, options)
                prompt = f"{legend}\n\n{body}"
                raw = run_direct_mcq(
                    model=model,
                    processor=processor,
                    image_paths=list(active_sample["image_paths"]),
                    prompt=prompt,
                    max_new_tokens=cfg.max_new_tokens,
                    frame_prefix_texts=active_prefixes,
                )
                letter, parse_status = parse_letter(raw)
                decode_method = "direct_mcq"
                extra = {
                    "frame_chunks": chunk_meta,
                    "direct_mcq_chunk_policy": chunk_meta.get("strategy", "uniform_sample"),
                }
            else:
                raise ValueError(
                    f"Unknown custom strategy for {question_type!r}: {strategy!r}",
                )

        row = {
            "id": item_id,
            "question_id": str(question.get("question_id", "")),
            "dataset": question.get("dataset", ""),
            "question_type": question_type,
            "answer": letter or "",
            "raw_answer": raw,
            "parse_status": parse_status,
            "decode_method": decode_method,
            "original_num_frames": len(original_paths),
            "effective_num_frames": len(frames),
            "effective_sampling_fps": effective_fps,
            "sampling_fps_meta": fps_meta,
            "max_frames_per_call": cfg.max_frames,
            "effective_frames": {
                "indices": window_meta.get("effective_frame_indices", []),
                "range_sec": window_meta.get("effective_range_sec"),
            },
            "frame_time": frame_time_meta,
            "time_window": window_meta,
            "strategy_frame_selection": strategy_frame_selection or {},
            **extra,
        }
        predictions.append(row)
        prog_path.write_text(
            json.dumps(predictions, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"Result {item_id}: pred={letter or ''}", flush=True)

    elapsed = round(time.perf_counter() - start_time, 3)
    pred_path.write_text(
        json.dumps(predictions, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


    print(f"Done. Wrote {pred_path}, {conf_path}", flush=True)


def main(config_path: Path | None = None, limit: int | None = None) -> int:
    package_root = Path(__file__).resolve().parents[1]
    path = config_path or (package_root / "configs" / "base.yaml")
    cfg = load_run_config(path, package_root=package_root)
    run_inference(cfg, limit=limit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
