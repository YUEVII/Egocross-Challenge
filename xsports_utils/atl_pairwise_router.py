#!/usr/bin/env python3
"""ATL adaptive router with VLM pairwise arbitration."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from direct_decode import _build_ab_token_ids, _candidate_times
from direct_decode import _next_choice_logprobs, _select_local_window
from frame_sampling import default_experiment_config, sample_frame_pack
from run import _frame_timestamps, _image_paths, _normalize_answer
from run import _timestamped_image_paths

HARD_ACTIONS = {"right", "curveright", "jump", "walk"}


def _load_model(model_path: Path, dtype: str, device_map: str, local_files_only: bool):
    from transformers import AutoModelForImageTextToText, AutoProcessor

    print(f"Loading model from {model_path}...")
    model = AutoModelForImageTextToText.from_pretrained(
        str(model_path),
        dtype=dtype,
        device_map=device_map,
        local_files_only=local_files_only,
    )
    processor = AutoProcessor.from_pretrained(
        str(model_path),
        local_files_only=local_files_only,
    )
    return model, processor


def _dedupe_panel(paths: list[str], labels: list[str]) -> tuple[list[str], list[str]]:
    seen: set[str] = set()
    out_paths = []
    out_labels = []
    for path, label in zip(paths, labels):
        if path in seen:
            continue
        seen.add(path)
        out_paths.append(path)
        out_labels.append(label)
    return out_paths, out_labels


def _panel_for_times(
    image_paths: list[str],
    frame_times: list[float],
    duration: float,
    times: list[float],
) -> tuple[list[str], list[str]]:
    target_times = [0.0, duration]
    target_times.extend(times)
    paths, labels = _select_local_window(
        image_paths,
        frame_times,
        [min(duration, max(0.0, t)) for t in target_times],
    )
    return _dedupe_panel(paths, labels)


def _candidate_panel(
    image_paths: list[str],
    frame_times: list[float],
    duration: float,
    t_a: float,
    t_b: float,
) -> tuple[list[str], list[str]]:
    # Keep the panel small. Qwen3-VL image tokens get expensive quickly; the
    # router only needs anchors plus before/at/after evidence for each time.
    targets = [0.0, duration]
    for t in (t_a, t_b):
        targets.extend([t - 0.8, t, t + 0.8])
    return _panel_for_times(image_paths, frame_times, duration, targets)


def _arbiter_prompt(question: dict[str, Any], time_a: float, time_b: float) -> str:
    return "\n".join(
        [
            "The target action is described in the original question below.",
            str(question.get("question_text", "")).strip(),
            "",
            "Two candidate start times are shown in the frame panel.",
            "For each candidate, compare frames before and after that time.",
            "",
            f"Candidate A: {time_a:.3f}s",
            f"Candidate B: {time_b:.3f}s",
            "",
            "Choose the candidate closer to the first visible moment when the target action begins.",
            "Do not choose the time when the action is already clearly ongoing.",
            "Do not choose based on option letters.",
            "",
            "Answer with exactly one letter: A or B.",
            "Answer:",
        ]
    )


def _choice_margin(
    model: Any,
    processor: Any,
    image_paths: list[str],
    labels: list[str],
    prompt: str,
    choice_a_ids: list[int],
    choice_b_ids: list[int],
) -> tuple[float, float, float]:
    import torch

    with torch.inference_mode():
        logp_a, logp_b = _next_choice_logprobs(
            model=model,
            processor=processor,
            image_paths=image_paths,
            prompt=prompt,
            choice_a_ids=choice_a_ids,
            choice_b_ids=choice_b_ids,
            frame_prefix_texts=labels,
        )
    return logp_a, logp_b, logp_b - logp_a


def run_pairwise_arbiter(
    model: Any,
    processor: Any,
    question: dict[str, Any],
    image_paths: list[str],
    frame_times: list[float],
    duration: float,
    t_time: float,
    o_time: float,
    choice_a_ids: list[int],
    choice_b_ids: list[int],
) -> dict[str, Any]:
    paths, labels = _candidate_panel(image_paths, frame_times, duration, t_time, o_time)

    # Run 1: A=T, B=O. Positive margin means O.
    logp_a1, logp_b1, margin_o_1 = _choice_margin(
        model,
        processor,
        paths,
        labels,
        _arbiter_prompt(question, t_time, o_time),
        choice_a_ids,
        choice_b_ids,
    )
    # Run 2: A=O, B=T. Positive-for-O is logp(A)-logp(B).
    logp_a2, logp_b2, margin_t_2 = _choice_margin(
        model,
        processor,
        paths,
        labels,
        _arbiter_prompt(question, o_time, t_time),
        choice_a_ids,
        choice_b_ids,
    )
    margin_o_2 = -margin_t_2
    margin_o = (margin_o_1 + margin_o_2) / 2.0
    return {
        "kind": "legacy_arbiter",
        "margin_legacy": margin_o,
        "choose_legacy": margin_o > 0,
        "runs": [
            {"A": "T", "B": "O", "logp_A": logp_a1, "logp_B": logp_b1, "margin_legacy": margin_o_1},
            {"A": "O", "B": "T", "logp_A": logp_a2, "logp_B": logp_b2, "margin_legacy": margin_o_2},
        ],
        "image_paths": paths,
        "frame_prefix_texts": labels,
    }


def should_consider_legacy(r: dict[str, Any], q_trans_60: float, q_old_65: float) -> bool:
    if r["T_answer"] == r["O_answer"]:
        return False
    t_time = r["T_pred_time"]
    o_time = r["O_pred_time"]
    if t_time is None or o_time is None or abs(t_time - o_time) < 1.0:
        return False
    hard = bool(r["hard_action"])
    transition_weak = r["T_margin"] <= q_trans_60
    legacy_confident = r["O_margin"] >= q_old_65
    legacy_earlier = o_time < t_time - 0.75
    legacy_lower_rank = (
        r["T_pred_rank"] is not None
        and r["O_pred_rank"] is not None
        and r["O_pred_rank"] < r["T_pred_rank"]
    )
    return (
        (transition_weak or legacy_confident or hard)
        and (legacy_earlier or legacy_lower_rank or hard)
    )


def _question_runtime_inputs(
    question: dict[str, Any],
    data_root: Path,
    exp_dir: Path,
    duration: float,
) -> tuple[list[str], list[float]]:
    original_paths = _image_paths(question, data_root)
    cfg = default_experiment_config()
    paths, sampling = sample_frame_pack(original_paths, max_frames=cfg.max_frames)
    frame_times = _frame_timestamps(question, sampling, duration_seconds=duration)
    if frame_times is None:
        frame_times = [duration * (idx + 0.5) / max(1, len(paths)) for idx in range(len(paths))]
    stamped = _timestamped_image_paths(
        paths,
        frame_times,
        exp_dir=exp_dir,
        item_id=question.get("id"),
        enabled=True,
    )
    return stamped, frame_times












