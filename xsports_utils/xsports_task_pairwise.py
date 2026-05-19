#!/usr/bin/env python3
"""Task-specific pairwise tournament for XSports sequence/direction/special-action."""

from __future__ import annotations

import itertools
import sys
from pathlib import Path
from typing import Any

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from direct_decode import _build_ab_token_ids, _next_choice_logprobs
from egocross.parsing import extract_option_body, normalize_option
from frame_sampling import default_experiment_config, sample_frame_pack
from run import _image_paths

LETTERS = ("A", "B", "C", "D")

SPECIAL_DEFS = """Action definitions:
- Jump: brief airborne motion, usually takeoff and landing; no need to cross an obstacle.
- Vault: crossing over an obstacle, ledge, rail, wall, or barrier.
- Climb: sustained upward movement with contact/support, not a single hop.
- Fly: prolonged airborne or gliding phase.
- Spin: yaw rotation around the vertical axis; not just a fast turn.
- Flip: head-over-heels pitch rotation.
- Roll: rotation around the forward axis; horizon tilts/rolls."""


def question_type_from_id(question_id: str) -> str:
    if "_q" not in question_id:
        return "unknown"
    _, _, suffix = question_id.partition("_q")
    _, sep, qtype = suffix.partition("_")
    return qtype if sep else "unknown"


def body(option: str) -> str:
    return extract_option_body(normalize_option(option)).strip()


def build_prompt(question: dict[str, Any], a: str, b: str, task: str) -> str:
    if task == "action-sequence-identification":
        task_lines = [
            "Which candidate sequence better matches the temporal order in the video?",
            "Do not answer by listing actions that appear somewhere in the clip.",
            "The order matters: judge early, middle, and late parts separately.",
            "A candidate is correct only if all stages match in order.",
        ]
    elif task == "direction-prediction":
        task_lines = [
            "Which candidate movement direction is better supported by the video?",
            "Directions are mutually exclusive; do not choose Forward by default.",
            "Choose Forward only when the trajectory remains mostly straight.",
            "Curve left/right means sustained curved movement, not a single instantaneous turn.",
            "Left then right requires a temporal direction change.",
        ]
    elif task == "special-action-identification":
        task_lines = [
            "Which candidate special action is better supported by the video segment?",
            "Use visual evidence, not option popularity.",
            "For Spin, require clear yaw rotation; fast turning alone is not enough.",
            "For Vault, look for crossing an obstacle or barrier.",
            "For Climb, look for sustained upward contact/support.",
            SPECIAL_DEFS,
        ]
    else:
        task_lines = ["Which candidate better answers the question?"]

    return "\n".join(
        [
            *task_lines,
            "",
            "Original question:",
            str(question.get("question_text", "")).strip(),
            "",
            f"Candidate A: {a}",
            f"Candidate B: {b}",
            "",
            "Answer with exactly one letter: A or B.",
            "Answer:",
        ]
    )


def score_pair(
    model: Any,
    processor: Any,
    image_paths: list[str],
    choice_a_ids: list[int],
    choice_b_ids: list[int],
    question: dict[str, Any],
    bodies: list[str],
    task: str,
    i: int,
    j: int,
) -> dict[str, Any]:
    import torch

    with torch.inference_mode():
        logp_a1, logp_b1 = _next_choice_logprobs(
            model,
            processor,
            image_paths,
            build_prompt(question, bodies[i], bodies[j], task),
            choice_a_ids,
            choice_b_ids,
        )
        logp_a2, logp_b2 = _next_choice_logprobs(
            model,
            processor,
            image_paths,
            build_prompt(question, bodies[j], bodies[i], task),
            choice_a_ids,
            choice_b_ids,
        )
    margin_i = ((logp_a1 - logp_b1) + (logp_b2 - logp_a2)) / 2.0
    return {
        "letter_i": LETTERS[i],
        "letter_j": LETTERS[j],
        "option_i": bodies[i],
        "option_j": bodies[j],
        "margin_i_over_j": margin_i,
        "runs": [
            {"A": LETTERS[i], "B": LETTERS[j], "logp_A": logp_a1, "logp_B": logp_b1},
            {"A": LETTERS[j], "B": LETTERS[i], "logp_A": logp_a2, "logp_B": logp_b2},
        ],
    }


def predict_one(
    model: Any,
    processor: Any,
    question: dict[str, Any],
    data_root: Path,
    task: str,
    choice_a_ids: list[int],
    choice_b_ids: list[int],
) -> dict[str, Any]:
    cfg = default_experiment_config()
    raw_paths = _image_paths(question, data_root)
    image_paths, sampling = sample_frame_pack(raw_paths, max_frames=cfg.max_frames)
    bodies = [body(str(opt)) for opt in question.get("options", [])]
    scores = [0.0, 0.0, 0.0, 0.0]
    pairs = []
    for i, j in itertools.combinations(range(4), 2):
        detail = score_pair(
            model,
            processor,
            image_paths,
            choice_a_ids,
            choice_b_ids,
            question,
            bodies,
            task,
            i,
            j,
        )
        margin = float(detail["margin_i_over_j"])
        scores[i] += margin
        scores[j] -= margin
        pairs.append(detail)
    chosen = max(range(4), key=lambda idx: scores[idx])
    return {
        "id": question.get("id"),
        "question_id": question.get("question_id", ""),
        "dataset": question.get("dataset", ""),
        "original_num_frames": len(raw_paths),
        "num_frames": len(image_paths),
        "frame_sampling": sampling,
        "decode_method": f"pairwise_{task}",
        "answer": LETTERS[chosen],
        "raw_answer": f"{LETTERS[chosen]} (pairwise_score={scores[chosen]:.6f})",
        "parse_status": "ok",
        "output_format_status": "pairwise_ab_logprob",
        "pairwise_task": {
            "task": task,
            "option_bodies": {LETTERS[idx]: bodies[idx] for idx in range(4)},
            "scores": {LETTERS[idx]: scores[idx] for idx in range(4)},
            "pairs": pairs,
        },
    }




