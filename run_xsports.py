#!/usr/bin/env python3
"""Direct XSports inference with a heuristic expert router.

Submission folder contract:
  1. testset/egocross_testbed_imgs.json: original full question JSON
  2. xsports_utils/: local implementation package
  3. run_xsports.py: this script

The script writes every question in submission format at submission.json, fills
only ExtrameSportFPV / XSports answers, and preserves other dataset answers if
the output file already exists. It does not read external answer files or
precomputed component prediction JSON files. Runtime scratch files are written
to a temporary directory and removed when the run finishes.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from typing import Any


def pin_stable_visible_gpus() -> None:
    """Keep XSports default inference on the two-GPU profile used for release."""

    if os.environ.get("EGOCROSS_XSPORTS_ALLOW_ALL_GPUS") == "1":
        return
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not visible.strip():
        os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
        return
    gpu_ids = [item.strip() for item in visible.split(",") if item.strip()]
    if len(gpu_ids) > 2:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_ids[:2])


pin_stable_visible_gpus()

VALID = {"A", "B", "C", "D"}
XSPORTS_DATASET = "ExtrameSportFPV"
SPECIAL_TASK = "special-action-identification"
ROUTER_CONFIG = {
    "action temporal localization": "atl_sft_aggressive_batch_v1",
    "special action identification": "special_pairwise_fallback_v1",
    "next direction prediction": "option_guided_v1",
    "action sequence identification": "option_guided_v1",
    "sport identification": "option_guided_v1",
}

QUESTION_ID_SUFFIX = {
    ("CholecTrack20", "action temporal localization"): "action-temporal-localization",
    ("CholecTrack20", "dominant held-object identification"): "dominant-instrument-operator-identificat",
    ("CholecTrack20", "next action prediction"): "next-action-prediction",
    ("CholecTrack20", "next phase prediction"): "next-phase-prediction",
    ("CholecTrack20", "object counting"): "distinct-instruments-counting",
    ("CholecTrack20", "object not visible identification"): "instrument-not-visible-identification",
    ("CholecTrack20", "object spatial localization"): "instrument-region-localization",
    ("ENIGMA", "action temporal localization"): "temporal-localization",
    ("ENIGMA", "dominant held-object identification"): "dominant-tool-operator-identification",
    ("ENIGMA", "next interaction prediction"): "next-interaction-prediction",
    ("ENIGMA", "object counting"): "distinct-object-types-counting",
    ("ENIGMA", "object not visible identification"): "object-not-visible-identification",
    ("ENIGMA", "object spatial localization"): "held-object-region-localization",
    ("EgoPet", "animal identification"): "animal-identification",
    ("EgoPet", "interaction identification"): "interaction-identification",
    ("EgoPet", "interaction temporal localization"): "interaction-temporal-localization",
    ("EgoSurgery", "dominant held-object identification"): "dominant-tool-operator-identification",
    ("EgoSurgery", "object counting"): "distinct-object-types-counting",
    ("EgoSurgery", "object not visible identification"): "object-not-visible-identification",
    ("EgoSurgery", "object spatial localization"): "held-object-region-localization",
    ("ExtrameSportFPV", "action sequence identification"): "action-sequence-identification",
    ("ExtrameSportFPV", "action temporal localization"): "action-temporal-localization",
    ("ExtrameSportFPV", "next direction prediction"): "direction-prediction",
    ("ExtrameSportFPV", "special action identification"): "special-action-identification",
    ("ExtrameSportFPV", "sport identification"): "sport-identification",
}


def discover_repo_root(script_dir: Path) -> Path:
    markers = ("ckpts", "testset")
    for candidate in (script_dir, *script_dir.parents):
        if all((candidate / marker).exists() for marker in markers):
            return candidate
    return script_dir


def repo_path(*parts: str) -> Path:
    path = REPO_ROOT.joinpath(*parts)
    if path.exists():
        return path
    return path


def install_import_paths(script_dir: Path) -> tuple[Path, Path]:
    xsports_root = script_dir / "xsports_utils"
    if not xsports_root.is_dir():
        xsports_root = script_dir
    repo_root = discover_repo_root(script_dir)
    for path in (repo_root, xsports_root / "sota", xsports_root):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    return xsports_root, repo_root


SCRIPT_DIR = Path(__file__).resolve().parent
XSPORTS_ROOT, REPO_ROOT = install_import_paths(SCRIPT_DIR)


def purge_foreign_modules(package_root: Path, module_names: list[str]) -> None:
    package_root = package_root.resolve()
    for module_name in module_names:
        module = sys.modules.get(module_name)
        module_file = getattr(module, "__file__", None)
        if not module_file:
            continue
        try:
            if not Path(module_file).resolve().is_relative_to(package_root):
                del sys.modules[module_name]
        except OSError:
            del sys.modules[module_name]


purge_foreign_modules(
    XSPORTS_ROOT,
    [
        "adaptive_atl_router",
        "atl_pairwise_router",
        "direct_decode",
        "frame_sampling",
        "run",
        "xsports_task_pairwise",
    ],
)

# Local imports after sys.path setup.
# pylint: disable=wrong-import-position
from adaptive_atl_router import HARD_ACTIONS, expert_features, extract_action, quantile
from atl_pairwise_router import _question_runtime_inputs, run_pairwise_arbiter
from atl_pairwise_router import should_consider_legacy
from direct_decode import _build_ab_token_ids
from frame_sampling import default_experiment_config
from run import _run_question
from xsports_task_pairwise import predict_one as predict_pairwise_one


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def normalize_answer(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip().upper()
    if text in VALID:
        return text
    for char in text:
        if char in VALID:
            return char
    return ""


def normalize_qtype(value: Any) -> str:
    return " ".join(str(value or "").strip().lower().replace("-", " ").replace("_", " ").split())


def parse_option_body(option: str) -> str:
    text = str(option).strip()
    if len(text) >= 2 and text[0].upper() in VALID and text[1] in {":", "."}:
        return text[2:].strip()
    return text


def option_label(question: dict[str, Any], answer: Any) -> str:
    answer = normalize_answer(answer)
    if not answer:
        return ""
    index = ord(answer) - ord("A")
    options = question.get("options") or []
    if 0 <= index < len(options):
        return parse_option_body(str(options[index]))
    return ""


def by_id(rows: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    return {int(row["id"]): row for row in rows if "id" in row}


def question_id_from_raw(row: dict[str, Any]) -> str:
    dataset = str(row.get("dataset", ""))
    qtype = str(row.get("question_type", ""))
    suffix = QUESTION_ID_SUFFIX[(dataset, qtype)]
    paths = row.get("video_path") or []
    if not paths:
        raise ValueError(f"Missing video_path for id={row.get('id')}")
    parts = str(paths[0]).split("/")
    video = parts[parts.index("generated") + 1]
    frame_dir = parts[parts.index("frames") + 1]
    match = re.search(r"_q(\d+)$", frame_dir) or re.search(r"q(\d+)", frame_dir)
    if not match:
        raise ValueError(f"Cannot infer question number from {paths[0]}")
    return f"{dataset}_{video}_q{match.group(1)}_{suffix}"


def prepare_questions(raw_questions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    questions = []
    for row in raw_questions:
        item = dict(row)
        item["question_id"] = question_id_from_raw(item)
        questions.append(item)
    return questions


def blank_submission_rows(questions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{
        "id": row["id"],
        "question_id": row["question_id"],
        "dataset": row["dataset"],
        "answer": "",
    } for row in questions]


def refresh_submission_metadata(
    rows: list[dict[str, Any]],
    questions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    questions_by_id = {int(q["id"]): q for q in questions if q.get("id") is not None}
    for row in rows:
        try:
            qid = int(row.get("id"))
        except (TypeError, ValueError):
            continue
        question = questions_by_id.get(qid)
        if not question:
            continue
        row["id"] = question.get("id")
        row["question_id"] = question.get("question_id", "")
        row["dataset"] = question.get("dataset", "")
        row["answer"] = normalize_answer(row.get("answer", ""))
    return rows


def base_submission_rows(questions: list[dict[str, Any]], output_path: Path) -> list[dict[str, Any]]:
    if output_path.exists():
        try:
            rows = load_json(output_path)
            if isinstance(rows, list) and len(rows) == len(questions):
                return refresh_submission_metadata(rows, questions)
        except Exception:
            pass
    return blank_submission_rows(questions)


def update_xsports_answer(output_rows: list[dict[str, Any]], item_id: int, answer: str) -> None:
    for row in output_rows:
        if int(row["id"]) == item_id and row["dataset"] == XSPORTS_DATASET:
            row["answer"] = answer
            return


def run_option_guided(
    model: Any,
    processor: Any,
    question: dict[str, Any],
    data_root: Path,
    work_dir: Path,
    *,
    atl: bool = False,
) -> dict[str, Any]:
    cfg = default_experiment_config()
    return _run_question(
        model=model,
        processor=processor,
        question=question,
        data_root=data_root,
        atl_temporal_anchors=atl,
        decode_method="option_guided",
        max_frames=cfg.max_frames,
        support_root=repo_path("EgoCross_support_set"),
        exp_dir=work_dir,
        atl_duration_seconds=None,
        atl_frame_timestamps=False,
        atl_transition_score="log",
    )


def run_transition_guided(
    model: Any,
    processor: Any,
    question: dict[str, Any],
    data_root: Path,
    work_dir: Path,
    *,
    duration_seconds: float = 15.0,
    transition_score: str = "log",
) -> dict[str, Any]:
    cfg = default_experiment_config()
    return _run_question(
        model=model,
        processor=processor,
        question=question,
        data_root=data_root,
        atl_temporal_anchors=True,
        decode_method="transition_guided",
        max_frames=cfg.max_frames,
        support_root=repo_path("EgoCross_support_set"),
        exp_dir=work_dir,
        atl_duration_seconds=duration_seconds,
        atl_frame_timestamps=True,
        atl_transition_score=transition_score,
    )


def option_time_map(question: dict[str, Any]) -> dict[str, float]:
    times: dict[str, float] = {}
    for option in question.get("options", []):
        text = str(option)
        letter = normalize_answer(text[:1])
        if not letter:
            continue
        match = None
        for token in text.replace(",", " ").split():
            if token.lower().endswith("s"):
                try:
                    match = float(token[:-1])
                    break
                except ValueError:
                    continue
        if match is not None:
            times[letter] = match
    return times


def answer_time(question: dict[str, Any], answer: str) -> float:
    return option_time_map(question).get(normalize_answer(answer), -1.0)


def conservative_atl_route(features: dict[str, Any], arbiter: dict[str, Any] | None) -> str:
    arb = arbiter or {}
    gap = features.get("time_gap_T_minus_O")
    o_time = features.get("O_pred_time")
    return "legacy" if (
        arb.get("choose_legacy")
        and arb.get("margin_legacy", -999) >= 0.5
        and gap is not None and float(gap) >= 7.0
        and o_time is not None and float(o_time) <= 5.0
        and features.get("O_pred_rank") == 1
        and features.get("action") not in {"fly", "run"}
    ) else "transition"


def choose_atl_meta_expert(features: dict[str, Any]) -> str:
    """Heuristic temporal stability tree over in-memory ATL experts.

    The features encode a simple prior: prefer the main SFT router unless the
    no-anchor option expert or the 30s transition expert gives a more stable
    early/late-time explanation under duration and base-model sanity checks.
    """
    if features["base_d15_time"] <= 14.05:
        if features["span"] <= 11.15:
            if features["sft_noanchor_time"] <= 6.05:
                if features["aggr_margin"] <= 3.44:
                    return "aggr"
                if features["retry_final_ans"] != features["sft_d30_ans"]:
                    return "sft_d30"
                return "aggr"
            if features["aggr_ans"] != features["sft_d30_ans"]:
                return "sft_d30"
            if features["sft_noanchor_time"] <= 6.90:
                return "sft_noanchor"
            return "aggr"
        if features["base_d30_ans"] == "D":
            return "sft_noanchor"
        return "aggr"
    if features["base_d30_time"] <= 13.70:
        return "sft_d30"
    return "sft_noanchor"


def atl_margin(row: dict[str, Any]) -> float:
    arb = (row.get("arbiter") or row.get("extra_arbiter") or {})
    value = arb.get("margin_legacy")
    return float(value) if value is not None else 0.0


def build_atl_feature_row(
    question: dict[str, Any],
    transition_pred: dict[str, Any],
    legacy_pred: dict[str, Any],
) -> dict[str, Any]:
    t = expert_features(transition_pred, question, "T")
    o = expert_features(legacy_pred, question, "O")
    gap = None
    if t.get("pred_time") is not None and o.get("pred_time") is not None:
        gap = float(t["pred_time"]) - float(o["pred_time"])
    action = extract_action(str(question.get("question_text", "")))
    return {
        "id": int(question["id"]),
        "question_id": question.get("question_id", ""),
        "question_text": question.get("question_text", ""),
        "action": action,
        "hard_action": action in HARD_ACTIONS,
        "T_answer": normalize_answer(t.get("answer") or transition_pred.get("answer")),
        "T_pred_time": t.get("pred_time"),
        "T_pred_rank": t.get("pred_rank"),
        "T_margin": t.get("margin"),
        "O_answer": normalize_answer(o.get("answer") or legacy_pred.get("answer")),
        "O_pred_time": o.get("pred_time"),
        "O_pred_rank": o.get("pred_rank"),
        "O_margin": o.get("margin"),
        "time_gap_T_minus_O": gap,
        "arbiter": None,
        "extra_arbiter": None,
    }


def is_extra_arbiter_target(row: dict[str, Any]) -> bool:
    gap = row.get("time_gap_T_minus_O")
    o_time = row.get("O_pred_time")
    return bool(
        row.get("T_answer") != row.get("O_answer")
        and not row.get("arbiter")
        and row.get("O_pred_rank") == 1
        and o_time is not None
        and float(o_time) <= 5.0
        and gap is not None
        and float(gap) >= 7.0
    )


def choose_atl_route(features: dict[str, Any], arbiter: dict[str, Any] | None) -> str:
    arb = arbiter or {}
    gap = features.get("time_gap_T_minus_O")
    o_time = features.get("O_pred_time")
    o_rank = features.get("O_pred_rank")
    t_rank = features.get("T_pred_rank")
    action = features.get("action")
    margin = arb.get("margin_legacy")

    if (
        arb.get("choose_legacy")
        and margin is not None and float(margin) >= 0.5
        and gap is not None and float(gap) >= 7.0
        and o_time is not None and float(o_time) <= 5.0
        and o_rank == 1
        and action not in {"fly", "run"}
    ):
        return "legacy"

    if not arb.get("choose_legacy"):
        if (
            action == "fly"
            and gap is not None and 1.4 <= float(gap) <= 2.0
            and o_rank == 1
            and t_rank == 2
            and o_time is not None and 2.5 <= float(o_time) <= 3.5
        ):
            return "legacy"
        return "transition"

    if gap is None or margin is None or o_time is None:
        return "transition"
    gap = float(gap)
    margin = float(margin)
    o_time = float(o_time)
    if action == "fly" and gap >= 9.0 and margin >= 4.0 and o_rank == 1 and o_time <= 3.5:
        return "legacy"
    if action == "jump" and o_rank == 2 and 5.0 <= gap <= 6.5 and 0.0 <= margin <= 0.6:
        return "legacy"
    if action == "curveright" and o_rank == 1 and 3.0 <= gap <= 4.2 and 0.2 <= margin <= 0.6:
        return "legacy"
    if action == "curveleft" and o_rank == 1 and gap <= 2.5 and margin >= 2.0:
        return "legacy"
    if action == "jump" and o_rank == 1 and 4.3 <= gap <= 5.0 and 0.5 <= margin <= 1.0:
        return "legacy"
    if action == "jump" and o_rank == 1 and gap <= 2.0 and 1.0 <= margin <= 2.0:
        return "legacy"
    return "transition"


def run_atl_batch(
    model: Any,
    processor: Any,
    base_model: Any,
    base_processor: Any,
    questions: list[dict[str, Any]],
    data_root: Path,
    work_dir: Path,
    choice_a_ids: list[int],
    choice_b_ids: list[int],
    elapsed_by_id: dict[int, float],
) -> dict[int, str]:
    features_by_id: dict[int, dict[str, Any]] = {}
    for index, question in enumerate(questions, start=1):
        item_id = int(question["id"])
        print(f"[ATL {index}/{len(questions)}] transition + legacy experts id={item_id}", flush=True)
        start = time.perf_counter()
        transition = run_transition_guided(
            model,
            processor,
            question,
            data_root,
            work_dir / "atl_transition_exp",
            duration_seconds=15.0,
            transition_score="log",
        )
        legacy = run_option_guided(
            model,
            processor,
            question,
            data_root,
            work_dir / "atl_legacy_exp",
            atl=True,
        )
        features_by_id[item_id] = build_atl_feature_row(question, transition, legacy)
        elapsed_by_id[item_id] += time.perf_counter() - start

    rows = [features_by_id[int(question["id"])] for question in questions]
    q_trans = quantile([float(row["T_margin"]) for row in rows if row.get("T_margin") is not None], 0.60)
    q_old = quantile([float(row["O_margin"]) for row in rows if row.get("O_margin") is not None], 0.65)

    for index, question in enumerate(questions, start=1):
        item_id = int(question["id"])
        row = features_by_id[item_id]
        if not should_consider_legacy(row, q_trans, q_old):
            continue
        print(f"[ATL {index}/{len(questions)}] pairwise arbiter id={item_id}", flush=True)
        start = time.perf_counter()
        paths, frame_times = _question_runtime_inputs(
            question,
            data_root,
            work_dir / "atl_pairwise_main",
            15.0,
        )
        row["arbiter"] = run_pairwise_arbiter(
            model,
            processor,
            question,
            paths,
            frame_times,
            15.0,
            float(row["T_pred_time"]),
            float(row["O_pred_time"]),
            choice_a_ids,
            choice_b_ids,
        )
        elapsed_by_id[item_id] += time.perf_counter() - start

    for index, question in enumerate(questions, start=1):
        item_id = int(question["id"])
        row = features_by_id[item_id]
        if not is_extra_arbiter_target(row):
            continue
        print(f"[ATL {index}/{len(questions)}] extra arbiter id={item_id}", flush=True)
        start = time.perf_counter()
        paths, frame_times = _question_runtime_inputs(
            question,
            data_root,
            work_dir / "atl_extra",
            15.0,
        )
        row["extra_arbiter"] = run_pairwise_arbiter(
            model,
            processor,
            question,
            paths,
            frame_times,
            15.0,
            float(row["T_pred_time"]),
            float(row["O_pred_time"]),
            choice_a_ids,
            choice_b_ids,
        )
        elapsed_by_id[item_id] += time.perf_counter() - start

    answers: dict[int, str] = {}
    for index, question in enumerate(questions, start=1):
        item_id = int(question["id"])
        print(f"[ATL {index}/{len(questions)}] meta experts id={item_id}", flush=True)
        start = time.perf_counter()
        row = features_by_id[item_id]
        arbiter = row.get("arbiter") or row.get("extra_arbiter")

        conservative_route = conservative_atl_route(row, arbiter)
        retry_final_answer = row["O_answer"] if conservative_route == "legacy" else row["T_answer"]

        aggressive_route = choose_atl_route(row, arbiter)
        aggr_answer = row["O_answer"] if aggressive_route == "legacy" else row["T_answer"]

        sft_noanchor = run_option_guided(
            model,
            processor,
            question,
            data_root,
            work_dir / "atl_sft_option_noanchors_exp",
            atl=False,
        )
        sft_d30 = run_transition_guided(
            model,
            processor,
            question,
            data_root,
            work_dir / "atl_sft_trans_d30_ts_exp",
            duration_seconds=30.0,
            transition_score="log",
        )
        base_d15 = run_transition_guided(
            base_model,
            base_processor,
            question,
            data_root,
            work_dir / "atl_base_trans_d15_product_exp",
            duration_seconds=15.0,
            transition_score="product",
        )
        base_d30 = run_transition_guided(
            base_model,
            base_processor,
            question,
            data_root,
            work_dir / "atl_base_trans_d30_ts_exp",
            duration_seconds=30.0,
            transition_score="log",
        )

        times = option_time_map(question)
        span = max(times.values()) - min(times.values()) if times else -1.0
        meta_features = {
            "span": span,
            "aggr_ans": normalize_answer(aggr_answer),
            "aggr_margin": atl_margin(row),
            "retry_final_ans": normalize_answer(retry_final_answer),
            "sft_noanchor_ans": normalize_answer(sft_noanchor.get("answer")),
            "sft_noanchor_time": answer_time(question, sft_noanchor.get("answer", "")),
            "sft_d30_ans": normalize_answer(sft_d30.get("answer")),
            "sft_d30_time": answer_time(question, sft_d30.get("answer", "")),
            "base_d15_ans": normalize_answer(base_d15.get("answer")),
            "base_d15_time": answer_time(question, base_d15.get("answer", "")),
            "base_d30_ans": normalize_answer(base_d30.get("answer")),
            "base_d30_time": answer_time(question, base_d30.get("answer", "")),
        }
        expert = choose_atl_meta_expert(meta_features)
        if expert == "sft_noanchor":
            answer = sft_noanchor.get("answer")
        elif expert == "sft_d30":
            answer = sft_d30.get("answer")
        else:
            answer = aggr_answer
        answers[item_id] = normalize_answer(answer)
        elapsed_by_id[item_id] += time.perf_counter() - start
    return answers


def pairwise_margin(row: dict[str, Any]) -> float:
    scores = row.get("pairwise_task", {}).get("scores", {})
    if not isinstance(scores, dict) or len(scores) < 2:
        return 0.0
    ordered = sorted(scores, key=scores.get, reverse=True)
    return float(scores[ordered[0]]) - float(scores[ordered[1]])


def run_option_batch(
    model: Any,
    processor: Any,
    questions: list[dict[str, Any]],
    data_root: Path,
    work_dir: Path,
    elapsed_by_id: dict[int, float],
) -> dict[int, dict[str, Any]]:
    rows: dict[int, dict[str, Any]] = {}
    for index, question in enumerate(questions, start=1):
        item_id = int(question["id"])
        print(f"[baseline {index}/{len(questions)}] option-guided id={item_id}", flush=True)
        start = time.perf_counter()
        rows[item_id] = run_option_guided(
            model,
            processor,
            question,
            data_root,
            work_dir / "baseline_option_guided_exp",
        )
        elapsed_by_id[item_id] += time.perf_counter() - start
    return rows


def run_special_batch(
    model: Any,
    processor: Any,
    questions: list[dict[str, Any]],
    option_rows: dict[int, dict[str, Any]],
    data_root: Path,
    choice_a_ids: list[int],
    choice_b_ids: list[int],
    elapsed_by_id: dict[int, float],
) -> dict[int, str]:
    answers: dict[int, str] = {}
    for index, question in enumerate(questions, start=1):
        item_id = int(question["id"])
        print(f"[special {index}/{len(questions)}] pairwise id={item_id}", flush=True)
        start = time.perf_counter()
        pair_row = predict_pairwise_one(
            model,
            processor,
            question,
            data_root,
            SPECIAL_TASK,
            choice_a_ids,
            choice_b_ids,
        )
        elapsed_by_id[item_id] += time.perf_counter() - start
        option_row = option_rows[item_id]
        pair_label = option_label(question, pair_row.get("answer"))
        option_label_text = option_label(question, option_row.get("answer"))
        margin = pairwise_margin(pair_row)
        use_option = pair_label == "Spin" or (
            pair_label == "Vault"
            and margin < 5
            and option_label_text in {"Jump", "Climb", "Flip", "Fly"}
        )
        answers[item_id] = normalize_answer(option_row.get("answer") if use_option else pair_row.get("answer"))
    return answers


def split_questions(questions: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for question in questions:
        qtype = normalize_qtype(question.get("question_type"))
        groups[qtype].append(question)
    return groups


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run XSports direct inference and write a submission-format JSON.")
    parser.add_argument("--questions", type=Path, default=SCRIPT_DIR / "testset" / "egocross_testbed_imgs.json")
    parser.add_argument("--output", type=Path, default=SCRIPT_DIR / "submission.json")
    parser.add_argument("--data-root", type=Path, default=SCRIPT_DIR / "testset")
    parser.add_argument("--model-path", type=Path, default=repo_path("ckpts", "xsports"))
    parser.add_argument("--base-model-path", type=Path, default=repo_path("ckpts", "base_model"))
    parser.add_argument("--work-dir", type=Path, default=None, help="Optional scratch directory. By default a temp directory is used and removed.")
    parser.add_argument("--keep-work-dir", action="store_true", help="Keep runtime scratch files for debugging.")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--allow-remote-model", action="store_true")
    parser.add_argument("--limit", type=int, default=None, help="Optional smoke-test limit over XSports questions.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = args.output

    questions = prepare_questions(load_json(args.questions))
    submission_rows = base_submission_rows(questions, output_path)
    xsports_questions = [q for q in questions if q.get("dataset") == XSPORTS_DATASET]
    if args.limit is not None:
        xsports_questions = xsports_questions[: args.limit]
    temp_work_dir: Path | None = None
    if args.work_dir is None:
        temp_work_dir = Path(tempfile.mkdtemp(prefix="xsports_runtime_"))
        work_dir = temp_work_dir
    else:
        work_dir = args.work_dir
        work_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_path, submission_rows)

    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    try:
        from transformers.utils import logging as hf_logging

        hf_logging.set_verbosity_error()
        hf_logging.disable_progress_bar()
    except Exception:
        pass

    try:
        from transformers import AutoModelForImageTextToText, AutoProcessor

        def load_model(model_path: Path) -> tuple[Any, Any]:
            print(f"[load] {model_path}", flush=True)
            model_obj = AutoModelForImageTextToText.from_pretrained(
                str(model_path),
                dtype=args.dtype,
                device_map=args.device_map,
                local_files_only=not args.allow_remote_model,
            )
            processor_obj = AutoProcessor.from_pretrained(
                str(model_path),
                local_files_only=not args.allow_remote_model,
            )
            print(f"[loaded] {model_path}", flush=True)
            return model_obj, processor_obj

        model, processor = load_model(args.model_path)
        choice_a_ids, choice_b_ids = _build_ab_token_ids(processor)

        groups = split_questions(xsports_questions)
        elapsed_by_id: dict[int, float] = defaultdict(float)
        answers: dict[int, str] = {}

        option_questions = []
        option_questions.extend(groups.get("next direction prediction", []))
        option_questions.extend(groups.get("action sequence identification", []))
        option_questions.extend(groups.get("sport identification", []))
        option_questions.extend(groups.get("special action identification", []))
        print(f"[run] baseline option-guided questions={len(option_questions)}", flush=True)
        option_rows = run_option_batch(
            model,
            processor,
            option_questions,
            args.data_root,
            work_dir,
            elapsed_by_id,
        )
        for question in option_questions:
            qtype = normalize_qtype(question.get("question_type"))
            if ROUTER_CONFIG.get(qtype) == "option_guided_v1":
                answers[int(question["id"])] = normalize_answer(option_rows[int(question["id"])].get("answer"))

        print(f"[run] special-action questions={len(groups.get('special action identification', []))}", flush=True)
        answers.update(
            run_special_batch(
                model,
                processor,
                groups.get("special action identification", []),
                option_rows,
                args.data_root,
                choice_a_ids,
                choice_b_ids,
                elapsed_by_id,
            )
        )
        print(f"[run] ATL questions={len(groups.get('action temporal localization', []))}", flush=True)
        base_model, base_processor = load_model(args.base_model_path)
        answers.update(
            run_atl_batch(
                model,
                processor,
                base_model,
                base_processor,
                groups.get("action temporal localization", []),
                args.data_root,
                work_dir,
                choice_a_ids,
                choice_b_ids,
                elapsed_by_id,
            )
        )

        for question in xsports_questions:
            item_id = int(question["id"])
            qtype = normalize_qtype(question.get("question_type"))
            answer = normalize_answer(answers.get(item_id))
            update_xsports_answer(submission_rows, item_id, answer)
            write_json(output_path, submission_rows)
            print(f"id={item_id} qtype=\"{qtype}\" time={elapsed_by_id[item_id]:.2f}s answer={answer}", flush=True)
    finally:
        if temp_work_dir is not None and not args.keep_work_dir:
            shutil.rmtree(temp_work_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
