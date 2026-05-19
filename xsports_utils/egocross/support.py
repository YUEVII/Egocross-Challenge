from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from egocross.parsing import parse_letter
from egocross.prompting import (
    clean_support_prompt,
    infer_task_family,
    tokenize_for_overlap,
)


def dataset_to_support_file(dataset_name: str) -> str:
    support_files = {
        "CHOLECTRACK20": "train_surgery.json",
        "EGOSURGERY": "train_surgery.json",
        "ENIGMA": "train_industry.json",
        "EXTRAMESPORTFPV": "train_xsports.json",
        "EGOPET": "train_animal.json",
    }
    return support_files.get(dataset_name.upper(), "train.json")


def load_support_examples(support_root: str) -> dict[str, list[dict[str, Any]]]:
    support_root_path = Path(support_root)
    support_examples: dict[str, list[dict[str, Any]]] = {}
    for file_name in [
        "train.json",
        "train_animal.json",
        "train_industry.json",
        "train_surgery.json",
        "train_xsports.json",
    ]:
        path = support_root_path / file_name
        with path.open("r", encoding="utf-8") as file_obj:
            support_examples[file_name] = json.load(file_obj)
    return support_examples


def sample_evenly(items: list[str], max_items: int) -> list[str]:
    if len(items) <= max_items:
        return items
    if max_items <= 1:
        return [items[0]]
    indices = []
    last_index = len(items) - 1
    for position in range(max_items):
        index = round(position * last_index / (max_items - 1))
        indices.append(index)
    return [items[index] for index in indices]


def build_support_message(
    example: dict[str, Any],
    support_root: str,
    images_per_example: int,
) -> list[dict[str, Any]]:
    messages = example.get("messages", [])
    if len(messages) < 2:
        raise ValueError("Support example must contain user and assistant turns.")
    prompt_text = clean_support_prompt(messages[0].get("content", ""))
    raw_ans = messages[1].get("content", "")
    letter, _ = parse_letter(str(raw_ans))
    answer_text = letter if letter else str(raw_ans).strip()
    image_paths = [
        str(Path(support_root) / relative_path)
        for relative_path in sample_evenly(
            example.get("images", []),
            images_per_example,
        )
    ]
    user_content: list[dict[str, Any]] = [
        {"type": "image", "image": image_path} for image_path in image_paths
    ]
    user_content.append({"type": "text", "text": prompt_text})
    return [
        {"role": "user", "content": user_content},
        {
            "role": "assistant",
            "content": [{"type": "text", "text": answer_text}],
        },
    ]


def select_support_examples(
    question: dict[str, Any],
    support_pool: list[dict[str, Any]],
    max_examples: int,
) -> list[dict[str, Any]]:
    task_family = infer_task_family(
        question_text=question.get("question_text", ""),
        question_type=question.get("question_type", ""),
        primary_category=question.get("primary_category", ""),
    )
    query_tokens = tokenize_for_overlap(question.get("question_text", ""))
    scored_examples: list[tuple[int, dict[str, Any]]] = []
    for example in support_pool:
        user_messages = example.get("messages", [])
        if not user_messages:
            continue
        example_prompt = clean_support_prompt(
            user_messages[0].get("content", "")
        )
        example_family = infer_task_family(
            question_text=example_prompt,
            question_type="",
            primary_category="",
        )
        overlap = len(query_tokens & tokenize_for_overlap(example_prompt))
        family_bonus = 5 if example_family == task_family else 0
        scored_examples.append((family_bonus + overlap, example))
    scored_examples.sort(key=lambda item: item[0], reverse=True)
    return [example for _, example in scored_examples[:max_examples]]


def should_use_support_for_question(
    question: dict[str, Any], dataset_allowlist: set[str]
) -> bool:
    dataset_name = str(question.get("dataset", "")).upper().replace(" ", "")
    normalized = dataset_name
    if normalized in dataset_allowlist:
        return True
    return False
