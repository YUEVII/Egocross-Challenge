"""Direct option-guided yes/no decoding for EgoCross questions."""

from __future__ import annotations

import math
import re
from typing import Any

from egocross.parsing import extract_option_body
from egocross.parsing import normalize_option


_LETTERS = ("A", "B", "C", "D")
_TIME_PATTERN = re.compile(r"-?\d+(?:\.\d+)?")
_EPS = 1e-6


def build_yes_no_token_ids(processor: Any) -> tuple[list[int], list[int]]:
    """Builds candidate token ids for yes/no next-token scoring.

    Args:
        processor: Hugging Face processor with a tokenizer.

    Returns:
        A tuple of yes token ids and no token ids.
    """

    yes_token_ids = _candidate_token_ids(processor, ("yes", "YES"))
    no_token_ids = _candidate_token_ids(processor, ("no", "NO"))
    return yes_token_ids, no_token_ids


def run_option_guided_verification(
    model: Any,
    processor: Any,
    image_paths: list[str],
    question: dict[str, Any],
    yes_token_ids: list[int],
    no_token_ids: list[int],
    *,
    prompt_prefix: str = "",
    frame_prefix_texts: list[str] | None = None,
) -> dict[str, Any]:
    """Scores every answer option with a yes/no visual verification prompt.

    Args:
        model: Loaded vision-language model.
        processor: Matching model processor.
        image_paths: Selected frame paths.
        question: EgoCross question item with four options.
        yes_token_ids: Candidate token ids for yes.
        no_token_ids: Candidate token ids for no.

    Returns:
        Decoding details containing the final answer and per-option scores.

    Raises:
        ValueError: If the question does not have four options.
    """

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
            prompt_prefix=prompt_prefix,
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
        "option_guided_verification": option_scores,
    }


def run_transition_guided_verification(
    model: Any,
    processor: Any,
    image_paths: list[str],
    question: dict[str, Any],
    yes_token_ids: list[int],
    no_token_ids: list[int],
    *,
    prompt_prefix: str = "",
    frame_times: list[float] | None = None,
    score_mode: str = "log",
    candidate_indices: set[int] | None = None,
) -> dict[str, Any]:
    """Scores ATL candidates by before/after transition evidence.

    For each candidate time ``t`` this asks two yes/no questions over local
    windows near ``t``:
      - before window: has the action already started before ``t``?
      - after window: has the action started by / just after ``t``?

    The final score rewards high post probability and low pre probability.
    """

    candidates = _candidate_times(question)
    if candidate_indices is not None:
        candidates = [
            candidate
            for candidate in candidates
            if int(candidate["option_index"]) in candidate_indices
        ]
    if not candidates:
        raise ValueError("No candidate timestamps available for transition guidance.")

    frame_times = _default_frame_times(image_paths, frame_times)
    sorted_block = _sorted_candidates_block(_candidate_times(question))
    option_scores = []
    for candidate in candidates:
        score = _score_transition_candidate(
            model=model,
            processor=processor,
            image_paths=image_paths,
            frame_times=frame_times,
            question=question,
            candidate=candidate,
            sorted_candidates_block=sorted_block,
            yes_token_ids=yes_token_ids,
            no_token_ids=no_token_ids,
            prompt_prefix=prompt_prefix,
            score_mode=score_mode,
        )
        option_scores.append(score)

    best_score = max(option_scores, key=lambda item: item["transition_score"])
    return {
        "answer": best_score["letter"],
        "raw_answer": (
            f"{best_score['letter']} "
            f"(transition_score={best_score['transition_score']:.6f}, "
            f"time={best_score['time_seconds']:.3f}s)"
        ),
        "transition_guided_verification": option_scores,
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
                "the correct answer to the question?"
            ),
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


def _build_ab_token_ids(processor: Any) -> tuple[list[int], list[int]]:
    return (
        _candidate_token_ids(processor, ("A", "a")),
        _candidate_token_ids(processor, ("B", "b")),
    )


def _extract_option_time(option_text: str) -> float | None:
    match = _TIME_PATTERN.search(option_text)
    if match is None:
        return None
    return float(match.group())


def _candidate_times(question: dict[str, Any]) -> list[dict[str, Any]]:
    options = question.get("options", [])
    if len(options) != 4:
        raise ValueError(f"Expected 4 options, got {len(options)}.")

    candidates = []
    for option_index, option_text in enumerate(options):
        option_text = str(option_text)
        time_seconds = _extract_option_time(option_text)
        if time_seconds is None:
            raise ValueError(f"Could not parse timestamp from option: {option_text}")
        candidates.append(
            {
                "letter": _LETTERS[option_index],
                "option_index": option_index,
                "option": normalize_option(option_text),
                "time_seconds": time_seconds,
            }
        )
    return sorted(candidates, key=lambda item: item["time_seconds"])


def _sorted_candidates_block(candidates: list[dict[str, Any]]) -> str:
    lines = ["Candidate times sorted by real time:"]
    for candidate in candidates:
        lines.append(
            f"- {candidate['time_seconds']:.3f}s "
            f"(original option {candidate['letter']})"
        )
    return "\n".join(lines)


def _default_frame_times(
    image_paths: list[str],
    frame_times: list[float] | None,
) -> list[float]:
    if frame_times is not None and len(frame_times) == len(image_paths):
        return frame_times
    return [float(index + 1) for index in range(len(image_paths))]


def _frame_label(
    index: int,
    frame_time: float,
    total_frames: int,
    *,
    use_edge_anchors: bool = True,
) -> str:
    if use_edge_anchors and index == 0:
        return f"START anchor. Frame {index + 1}, timestamp ~= {frame_time:.2f}s."
    if use_edge_anchors and index == total_frames - 1:
        return f"END anchor. Frame {index + 1}, timestamp ~= {frame_time:.2f}s."
    suffix = "." if use_edge_anchors else ""
    return f"Frame {index + 1}, timestamp ~= {frame_time:.2f}s{suffix}"


def _window_targets(candidate_time: float, offsets: tuple[float, ...]) -> list[float]:
    return [max(0.0, candidate_time + offset) for offset in offsets]


def _select_local_window(
    image_paths: list[str],
    frame_times: list[float],
    target_times: list[float],
    *,
    use_edge_anchors: bool = True,
) -> tuple[list[str], list[str]]:
    frame_to_targets: dict[int, list[float]] = {}
    for target_time in target_times:
        nearest_index = min(
            range(len(frame_times)),
            key=lambda index: abs(frame_times[index] - target_time),
        )
        frame_to_targets.setdefault(nearest_index, []).append(target_time)

    selected_paths = []
    labels = []
    for frame_index in sorted(frame_to_targets, key=lambda index: frame_times[index]):
        selected_paths.append(image_paths[frame_index])
        targets = ", ".join(
            f"{target_time:.2f}s" for target_time in frame_to_targets[frame_index]
        )
        labels.append(
            f"{_frame_label(frame_index, frame_times[frame_index], len(image_paths), use_edge_anchors=use_edge_anchors)} "
            f"(nearest requested local time(s): {targets})."
        )
    return selected_paths, labels


def _score_transition_candidate(
    model: Any,
    processor: Any,
    image_paths: list[str],
    frame_times: list[float],
    question: dict[str, Any],
    candidate: dict[str, Any],
    sorted_candidates_block: str,
    yes_token_ids: list[int],
    no_token_ids: list[int],
    *,
    prompt_prefix: str = "",
    score_mode: str = "log",
) -> dict[str, Any]:
    candidate_time = float(candidate["time_seconds"])
    pre_paths, pre_labels = _select_local_window(
        image_paths,
        frame_times,
        _window_targets(candidate_time, (-1.2, -0.6, -0.2)),
        use_edge_anchors=False,
    )
    post_paths, post_labels = _select_local_window(
        image_paths,
        frame_times,
        _window_targets(candidate_time, (0.2, 0.6, 1.2)),
        use_edge_anchors=False,
    )

    pre_prob, pre_detail = _transition_probability(
        model=model,
        processor=processor,
        image_paths=pre_paths,
        frame_prefix_texts=pre_labels,
        question=question,
        candidate=candidate,
        phase="pre",
        sorted_candidates_block=sorted_candidates_block,
        yes_token_ids=yes_token_ids,
        no_token_ids=no_token_ids,
        prompt_prefix=prompt_prefix,
    )
    post_prob, post_detail = _transition_probability(
        model=model,
        processor=processor,
        image_paths=post_paths,
        frame_prefix_texts=post_labels,
        question=question,
        candidate=candidate,
        phase="post",
        sorted_candidates_block=sorted_candidates_block,
        yes_token_ids=yes_token_ids,
        no_token_ids=no_token_ids,
        prompt_prefix=prompt_prefix,
    )
    if score_mode == "product":
        score = post_prob * (1.0 - pre_prob)
    else:
        score = math.log(post_prob + _EPS) + math.log(1.0 - pre_prob + _EPS)

    return {
        **candidate,
        "pre_started_probability": pre_prob,
        "post_started_probability": post_prob,
        "transition_score": score,
        "score_mode": score_mode,
        "pre_window": pre_detail,
        "post_window": post_detail,
    }


def _transition_probability(
    model: Any,
    processor: Any,
    image_paths: list[str],
    frame_prefix_texts: list[str],
    question: dict[str, Any],
    candidate: dict[str, Any],
    phase: str,
    sorted_candidates_block: str,
    yes_token_ids: list[int],
    no_token_ids: list[int],
    *,
    prompt_prefix: str = "",
) -> tuple[float, dict[str, Any]]:
    prompt = _build_transition_prompt(
        question=question,
        candidate=candidate,
        phase=phase,
        sorted_candidates_block=sorted_candidates_block,
    )
    if prompt_prefix:
        prompt = f"{prompt_prefix}\n\n{prompt}"
    yes_logprob, no_logprob = _next_token_logprobs(
        model=model,
        processor=processor,
        image_paths=image_paths,
        prompt=prompt,
        yes_token_ids=yes_token_ids,
        no_token_ids=no_token_ids,
        frame_prefix_texts=frame_prefix_texts,
    )
    probability = _binary_probability(yes_logprob, no_logprob)
    return probability, {
        "image_paths": image_paths,
        "frame_prefix_texts": frame_prefix_texts,
        "yes_logprob": yes_logprob,
        "no_logprob": no_logprob,
        "yes_probability": probability,
    }


def _build_transition_prompt(
    question: dict[str, Any],
    candidate: dict[str, Any],
    phase: str,
    sorted_candidates_block: str,
) -> str:
    question_text = str(question.get("question_text", "")).strip()
    candidate_time = float(candidate["time_seconds"])
    phase_text = (
        "BEFORE window immediately before the candidate time"
        if phase == "pre"
        else "AFTER window immediately after the candidate time"
    )
    final_question = (
        f"Has the target action already started before {candidate_time:.3f}s?"
        if phase == "pre"
        else f"Has the target action started by or just after {candidate_time:.3f}s?"
    )
    lines = [
        "You are doing transition-guided temporal localization.",
        "Use only the local window frames shown above.",
        f"These frames are the {phase_text}.",
        "",
        "Original question:",
        question_text,
        "",
        sorted_candidates_block,
        "",
        f"Candidate time under test: {candidate_time:.3f}s.",
    ]
    lines.extend(
        [
            "",
            "Important: preparation before the action is not the action onset.",
            "Answer with exactly one word: yes or no.",
            "",
            final_question,
            "Answer:",
        ]
    )
    return "\n".join(lines)


def _verify_option(
    model: Any,
    processor: Any,
    image_paths: list[str],
    question: dict[str, Any],
    option_text: str,
    yes_token_ids: list[int],
    no_token_ids: list[int],
    *,
    prompt_prefix: str = "",
    frame_prefix_texts: list[str] | None = None,
) -> dict[str, Any]:
    prompt = _build_verify_prompt(question, option_text)
    if prompt_prefix:
        prompt = f"{prompt_prefix}\n\n{prompt}"
    yes_logprob, no_logprob = _next_token_logprobs(
        model=model,
        processor=processor,
        image_paths=image_paths,
        prompt=prompt,
        yes_token_ids=yes_token_ids,
        no_token_ids=no_token_ids,
        frame_prefix_texts=frame_prefix_texts,
    )
    yes_prob = _binary_probability(yes_logprob, no_logprob)
    return {
        "option": normalize_option(option_text),
        "yes_logprob": yes_logprob,
        "no_logprob": no_logprob,
        "yes_probability": yes_prob,
    }


def _next_token_logprobs(
    model: Any,
    processor: Any,
    image_paths: list[str],
    prompt: str,
    yes_token_ids: list[int],
    no_token_ids: list[int],
    *,
    frame_prefix_texts: list[str] | None = None,
) -> tuple[float, float]:
    # Import lazily so CLI help for callers stays lightweight.
    # pylint: disable=import-outside-toplevel
    import torch

    content: list[dict[str, Any]] = []
    for idx, image_path in enumerate(image_paths):
        content.append({"type": "image", "image": image_path})
        if frame_prefix_texts is not None and idx < len(frame_prefix_texts):
            content.append({"type": "text", "text": frame_prefix_texts[idx]})
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
    choice_a_ids: list[int],
    choice_b_ids: list[int],
    *,
    frame_prefix_texts: list[str] | None = None,
) -> tuple[float, float]:
    # Import lazily so CLI help for callers stays lightweight.
    # pylint: disable=import-outside-toplevel
    import torch

    content: list[dict[str, Any]] = []
    for idx, image_path in enumerate(image_paths):
        content.append({"type": "image", "image": image_path})
        if frame_prefix_texts is not None and idx < len(frame_prefix_texts):
            content.append({"type": "text", "text": frame_prefix_texts[idx]})
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
    choice_a_scores = log_probs[0, choice_a_ids]
    choice_b_scores = log_probs[0, choice_b_ids]
    return (
        float(torch.max(choice_a_scores).detach().cpu()),
        float(torch.max(choice_b_scores).detach().cpu()),
    )


def _binary_probability(yes_logprob: float, no_logprob: float) -> float:
    max_logprob = max(yes_logprob, no_logprob)
    yes_score = math.exp(yes_logprob - max_logprob)
    no_score = math.exp(no_logprob - max_logprob)
    return yes_score / (yes_score + no_score)
