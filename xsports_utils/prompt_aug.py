"""Prompt templates and prompt-level metrics for EgoCross experiments."""

from __future__ import annotations

import re
from dataclasses import asdict
from dataclasses import dataclass
from typing import Any

from egocross.parsing import normalize_option


_VALID_ANSWERS = frozenset({"A", "B", "C", "D"})


@dataclass(frozen=True)
class PromptConfig:
    """Configuration for one prompt experiment.

    Args:
        name: Prompt experiment id, for example P0.
        template_name: Template family name.
        description: Human-readable description.
    """

    name: str
    template_name: str
    description: str


_PROMPT_EXPERIMENTS = {
    "P0": PromptConfig(
        name="P0",
        template_name="direct",
        description="Direct",
    ),
    "P1": PromptConfig(
        name="P1",
        template_name="evidence_first",
        description="Evidence First",
    ),
    "P2": PromptConfig(
        name="P2",
        template_name="option_verification",
        description="Option Verification",
    ),
    "P3": PromptConfig(
        name="P3",
        template_name="elimination",
        description="Elimination",
    ),
    "P4": PromptConfig(
        name="P4",
        template_name="domain_expert",
        description="Domain Expert",
    ),
    "P5": PromptConfig(
        name="P5",
        template_name="type_expert",
        description="Question-Type Expert",
    ),
    "P6": PromptConfig(
        name="P6",
        template_name="domain_type_expert",
        description="Domain + Type Expert",
    ),
}


def available_prompt_experiments() -> tuple[str, ...]:
    """Returns available prompt experiment ids."""

    return tuple(sorted(_PROMPT_EXPERIMENTS))


def get_prompt_config(prompt_experiment: str) -> PromptConfig:
    """Returns a prompt experiment configuration.

    Args:
        prompt_experiment: Prompt experiment id, for example P0.

    Returns:
        Prompt configuration.

    Raises:
        ValueError: If the prompt experiment is unknown.
    """

    normalized_name = prompt_experiment.upper()
    if normalized_name not in _PROMPT_EXPERIMENTS:
        valid_names = ", ".join(available_prompt_experiments())
        raise ValueError(
            f"Unknown prompt experiment {prompt_experiment!r}. "
            f"Valid values: {valid_names}"
        )
    return _PROMPT_EXPERIMENTS[normalized_name]


def prompt_config_to_dict(config: PromptConfig) -> dict[str, Any]:
    """Converts a prompt config to a JSON-serializable dict.

    Args:
        config: Prompt experiment configuration.

    Returns:
        JSON-serializable config dictionary.
    """

    return asdict(config)


def build_prompt(
    question: dict[str, Any],
    prompt_config: PromptConfig,
) -> str:
    """Builds a prompt for one EgoCross question.

    Args:
        question: EgoCross question item.
        prompt_config: Prompt experiment configuration.

    Returns:
        Prompt string.
    """

    template_name = prompt_config.template_name
    if template_name == "direct":
        return _build_direct_prompt(question)
    if template_name == "evidence_first":
        return _build_evidence_first_prompt(question)
    if template_name == "option_verification":
        return _build_option_verification_prompt(question)
    if template_name == "elimination":
        return _build_elimination_prompt(question)
    if template_name == "domain_expert":
        return _build_expert_prompt(
            question=question,
            expert_text=_domain_expert_text(question),
        )
    if template_name == "type_expert":
        return _build_expert_prompt(
            question=question,
            expert_text=_type_expert_text(question),
        )
    if template_name == "domain_type_expert":
        return _build_expert_prompt(
            question=question,
            expert_text="\n".join(
                [_domain_expert_text(question), _type_expert_text(question)]
            ),
        )
    raise ValueError(f"Unknown prompt template: {template_name}")


def extract_answer(raw_answer: str | None) -> tuple[str | None, str]:
    """Extracts the final answer and classifies output format.

    Args:
        raw_answer: Raw model output.

    Returns:
        Tuple of answer letter and format status.
    """

    if raw_answer is None:
        return None, "empty"
    text = str(raw_answer).strip()
    if not text:
        return None, "empty"

    upper_text = text.upper()
    if upper_text in _VALID_ANSWERS:
        return upper_text, "single_letter"

    final_match = re.search(
        r"(?:FINAL\s+ANSWER|ANSWER)\s*[:：]?\s*([ABCD])\b",
        upper_text,
    )
    if final_match:
        return final_match.group(1), "final_answer_line"

    first_line = upper_text.splitlines()[0].strip()
    if first_line in _VALID_ANSWERS:
        return first_line, "first_line_letter"

    letters = [char for char in upper_text if char in _VALID_ANSWERS]
    if letters:
        return letters[0], "fallback_letter_scan"
    return None, "invalid"


def count_text_tokens(processor: Any, text: str) -> int:
    """Estimates text-token cost for a prompt or model output.

    Args:
        processor: Hugging Face processor, usually with a tokenizer.
        text: Text to tokenize.

    Returns:
        Token count. Falls back to whitespace count if no tokenizer exists.
    """

    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None:
        return len(text.split())
    try:
        token_ids = tokenizer.encode(text, add_special_tokens=False)
    except (TypeError, ValueError):
        return len(text.split())
    return len(token_ids)


def build_prompt_metrics(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    """Builds prompt-level summary metrics from prediction rows.

    Args:
        predictions: Prediction rows emitted by the runner.

    Returns:
        Summary metrics for output format and token cost.
    """

    total_count = len(predictions)
    format_counts: dict[str, int] = {}
    error_counts: dict[str, int] = {}
    prompt_tokens = 0
    output_tokens = 0

    for row in predictions:
        format_status = str(row.get("output_format_status", "unknown"))
        format_counts[format_status] = format_counts.get(format_status, 0) + 1
        error_type = str(row.get("error_type", "unknown"))
        error_counts[error_type] = error_counts.get(error_type, 0) + 1
        prompt_tokens += int(row.get("prompt_token_count", 0))
        output_tokens += int(row.get("output_token_count", 0))

    stable_count = sum(
        format_counts.get(status, 0)
        for status in (
            "single_letter",
            "final_answer_line",
            "first_line_letter",
        )
    )
    average_prompt_tokens = prompt_tokens / total_count if total_count else 0.0
    average_output_tokens = output_tokens / total_count if total_count else 0.0
    return {
        "output_format_stability": stable_count / total_count
        if total_count
        else 0.0,
        "stable_output_count": stable_count,
        "format_counts": dict(sorted(format_counts.items())),
        "average_prompt_text_tokens": round(average_prompt_tokens, 3),
        "average_output_text_tokens": round(average_output_tokens, 3),
        "average_total_text_tokens": round(
            average_prompt_tokens + average_output_tokens,
            3,
        ),
        "error_type_counts": dict(sorted(error_counts.items())),
    }


def _build_direct_prompt(question: dict[str, Any]) -> str:
    return "\n".join(
        [
            "You are answering a multiple-choice question about an "
            "egocentric video.",
            "Inspect the video frames carefully.",
            "Focus on visible evidence only.",
            "",
            _question_block(question),
            "",
            _options_block(question),
            "",
            "Output the final answer as one letter only: A, B, C, or D.",
        ]
    )


def _build_evidence_first_prompt(question: dict[str, Any]) -> str:
    return "\n".join(
        [
            _common_header(),
            "",
            _question_block(question),
            "",
            _options_block(question),
            "",
            "Step 1: describe the relevant visual evidence.",
            "Step 2: compare each option against the evidence.",
            "Step 3: output the final answer as: Final answer: <A/B/C/D>.",
        ]
    )


def _build_option_verification_prompt(question: dict[str, Any]) -> str:
    return "\n".join(
        [
            _common_header(),
            "",
            _question_block(question),
            "",
            _options_block(question),
            "",
            "Check A, B, C, and D separately against the frames.",
            "Mark each option as supported or not supported.",
            "Then output exactly one final line: Final answer: <A/B/C/D>.",
        ]
    )


def _build_elimination_prompt(question: dict[str, Any]) -> str:
    return "\n".join(
        [
            _common_header(),
            "",
            _question_block(question),
            "",
            _options_block(question),
            "",
            "Eliminate options contradicted by the visible evidence.",
            "If multiple options remain, choose the one best supported.",
            "End with exactly one final line: Final answer: <A/B/C/D>.",
        ]
    )


def _build_expert_prompt(question: dict[str, Any], expert_text: str) -> str:
    return "\n".join(
        [
            _common_header(),
            "",
            expert_text,
            "",
            _question_block(question),
            "",
            _options_block(question),
            "",
            "Use the expert checklist, but rely only on visible evidence.",
            "End with exactly one final line: Final answer: <A/B/C/D>.",
        ]
    )


def _common_header() -> str:
    return "\n".join(
        [
            "You are answering a multiple-choice question about an "
            "egocentric video.",
            "First inspect the video frames carefully.",
            "Focus on visible evidence only.",
            "Do not guess from commonsense unless visual evidence supports it.",
        ]
    )


def _question_block(question: dict[str, Any]) -> str:
    return "\n".join(
        [
            "Question:",
            str(question.get("question_text", "")).strip(),
        ]
    )


def _options_block(question: dict[str, Any]) -> str:
    option_lines = []
    for option in question.get("options", []):
        option_lines.append(normalize_option(str(option)))
    return "\n".join(["Options:"] + option_lines)


def _domain_expert_text(question: dict[str, Any]) -> str:
    dataset = str(question.get("dataset", "")).lower()
    if "cholec" in dataset or "surgery" in dataset:
        return (
            "Domain expert perspective: inspect surgical instruments, tissue "
            "regions, hand/tool interactions, and temporal phase changes."
        )
    if "enigma" in dataset:
        return (
            "Domain expert perspective: inspect industrial parts, worker hand "
            "actions, assembly state, tools, and object locations."
        )
    if "sport" in dataset or "fpv" in dataset:
        return (
            "Domain expert perspective: inspect motion direction, camera "
            "trajectory, visible obstacles, and upcoming action cues."
        )
    if "pet" in dataset or "animal" in dataset:
        return (
            "Domain expert perspective: inspect animal viewpoint cues, nearby "
            "objects, interaction targets, and movement direction."
        )
    return (
        "Domain expert perspective: inspect domain-specific objects, actions, "
        "locations, and temporal changes visible in the frames."
    )


def _type_expert_text(question: dict[str, Any]) -> str:
    family = _infer_question_family(question)
    if family == "prediction":
        return (
            "Question-type expert checklist: for prediction, focus on the "
            "latest visible state and the most likely immediate next event."
        )
    if family == "counting":
        return (
            "Question-type expert checklist: for counting, count distinct "
            "visible objects or events across the full sampled sequence."
        )
    if family == "localization":
        return (
            "Question-type expert checklist: for localization, match the "
            "target object or event to the best spatial or temporal option."
        )
    return (
        "Question-type expert checklist: for identification, recognize the "
        "object, action, entity, or absent item best supported by the frames."
    )


def _infer_question_family(question: dict[str, Any]) -> str:
    combined_text = " ".join(
        [
            str(question.get("question_text", "")),
            str(question.get("question_type", "")),
            str(question.get("primary_category", "")),
        ]
    ).lower()
    if "next" in combined_text or "prediction" in combined_text:
        return "prediction"
    if "count" in combined_text or "how many" in combined_text:
        return "counting"
    if (
        "localization" in combined_text
        or "located" in combined_text
        or "region" in combined_text
        or "timestamp" in combined_text
    ):
        return "localization"
    return "identification"
