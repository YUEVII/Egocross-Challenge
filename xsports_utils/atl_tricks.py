"""ATL helpers for the fixed action-temporal-localization anchor mode."""

from __future__ import annotations

TASK_TYPE_ACTION_TEMPORAL_LOCALIZATION = "action-temporal-localization"
XSPORTS_ATL_DURATION_SECONDS = 30.0


def question_type_from_question_id(question_id: str) -> str:
    """Parses the task suffix after ``_q<n>_`` from a question id."""

    if "_q" not in question_id:
        return "unknown"
    _, _, suffix = question_id.partition("_q")
    _, separator, question_type = suffix.partition("_")
    if not separator or not question_type:
        return "unknown"
    return question_type


def is_action_temporal_localization_question(question: dict[str, object]) -> bool:
    """True only for ATL items inferred from ``question_id``."""

    qid = str(question.get("question_id", "")).strip()
    return question_type_from_question_id(qid) == TASK_TYPE_ACTION_TEMPORAL_LOCALIZATION


def estimate_video_duration_seconds(
    question: dict[str, object],
    override_seconds: float | None = None,
) -> float | None:
    """Returns the fixed Xsports ATL duration used by the benchmark."""

    if override_seconds is not None:
        return override_seconds
    if is_action_temporal_localization_question(question):
        return XSPORTS_ATL_DURATION_SECONDS
    return None


def build_atl_timestamp_instruction(
    question: dict[str, object],
    num_frames_shown: int,
    *,
    duration_seconds: float | None = None,
) -> str:
    """Builds the fixed ATL timestamp anchor instruction."""

    duration_seconds = estimate_video_duration_seconds(
        question,
        override_seconds=duration_seconds,
    )
    chunks = [
        "The frames are uniformly sampled from the full video segment.",
        "This Xsports ATL segment is about 30 seconds long.",
        (
            f"You are given {num_frames_shown} frames below in chronological "
            "order. Use frame_index from 1 to "
            f"{num_frames_shown} counting from the first listed frame."
        ),
        "Treat each shown frame as the center of one equal time bin.",
        "Approximate each frame timestamp with:",
        "timestamp_seconds ~= total_video_duration_seconds x (frame_index - 0.5)",
        "/ total_number_of_frames_I_give_you",
        "",
        "Compare the four candidate timestamps against this full 30-second scale.",
    ]
    if duration_seconds is not None:
        chunks.insert(2, f"(Use total_video_duration_seconds = {duration_seconds:.1f}.)")
    return "\n".join(chunks)
