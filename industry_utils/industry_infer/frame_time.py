"""Per-frame timestamps using spacing ``1 / effective_sampling_fps`` seconds."""

from __future__ import annotations

import re
from typing import Any

_ZERO_START_HINT = re.compile(
    r"(?:from\s+0(?:\.0+)?s\b|segment\s+from\s+0(?:\.0+)?s\b|"
    r"within\s+segment\s+0(?:\.0+)?s\s*-\s*[\d.]+s\b|"
    r"\b0\.00s\b|\bfrom\s+0s\b|\bat\s+0(?:\.0+)?s\b)",
    re.IGNORECASE,
)


def seconds_per_slot(effective_sampling_fps: float) -> float:
    if effective_sampling_fps <= 0:
        raise ValueError("effective_sampling_fps must be positive.")
    return 1.0 / effective_sampling_fps


def first_slot_starts_at_zero(
    question_text: str,
    options: list[str],
) -> bool:
    blob = question_text + "\n" + "\n".join(str(option) for option in options)
    return bool(_ZERO_START_HINT.search(blob))


def build_frame_prefix_texts(
    num_frames: int,
    effective_sampling_fps: float,
    *,
    question_text: str,
    options: list[str],
) -> tuple[list[str], dict[str, Any]]:
    """One prefix per frame using point timestamps on the effective grid."""

    slot = seconds_per_slot(effective_sampling_fps)
    start_zero = first_slot_starts_at_zero(question_text, options)
    timestamp0 = 0.0 if start_zero else slot

    prefixes = []
    timestamps = []
    for index in range(num_frames):
        timestamp_sec = timestamp0 + index * slot
        timestamps.append(timestamp_sec)
        prefixes.append(
            (
                f"Image {index + 1}/{num_frames} — approximately at "
                f"{timestamp_sec:.1f} s in the clip."
            )
        )
    meta = {
        "effective_sampling_fps": effective_sampling_fps,
        "seconds_per_step": slot,
        "first_slot_starts_at_zero": start_zero,
        "first_frame_timestamp_sec": timestamp0,
        "frame_timestamps_sec": timestamps,
    }
    return prefixes, meta


def build_timing_legend(num_frames: int, effective_sampling_fps: float) -> str:
    slot = seconds_per_slot(effective_sampling_fps)
    return "\n".join(
        [
            "Time reference: The images below are in chronological order (earliest to latest).",
            (
                f"Assume they were sampled at {effective_sampling_fps:g} FPS "
                f"along the clip: adjacent images are about {slot:.1f} s apart."
            ),
            f"You are given {num_frames} image(s); use the per-image timestamps to answer.",
        ],
    )
