"""Resolve per-question effective sampling FPS with dataset-specific overrides."""

from __future__ import annotations

from typing import Any

from industry_infer.config import RunConfig


def resolve_effective_sampling_fps(
    question: dict[str, Any],
    cfg: RunConfig,
) -> tuple[float, dict[str, Any]]:
    """Use dataset-specific FPS first, else fall back to the config default."""

    dataset_name = str(question.get("dataset", "")).strip()
    if dataset_name in cfg.dataset_sampling_fps:
        fps = cfg.dataset_sampling_fps[dataset_name]
        return fps, {
            "source": "dataset_default",
            "dataset": dataset_name,
            "effective_sampling_fps": fps,
        }
    return cfg.effective_sampling_fps, {
        "source": "config_default",
        "dataset": dataset_name,
        "effective_sampling_fps": cfg.effective_sampling_fps,
    }
