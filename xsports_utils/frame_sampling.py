"""Default frame sampling for Xsports inference."""

from __future__ import annotations

from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


_EXPERIMENT_ROOT = Path(__file__).resolve().parent / "outputs"


@dataclass(frozen=True)
class ExperimentConfig:
    """Configuration for the fixed default frame setup."""

    name: str
    prompt_mode: str
    sampling_strategy: str
    max_frames: int
    description: str


@dataclass(frozen=True)
class ExperimentPaths:
    """Filesystem paths for one experiment run."""

    exp_dir: Path
    prediction_path: Path
    summary_path: Path
    progress_path: Path
    config_path: Path


_DEFAULT_EXPERIMENT = ExperimentConfig(
    name="F0",
    prompt_mode="direct",
    sampling_strategy="uniform-8",
    max_frames=8,
    description="direct prompt + uniform-8",
)


def available_experiments() -> tuple[str, ...]:
    """Returns the only supported frame experiment id."""

    return (_DEFAULT_EXPERIMENT.name,)


def default_experiment_config() -> ExperimentConfig:
    """Returns the fixed default frame configuration."""

    return _DEFAULT_EXPERIMENT


def get_experiment_config(experiment_name: str) -> ExperimentConfig:
    """Returns the fixed default frame configuration."""

    normalized_name = experiment_name.upper()
    if normalized_name != _DEFAULT_EXPERIMENT.name:
        raise ValueError(
            "Frame experiment logic has been simplified to the default "
            f"configuration only: {_DEFAULT_EXPERIMENT.name}."
        )
    return _DEFAULT_EXPERIMENT


def experiment_config_to_dict(config: ExperimentConfig) -> dict[str, Any]:
    """Converts an experiment config to a JSON-serializable dict."""

    return asdict(config)


def build_experiment_paths(
    experiment_name: str,
    exp_root: Path = _EXPERIMENT_ROOT,
) -> ExperimentPaths:
    """Builds standard output paths under the experiment directory."""

    exp_dir = exp_root / experiment_name
    return ExperimentPaths(
        exp_dir=exp_dir,
        prediction_path=exp_dir / "predictions.json",
        summary_path=exp_dir / "summary.json",
        progress_path=exp_dir / "progress.jsonl",
        config_path=exp_dir / "config.json",
    )


def sample_frame_pack(
    image_paths: list[str],
    *,
    max_frames: int | None = None,
) -> tuple[list[str], dict[str, Any]]:
    """Uniformly samples up to the default maximum number of frames."""

    target_count = max_frames or _DEFAULT_EXPERIMENT.max_frames
    if target_count <= 0:
        raise ValueError("max_frames must be positive.")
    if not image_paths:
        return [], _build_metadata(target_count, [], 0)

    indices = _uniform_indices(len(image_paths), target_count)
    selected_paths = [image_paths[index] for index in indices]
    metadata = _build_metadata(target_count, indices, len(image_paths))
    return selected_paths, metadata


def _build_metadata(
    max_frames: int,
    selected_indices: list[int],
    original_count: int,
) -> dict[str, Any]:
    return {
        "strategy": _DEFAULT_EXPERIMENT.sampling_strategy,
        "max_frames": max_frames,
        "original_count": original_count,
        "selected_count": len(selected_indices),
        "selected_indices": selected_indices,
    }


def _uniform_indices(total_count: int, target_count: int) -> list[int]:
    if total_count <= target_count:
        return list(range(total_count))
    if target_count == 1:
        return [0]

    step = (total_count - 1) / (target_count - 1)
    indices = {round(step * index) for index in range(target_count)}
    ordered = sorted(indices)
    if len(ordered) == target_count:
        return ordered

    for index in range(total_count):
        if index not in indices:
            ordered.append(index)
        if len(ordered) == target_count:
            break
    return sorted(ordered)
