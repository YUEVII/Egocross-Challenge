"""Load and validate YAML configs for Industry / ENIGMA inference."""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import Any

import yaml

RUN_FLAG_TO_QUESTION_TYPE: dict[str, str] = {
    "action_temporal_localization": "action temporal localization",
    "dominant_held_object_identification": "dominant held-object identification",
    "next_interaction_prediction": "next interaction prediction",
    "object_counting": "object counting",
    "object_not_visible_identification": "object not visible identification",
    "object_spatial_localization": "object spatial localization",
}


@dataclass
class RunConfig:
    datasets: list[str]
    repo_root: Path
    dataset_json: Path
    data_root: Path
    use_sft: bool
    model_path_base: Path
    model_path_sft_local: Path
    device: str
    max_frames: int
    effective_sampling_fps: float
    dataset_sampling_fps: dict[str, float]
    max_new_tokens: int
    dtype: str
    allow_remote_model: bool
    question_type_enabled: dict[str, bool]
    question_type_settings: dict[str, dict[str, Any]]
    output_dir: Path
    config_yaml_path: Path
    raw: dict[str, Any] = field(default_factory=dict)


def _resolve_path(repo_root: Path, path_like: str | Path) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path
    return (repo_root / path).resolve()


def _available_datasets(dataset_json: Path) -> list[str]:
    rows = yaml.safe_load(dataset_json.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError(f"{dataset_json} must contain a JSON list.")
    names = sorted(
        {
            str(row.get("dataset", "")).strip()
            for row in rows
            if isinstance(row, dict) and str(row.get("dataset", "")).strip()
        }
    )
    if not names:
        raise ValueError(f"No datasets found in {dataset_json}.")
    return names


def _parse_datasets(raw_dataset: Any, dataset_json: Path) -> list[str]:
    available = set(_available_datasets(dataset_json))
    if isinstance(raw_dataset, str):
        token = raw_dataset.strip()
        if not token:
            raise ValueError("dataset must not be empty.")
        if token.lower() == "all":
            return sorted(available)
        if token not in available:
            raise ValueError(
                f"Unknown dataset {token!r}; available datasets: {sorted(available)}"
            )
        return [token]

    if isinstance(raw_dataset, list):
        datasets = []
        for item in raw_dataset:
            if not isinstance(item, str) or not item.strip():
                raise ValueError("dataset list entries must be non-empty strings.")
            token = item.strip()
            if token not in available:
                raise ValueError(
                    f"Unknown dataset {token!r}; available datasets: {sorted(available)}"
                )
            datasets.append(token)
        deduped = list(dict.fromkeys(datasets))
        if not deduped:
            raise ValueError("dataset list must not be empty.")
        return deduped

    raise ValueError("dataset must be a string, a list of strings, or 'all'.")


def _default_question_type_setting(qt_label: str) -> dict[str, Any]:
    if qt_label == "action temporal localization":
        return {
            "run": True,
            "use_custom_strategy": True,
            "strategy": "temporal_option_frame_pair_mcq",
        }
    if qt_label == "dominant held-object identification":
        return {
            "run": True,
            "use_custom_strategy": True,
            "strategy": "dominant_held_object_frame_weighted_mean",
        }
    if qt_label == "object counting":
        return {
            "run": True,
            "use_custom_strategy": True,
            "strategy": "object_counting_option_guided",
            "counting_object_labels": [],
        }
    if qt_label == "object not visible identification":
        return {
            "run": True,
            "use_custom_strategy": True,
            "strategy": "not_visible_mcq_logprob",
        }
    if qt_label == "object spatial localization":
        # Keep the region-MCQ path as the default baseline.
        # Experimental coordinate regression can be enabled with
        # `question_timepoint_point_coord_mcq` in YAML.
        return {
            "run": True,
            "use_custom_strategy": True,
            "strategy": "question_timepoint_neighborhood_mcq",
            "timepoint_neighbor_radius": 3,
            "point_output_count": 1,
        }
    if qt_label == "next interaction prediction":
        return {
            "run": True,
            "use_custom_strategy": True,
            "strategy": "next_interaction_tail_option_guided",
        }
    return {
        "run": True,
        "use_custom_strategy": False,
        "strategy": "direct_mcq",
    }


def _parse_question_type_settings(
    qt_raw: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    settings: dict[str, dict[str, Any]] = {}
    for flag_key, qt_label in RUN_FLAG_TO_QUESTION_TYPE.items():
        default = _default_question_type_setting(qt_label)
        raw_value = qt_raw.get(flag_key, qt_raw.get(qt_label, default))
        if isinstance(raw_value, bool):
            merged = dict(default)
            merged["run"] = raw_value
        elif isinstance(raw_value, dict):
            merged = dict(default)
            merged.update(raw_value)
            merged["run"] = bool(merged.get("run", True))
            merged["use_custom_strategy"] = bool(
                merged.get("use_custom_strategy", False),
            )
            merged["strategy"] = str(merged.get("strategy", "direct_mcq"))
            if "timepoint_neighbor_radius" in merged:
                merged["timepoint_neighbor_radius"] = int(
                    merged.get("timepoint_neighbor_radius", 3),
                )
            if "point_output_count" in merged:
                merged["point_output_count"] = int(merged.get("point_output_count", 1))
                if merged["point_output_count"] < 1:
                    raise ValueError("point_output_count must be >= 1.")
            if "counting_object_labels" in merged:
                merged["counting_object_labels"] = [
                    str(label).strip()
                    for label in merged.get("counting_object_labels", [])
                    if str(label).strip()
                ]
        else:
            raise ValueError(
                f"question_types.{flag_key} must be a bool or mapping, got "
                f"{type(raw_value).__name__}.",
            )
        settings[qt_label] = merged
    return settings


def load_run_config(yaml_path: Path, package_root: Path | None = None) -> RunConfig:
    """Load YAML relative to the packaged Industry utility directory."""

    yaml_path = yaml_path.resolve()
    raw = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{yaml_path} must be a YAML mapping.")

    if package_root is None:
        package_root = yaml_path.parent.parent.resolve()

    repo_root_raw = raw.get("repo_root")
    if repo_root_raw:
        repo_root = Path(repo_root_raw).expanduser().resolve()
    else:
        repo_root = package_root.parent.resolve()

    paths = raw.get("paths") or {}
    if not isinstance(paths, dict):
        raise ValueError("paths must be a mapping.")

    dataset_json = _resolve_path(repo_root, paths.get("dataset_json", ""))
    data_root = _resolve_path(repo_root, paths.get("data_root", ""))
    model = raw.get("model") or {}
    use_sft = bool(model.get("use_sft", False))
    path_base = _resolve_path(repo_root, model.get("path_base", ""))
    path_sft = _resolve_path(repo_root, model.get("path_sft_local", ""))

    inference = raw.get("inference") or {}
    qt_raw = raw.get("question_types") or {}
    if not isinstance(qt_raw, dict):
        raise ValueError("question_types must be a mapping.")

    question_type_settings = _parse_question_type_settings(qt_raw)
    question_type_enabled = {
        qt_label: bool(question_type_settings[qt_label]["run"])
        for qt_label in RUN_FLAG_TO_QUESTION_TYPE.values()
    }

    output = raw.get("output") or {}
    out_dir_raw = output.get("dir", "output/base")
    output_dir = Path(out_dir_raw)
    if not output_dir.is_absolute():
        output_dir = (package_root / output_dir).resolve()

    dataset_fps_map: dict[str, float] = {}
    raw_dataset_fps = inference.get("dataset_sampling_fps")
    if raw_dataset_fps is not None:
        if not isinstance(raw_dataset_fps, dict):
            raise ValueError("inference.dataset_sampling_fps must be a mapping or omitted.")
        dataset_fps_map.update(
            {str(dataset_name): float(fps) for dataset_name, fps in raw_dataset_fps.items()}
        )

    cfg = RunConfig(
        datasets=_parse_datasets(raw.get("dataset", "ENIGMA"), dataset_json),
        repo_root=repo_root,
        dataset_json=dataset_json,
        data_root=data_root,
        use_sft=use_sft,
        model_path_base=path_base,
        model_path_sft_local=path_sft,
        device=str(raw.get("device", "cuda:0")),
        max_frames=int(inference.get("max_frames", 15)),
        effective_sampling_fps=float(inference.get("effective_sampling_fps", 0.5)),
        dataset_sampling_fps=dataset_fps_map,
        max_new_tokens=int(inference.get("max_new_tokens", 16)),
        dtype=str(inference.get("dtype", "auto")),
        allow_remote_model=bool(inference.get("allow_remote_model", False)),
        question_type_enabled=question_type_enabled,
        question_type_settings=question_type_settings,
        output_dir=output_dir,
        config_yaml_path=yaml_path,
        raw=raw,
    )
    _validate_paths(cfg)
    return cfg


def _validate_paths(cfg: RunConfig) -> None:
    if not cfg.dataset_json.is_file():
        raise FileNotFoundError(f"dataset_json not found: {cfg.dataset_json}")
    if not cfg.data_root.is_dir():
        raise FileNotFoundError(f"data_root not a directory: {cfg.data_root}")
    chosen = cfg.model_path_sft_local if cfg.use_sft else cfg.model_path_base
    if not chosen.exists():
        raise FileNotFoundError(
            f"Model path not found (use_sft={cfg.use_sft}): {chosen}",
        )


def config_to_json_dict(cfg: RunConfig) -> dict[str, Any]:
    """Flatten config for ``config.json`` output."""

    return {
        "dataset": cfg.raw.get(
            "dataset",
            cfg.datasets[0] if len(cfg.datasets) == 1 else cfg.datasets,
        ),
        "datasets": cfg.datasets,
        "repo_root": str(cfg.repo_root),
        "config_yaml": str(cfg.config_yaml_path),
        "dataset_json": str(cfg.dataset_json),
        "data_root": str(cfg.data_root),
        "use_sft": cfg.use_sft,
        "model_path": str(
            cfg.model_path_sft_local if cfg.use_sft else cfg.model_path_base,
        ),
        "model_path_base": str(cfg.model_path_base),
        "model_path_sft_local": str(cfg.model_path_sft_local),
        "device": cfg.device,
        "max_frames": cfg.max_frames,
        "effective_sampling_fps": cfg.effective_sampling_fps,
        "dataset_sampling_fps": cfg.dataset_sampling_fps,
        "max_new_tokens": cfg.max_new_tokens,
        "dtype": cfg.dtype,
        "allow_remote_model": cfg.allow_remote_model,
        "question_type_enabled": cfg.question_type_enabled,
        "question_type_settings": cfg.question_type_settings,
        "output_dir": str(cfg.output_dir),
        "frame_time_rule": (
            "Per-frame labels use spacing 1/effective_sampling_fps. "
            "A dataset can override the base value through inference.dataset_sampling_fps. "
            "First frame timestamp starts at 0 s if question/options hint at clip-from-zero "
            "or within-segment-from-zero wording; otherwise the first timestamp starts one step later."
        ),
        "time_window_strip_rule": (
            "Remove time spans that match the redundant whole-clip heuristic, "
            "for either 'from … s to … s' or '(within segment …s-…s)' text."
        ),
        "frame_selection_rule": (
            "Runner keeps all effective frames. Non-whole time windows are expanded "
            "outward to the nearest available frame timestamps on both sides. "
            "max_frames is a per-VLM-call cap used by chunked decoders or direct uniform sampling."
        ),
    }
