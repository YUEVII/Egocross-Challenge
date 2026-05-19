"""Resolve testbed image paths against ``data_root``."""

from __future__ import annotations

from pathlib import Path


def resolve_image_path(raw_path: str, data_root: Path) -> str:
    image_path = Path(raw_path)
    if image_path.is_absolute() and image_path.is_file():
        return str(image_path)

    relative_path = raw_path.lstrip("/")
    candidate = data_root / relative_path
    if candidate.is_file():
        return str(candidate)

    path_parts = Path(relative_path).parts
    if path_parts and path_parts[0] == data_root.name:
        deduped = data_root / Path(*path_parts[1:])
        if deduped.is_file():
            return str(deduped)
    return str(candidate)


def image_paths_for_question(
    question: dict,
    data_root: Path,
) -> list[str]:
    paths: list[str] = []
    for raw_path in question.get("video_path", []):
        if isinstance(raw_path, str):
            paths.append(resolve_image_path(raw_path, data_root))
    return paths
