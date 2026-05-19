"""Frame chunking helpers.

``max_frames`` is a per-call cap, not a runner-level sampling policy.
"""

from __future__ import annotations

from typing import Any


def chunk_frames(
    paths: list[str],
    max_frames_per_call: int,
    *,
    indices: list[int] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Split ordered frames into chronological chunks for VLM calls."""

    if indices is None:
        indices = list(range(len(paths)))
    if len(indices) != len(paths):
        raise ValueError("indices must match paths length.")

    if not paths:
        return [], {
            "strategy": "no_effective_frames",
            "max_frames_per_call": max_frames_per_call,
            "num_chunks": 0,
            "num_frames": 0,
        }

    chunk_size = len(paths) if max_frames_per_call <= 0 else max_frames_per_call
    chunks: list[dict[str, Any]] = []
    for start in range(0, len(paths), chunk_size):
        end = min(start + chunk_size, len(paths))
        chunks.append(
            {
                "chunk_index": len(chunks),
                "image_paths": paths[start:end],
                "original_indices": indices[start:end],
                "start_offset": start,
                "end_offset_exclusive": end,
            }
        )

    return chunks, {
        "strategy": "chronological_chunks",
        "max_frames_per_call": max_frames_per_call,
        "num_chunks": len(chunks),
        "num_frames": len(paths),
        "chunk_sizes": [len(chunk["image_paths"]) for chunk in chunks],
    }


def uniform_sample_frames(
    paths: list[str],
    max_frames_per_call: int,
    *,
    indices: list[int] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Uniformly subsample ordered frames to fit one MCQ call."""

    if indices is None:
        indices = list(range(len(paths)))
    if len(indices) != len(paths):
        raise ValueError("indices must match paths length.")

    if not paths:
        empty = {
            "image_paths": [],
            "original_indices": [],
            "sampled_offsets": [],
            "start_offset": 0,
            "end_offset_exclusive": 0,
        }
        return empty, {
            "strategy": "no_effective_frames",
            "max_frames_per_call": max_frames_per_call,
            "num_frames": 0,
            "num_sampled_frames": 0,
        }

    if max_frames_per_call <= 0 or len(paths) <= max_frames_per_call:
        full = {
            "image_paths": list(paths),
            "original_indices": list(indices),
            "sampled_offsets": list(range(len(paths))),
            "start_offset": 0,
            "end_offset_exclusive": len(paths),
        }
        return full, {
            "strategy": "all_effective_frames",
            "max_frames_per_call": max_frames_per_call,
            "num_frames": len(paths),
            "num_sampled_frames": len(paths),
            "sampled_offsets": list(range(len(paths))),
        }

    if max_frames_per_call == 1:
        sampled_offsets = [len(paths) // 2]
    else:
        sampled_offsets = [
            int(round(i * (len(paths) - 1) / (max_frames_per_call - 1)))
            for i in range(max_frames_per_call)
        ]
    sampled_offsets = sorted(dict.fromkeys(sampled_offsets))
    sampled_paths = [paths[idx] for idx in sampled_offsets]
    sampled_indices = [indices[idx] for idx in sampled_offsets]
    sampled = {
        "image_paths": sampled_paths,
        "original_indices": sampled_indices,
        "sampled_offsets": sampled_offsets,
        "start_offset": sampled_offsets[0],
        "end_offset_exclusive": sampled_offsets[-1] + 1,
    }
    return sampled, {
        "strategy": "uniform_sample",
        "max_frames_per_call": max_frames_per_call,
        "num_frames": len(paths),
        "num_sampled_frames": len(sampled_paths),
        "sampled_offsets": sampled_offsets,
        "sampled_original_indices": sampled_indices,
    }
