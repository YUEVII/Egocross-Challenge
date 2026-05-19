"""Fixed no-shuffle helpers for the default Xsports setup."""

from __future__ import annotations

from dataclasses import asdict
from dataclasses import dataclass
from typing import Any


_LETTERS = ("A", "B", "C", "D")
_ZERO_COUNTS = {letter: 0 for letter in _LETTERS}


@dataclass(frozen=True)
class ShuffleConfig:
    """Configuration for the fixed no-shuffle setup."""

    name: str
    num_orders: int
    description: str


_DEFAULT_SHUFFLE = ShuffleConfig(
    name="S0",
    num_orders=1,
    description="no shuffle",
)


def available_shuffle_experiments() -> tuple[str, ...]:
    """Returns the only supported shuffle id."""

    return (_DEFAULT_SHUFFLE.name,)


def available_shuffle_aggregations() -> tuple[str, ...]:
    """Returns the legacy aggregation label kept for metadata compatibility."""

    return ("majority_vote",)


def default_shuffle_config() -> ShuffleConfig:
    """Returns the fixed no-shuffle configuration."""

    return _DEFAULT_SHUFFLE


def get_shuffle_config(shuffle_experiment: str) -> ShuffleConfig:
    """Returns the fixed no-shuffle configuration."""

    normalized_name = shuffle_experiment.upper()
    if normalized_name != _DEFAULT_SHUFFLE.name:
        raise ValueError(
            "Shuffle logic has been simplified to the default configuration "
            f"only: {_DEFAULT_SHUFFLE.name}."
        )
    return _DEFAULT_SHUFFLE


def shuffle_config_to_dict(config: ShuffleConfig) -> dict[str, Any]:
    """Converts a shuffle config to a JSON-serializable dict."""

    return asdict(config)


def build_shuffle_metrics(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    """Builds summary metadata after removing option shuffling."""

    final_counts = {letter: 0 for letter in _LETTERS}
    for row in predictions:
        answer = row.get("answer")
        if answer in final_counts:
            final_counts[answer] += 1

    total_predictions = sum(final_counts.values())
    final_prior = {
        letter: round(count / total_predictions, 6) if total_predictions else 0.0
        for letter, count in final_counts.items()
    }
    return {
        "model_raw_letter_prior": dict(_ZERO_COUNTS),
        "mapped_letter_prior": dict(_ZERO_COUNTS),
        "final_answer_prior": final_prior,
        "support_true_letter_prior": dict(_ZERO_COUNTS),
        "raw_letter_counts": dict(_ZERO_COUNTS),
        "mapped_letter_counts": dict(_ZERO_COUNTS),
        "final_answer_counts": final_counts,
        "support_true_letter_counts": dict(_ZERO_COUNTS),
        "total_shuffle_runs": 0,
        "calibration": "fixed default option order; shuffle calibration removed.",
    }
