"""Fixed no-retrieval helpers for the default Xsports setup."""

from __future__ import annotations

from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RetrievalConfig:
    """Configuration for the fixed no-retrieval setup."""

    name: str
    strategy: str
    num_examples: int
    description: str


_DEFAULT_RETRIEVAL = RetrievalConfig(
    name="R0",
    strategy="none",
    num_examples=0,
    description="no support",
)


def available_retrieval_experiments() -> tuple[str, ...]:
    """Returns the only supported retrieval id."""

    return (_DEFAULT_RETRIEVAL.name,)


def default_retrieval_config() -> RetrievalConfig:
    """Returns the fixed no-retrieval configuration."""

    return _DEFAULT_RETRIEVAL


def get_retrieval_config(retrieval_experiment: str) -> RetrievalConfig:
    """Returns the fixed no-retrieval configuration."""

    normalized_name = retrieval_experiment.upper()
    if normalized_name != _DEFAULT_RETRIEVAL.name:
        raise ValueError(
            "Retrieval experiment logic has been simplified to the default "
            f"configuration only: {_DEFAULT_RETRIEVAL.name}."
        )
    return _DEFAULT_RETRIEVAL


def retrieval_config_to_dict(config: RetrievalConfig) -> dict[str, Any]:
    """Converts a retrieval config to a JSON-serializable dict."""

    return asdict(config)


def load_support_index(_support_root: Path) -> list[Any]:
    """Retrieval has been removed, so the support index is always empty."""

    return []


def retrieve_support_examples(
    question: dict[str, Any],
    support_index: list[Any],
    config: RetrievalConfig,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Always returns no support examples plus default metadata."""

    del question
    del support_index
    metadata = {
        "experiment": config.name,
        "strategy": config.strategy,
        "requested_examples": config.num_examples,
        "selected_examples": 0,
        "selected_support_ids": [],
    }
    return [], metadata
