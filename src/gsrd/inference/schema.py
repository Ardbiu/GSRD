from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from gsrd.constants import PREDICTION_FORMAT_VERSION


@dataclass(frozen=True)
class PredictionArtifactSpec:
    detector: str
    dataset: str
    granularity: str
    vocab_id: str

    def artifact_dir(self, cache_root: Path) -> Path:
        return (
            cache_root
            / "predictions"
            / self.detector
            / self.dataset
            / self.granularity
            / self.vocab_id
        )


def make_prediction_metadata(
    detector_name: str,
    detector_version: str,
    dataset_name: str,
    granularity: str,
    vocab: dict[str, Any],
    num_shards: int,
) -> dict[str, Any]:
    return {
        "format_version": PREDICTION_FORMAT_VERSION,
        "detector": {"name": detector_name, "version": detector_version},
        "dataset": dataset_name,
        "granularity": granularity,
        "vocab": {
            "vocab_id": vocab["vocab_id"],
            "terms": vocab["terms"],
            "class_to_term": vocab["class_to_term"],
            "term_to_classes": vocab["term_to_classes"],
        },
        "num_shards": num_shards,
    }
