from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from PIL.Image import Image

from gsrd.detectors.types import Detection


class OpenVocabDetector(ABC):
    """Adapter contract for fixed pretrained open-vocabulary detectors."""

    name: str
    model_id: str

    @abstractmethod
    def predict(self, image: Image, terms: list[str]) -> list[Detection]:
        raise NotImplementedError

    @abstractmethod
    def version(self) -> str:
        raise NotImplementedError

    def close(self) -> None:
        return None

    def __enter__(self) -> "OpenVocabDetector":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
