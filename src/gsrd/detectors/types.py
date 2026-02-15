from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Detection:
    term: str
    score: float
    bbox_xyxy: tuple[float, float, float, float]
