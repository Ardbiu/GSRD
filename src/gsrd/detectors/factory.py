from __future__ import annotations

from gsrd.config.schema import DetectorConfig
from gsrd.detectors.base import OpenVocabDetector
from gsrd.detectors.grounding_dino import GroundingDinoDetector
from gsrd.detectors.mock import MockDetector
from gsrd.detectors.owlvit import OwlViTDetector


def create_detector(cfg: DetectorConfig) -> OpenVocabDetector:
    if cfg.name == "grounding_dino":
        model_id = cfg.model_id or "IDEA-Research/grounding-dino-base"
        return GroundingDinoDetector(
            name=cfg.name,
            model_id=model_id,
            device=cfg.device,
            score_threshold=cfg.score_threshold,
            max_detections=cfg.max_detections,
        )
    if cfg.name == "owlvit":
        model_id = cfg.model_id or "google/owlvit-base-patch32"
        return OwlViTDetector(
            name=cfg.name,
            model_id=model_id,
            device=cfg.device,
            score_threshold=cfg.score_threshold,
            max_detections=cfg.max_detections,
        )
    if cfg.name == "mock":
        return MockDetector(
            score_threshold=cfg.score_threshold,
            max_detections=cfg.max_detections,
        )
    raise ValueError(f"Unsupported detector: {cfg.name}")
