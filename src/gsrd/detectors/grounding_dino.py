from __future__ import annotations

from dataclasses import dataclass

import torch
from PIL.Image import Image

from gsrd.detectors.base import OpenVocabDetector
from gsrd.detectors.types import Detection


@dataclass
class GroundingDinoDetector(OpenVocabDetector):
    name: str
    model_id: str
    device: str = "cuda"
    score_threshold: float = 0.05
    max_detections: int = 300

    def __post_init__(self) -> None:
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        self._device = self.device if torch.cuda.is_available() and self.device.startswith("cuda") else "cpu"
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(self.model_id)
        self.model.to(self._device)
        self.model.eval()

    def predict(self, image: Image, terms: list[str]) -> list[Detection]:
        if not terms:
            return []
        text_queries = [[t for t in terms]]
        inputs = self.processor(images=image, text=text_queries, return_tensors="pt").to(self._device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=self.score_threshold,
            text_threshold=0.2,
            target_sizes=[image.size[::-1]],
        )[0]

        detections: list[Detection] = []
        labels = results.get("labels", [])
        scores = results.get("scores", [])
        boxes = results.get("boxes", [])

        for label_obj, score_obj, box_obj in zip(labels, scores, boxes):
            if len(detections) >= self.max_detections:
                break
            label = str(label_obj)
            score = float(score_obj)
            x1, y1, x2, y2 = [float(v) for v in box_obj.tolist()]
            detections.append(
                Detection(
                    term=label,
                    score=score,
                    bbox_xyxy=(x1, y1, x2, y2),
                )
            )
        return detections

    def version(self) -> str:
        import transformers

        return f"{self.model_id}@transformers-{transformers.__version__}"
