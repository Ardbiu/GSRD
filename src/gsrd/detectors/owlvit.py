from __future__ import annotations

from dataclasses import dataclass

import torch
from PIL.Image import Image

from gsrd.detectors.base import OpenVocabDetector
from gsrd.detectors.types import Detection


@dataclass
class OwlViTDetector(OpenVocabDetector):
    name: str
    model_id: str
    device: str = "cuda"
    score_threshold: float = 0.05
    max_detections: int = 300

    def __post_init__(self) -> None:
        from transformers import AutoProcessor, OwlViTForObjectDetection

        self._device = self.device if torch.cuda.is_available() and self.device.startswith("cuda") else "cpu"
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.model = OwlViTForObjectDetection.from_pretrained(self.model_id).to(self._device)
        self.model.eval()

    def predict(self, image: Image, terms: list[str]) -> list[Detection]:
        if not terms:
            return []
        queries = [terms]
        inputs = self.processor(text=queries, images=image, return_tensors="pt").to(self._device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        target_sizes = torch.tensor([image.size[::-1]], device=self._device)
        results = self.processor.post_process_object_detection(
            outputs=outputs,
            threshold=self.score_threshold,
            target_sizes=target_sizes,
        )[0]

        detections: list[Detection] = []
        for score, label_idx, box in zip(results["scores"], results["labels"], results["boxes"]):
            if len(detections) >= self.max_detections:
                break
            idx = int(label_idx)
            term = terms[idx] if idx < len(terms) else f"term-{idx}"
            x1, y1, x2, y2 = [float(v) for v in box.tolist()]
            detections.append(
                Detection(term=term, score=float(score), bbox_xyxy=(x1, y1, x2, y2))
            )
        return detections

    def version(self) -> str:
        import transformers

        return f"{self.model_id}@transformers-{transformers.__version__}"
