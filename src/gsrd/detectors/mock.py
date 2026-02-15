from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
from PIL.Image import Image

from gsrd.detectors.base import OpenVocabDetector
from gsrd.detectors.types import Detection
from gsrd.utils.hashing import stable_hash

TERM_MODIFIERS = {
    "occluded",
    "distant",
    "tiny",
    "blurry",
    "side",
    "view",
    "partial",
    "instance",
    "target",
}

DISTRACTOR_TOKENS = {
    "background",
    "texture",
    "marking",
    "shadow",
    "patch",
    "window",
    "reflection",
    "tree",
    "canopy",
    "building",
    "facade",
    "camera",
    "flare",
    "pavement",
    "pattern",
    "glass",
    "glare",
    "motion",
    "blur",
}

CLASS_KEYWORDS: dict[str, set[str]] = {
    "car": {"car", "sedan", "hatchback", "automobile", "vehicle"},
    "truck": {"truck", "pickup", "cargo", "heavy", "vehicle"},
    "person": {"person", "pedestrian", "human", "adult", "walking", "standing"},
    "rider": {"rider", "cyclist", "motor", "bike"},
    "traffic light": {"traffic", "light", "signal", "intersection"},
}

SHARED_GROUP_HINTS: dict[str, set[str]] = {
    "car": {"road"},
    "truck": {"road"},
    "person": {"human"},
    "rider": {"human"},
    "traffic light": {"control"},
}


@dataclass
class MockDetector(OpenVocabDetector):
    name: str = "mock"
    model_id: str = "mock-v1"
    score_threshold: float = 0.05
    max_detections: int = 50

    def _bbox_from_mask(self, mask: np.ndarray) -> tuple[float, float, float, float] | None:
        ys, xs = np.where(mask)
        if ys.size < 40:
            return None
        x1, x2 = int(xs.min()), int(xs.max())
        y1, y2 = int(ys.min()), int(ys.max())
        if x2 <= x1 or y2 <= y1:
            return None
        return float(x1), float(y1), float(x2), float(y2)

    def _normalize_term(self, term: str) -> str:
        return " ".join(term.lower().replace("_", " ").replace("-", " ").split())

    def _toy_color_detections(self, image: Image) -> list[tuple[str, tuple[float, float, float, float], float]]:
        arr = np.asarray(image.convert("RGB"), dtype=np.uint8)
        r = arr[..., 0]
        g = arr[..., 1]
        b = arr[..., 2]

        class_masks: list[tuple[str, np.ndarray]] = [
            ("car", (r > 150) & (g < 110) & (b < 132)),
            ("truck", (r > 165) & (g >= 120) & (g < 210) & (b < 150)),
            ("person", (g > 140) & (r < 150) & (b < 155)),
            ("rider", (r > 118) & (b > 130) & (g < 132)),
            ("traffic light", (b > 160) & (r < 110) & (g < 165)),
        ]

        dets: list[tuple[str, tuple[float, float, float, float], float]] = []
        for cls, mask in class_masks:
            bbox = self._bbox_from_mask(mask)
            if bbox is None:
                continue
            x1, y1, x2, y2 = bbox
            area = max(1.0, (x2 - x1) * (y2 - y1))
            conf = min(0.995, max(0.55, 0.62 + min(0.28, area / 52000.0)))
            dets.append((cls, bbox, conf))
        return dets

    def _term_affinity(self, class_name: str, term: str) -> float:
        normalized = self._normalize_term(term)
        tokens = set(normalized.split())

        score = 0.0
        for kw in CLASS_KEYWORDS.get(class_name, set()):
            if kw in tokens or kw in normalized:
                score += 1.0
        for kw in SHARED_GROUP_HINTS.get(class_name, set()):
            if kw in tokens or kw in normalized:
                score += 0.45

        other_kw = set().union(*[v for k, v in CLASS_KEYWORDS.items() if k != class_name])
        foreign = len(tokens.intersection(other_kw))
        if foreign > 0:
            score -= 0.22 * float(foreign)

        for tok in TERM_MODIFIERS:
            if tok in tokens:
                score -= 0.18
        for tok in DISTRACTOR_TOKENS:
            if tok in tokens:
                score -= 0.75

        return score

    def _term_is_distractor(self, term: str) -> bool:
        normalized = self._normalize_term(term)
        tokens = set(normalized.split())
        if tokens.intersection(DISTRACTOR_TOKENS):
            return True
        combined = set().union(*CLASS_KEYWORDS.values())
        return bool(tokens.intersection({"background", "unknown"})) and not bool(tokens.intersection(combined))

    def _select_vocab_term(self, class_name: str, terms: list[str], rng: random.Random) -> tuple[str | None, float]:
        if not terms:
            return None, 0.0
        scored = sorted(
            [(self._term_affinity(class_name, term), term) for term in terms],
            reverse=True,
        )
        best_score, best_term = scored[0]
        second_score, second_term = scored[1] if len(scored) > 1 else (-1e6, best_term)

        if best_score < 0.30 and rng.random() < 0.82:
            return None, 0.0

        if (
            len(scored) > 1
            and (best_score - second_score) < 0.15
            and second_score > -0.30
            and rng.random() < 0.10
        ):
            chosen_score, chosen_term = second_score, second_term
        else:
            chosen_score, chosen_term = best_score, best_term

        lexical_quality = max(0.08, min(1.0, 0.42 + 0.24 * chosen_score))
        return chosen_term, lexical_quality

    def _deterministic_rng(self, image: Image, terms: list[str]) -> random.Random:
        thumb = np.asarray(image.resize((16, 16)).convert("RGB"), dtype=np.uint8)
        signature = {
            "model": self.model_id,
            "size": image.size,
            "mean_rgb": [int(x) for x in thumb.mean(axis=(0, 1)).tolist()],
            "terms": sorted(self._normalize_term(t) for t in terms),
        }
        seed = int(stable_hash(signature, n_chars=16), 16) & 0xFFFFFFFF
        return random.Random(seed)

    def predict(self, image: Image, terms: list[str]) -> list[Detection]:
        color_dets = self._toy_color_detections(image)
        if color_dets:
            normalized_terms = sorted({self._normalize_term(t) for t in terms if t.strip()})
            rng = self._deterministic_rng(image, normalized_terms)
            luminance = float(np.asarray(image.convert("L"), dtype=np.uint8).mean()) / 255.0
            scene_quality = min(1.0, max(0.25, 0.55 + 0.9 * luminance))

            detections: list[Detection] = []
            for class_name, bbox, base_conf in color_dets:
                chosen_term, lexical_quality = self._select_vocab_term(class_name, normalized_terms, rng)
                if chosen_term is None:
                    continue
                jitter = rng.uniform(-0.03, 0.03)
                score = max(0.0, min(0.995, (base_conf * lexical_quality * scene_quality) + jitter))
                if score < self.score_threshold:
                    continue
                detections.append(Detection(term=chosen_term, score=score, bbox_xyxy=bbox))

            distractors = [t for t in normalized_terms if self._term_is_distractor(t)]
            if distractors and rng.random() < min(0.85, 0.20 + 0.09 * len(distractors)):
                width, height = image.size
                for _ in range(min(2, max(1, len(distractors) // 2))):
                    x1 = rng.uniform(0, width * 0.78)
                    y1 = rng.uniform(0, height * 0.78)
                    w = rng.uniform(20, width * 0.25)
                    h = rng.uniform(20, height * 0.25)
                    detections.append(
                        Detection(
                            term=rng.choice(distractors),
                            score=rng.uniform(max(self.score_threshold, 0.08), 0.36),
                            bbox_xyxy=(x1, y1, min(width - 1, x1 + w), min(height - 1, y1 + h)),
                        )
                    )

            detections.sort(key=lambda d: d.score, reverse=True)
            return detections[: self.max_detections]

        # Fallback random behavior for non-synthetic toy images.
        rng = self._deterministic_rng(image, terms)
        width, height = image.size
        detections: list[Detection] = []
        for term in sorted({self._normalize_term(t) for t in terms}):
            for _ in range(rng.randint(0, 2)):
                x1 = rng.uniform(0, width * 0.75)
                y1 = rng.uniform(0, height * 0.75)
                w = rng.uniform(15, width * 0.3)
                h = rng.uniform(15, height * 0.3)
                score = rng.uniform(self.score_threshold, 1.0)
                detections.append(
                    Detection(
                        term=term,
                        score=score,
                        bbox_xyxy=(x1, y1, min(width - 1, x1 + w), min(height - 1, y1 + h)),
                    )
                )
                if len(detections) >= self.max_detections:
                    break
            if len(detections) >= self.max_detections:
                break
        return detections

    def version(self) -> str:
        return self.model_id
