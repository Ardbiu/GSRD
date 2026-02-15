from __future__ import annotations

import contextlib
import io
import logging
import random
from collections import defaultdict
from copy import deepcopy
from typing import Any

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

LOGGER = logging.getLogger(__name__)


def _empty_stats() -> dict[str, float]:
    return {
        "mAP": 0.0,
        "mAP50": 0.0,
        "mAP75": 0.0,
        "mAP_small": 0.0,
        "mAP_medium": 0.0,
        "mAP_large": 0.0,
        "mAR_1": 0.0,
        "mAR_10": 0.0,
        "mAR_100": 0.0,
        "mAR_small": 0.0,
        "mAR_medium": 0.0,
        "mAR_large": 0.0,
    }


def run_coco_eval(
    gt_payload: dict[str, Any],
    detections: list[dict[str, Any]],
    image_ids: list[int] | None = None,
) -> dict[str, float]:
    if not gt_payload.get("images"):
        return _empty_stats()

    with contextlib.redirect_stdout(io.StringIO()):
        coco_gt = COCO()
        coco_gt.dataset = deepcopy(gt_payload)
        coco_gt.createIndex()

        if detections:
            coco_dt = coco_gt.loadRes(detections)
        else:
            dummy = {
                "image_id": int(gt_payload["images"][0]["id"]),
                "category_id": int(gt_payload["categories"][0]["id"]),
                "bbox": [0.0, 0.0, 1.0, 1.0],
                "score": 1e-8,
            }
            coco_dt = coco_gt.loadRes([dummy])

        coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
        if image_ids is not None:
            coco_eval.params.imgIds = [int(i) for i in image_ids]
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

    stats = coco_eval.stats.tolist()
    keys = [
        "mAP",
        "mAP50",
        "mAP75",
        "mAP_small",
        "mAP_medium",
        "mAP_large",
        "mAR_1",
        "mAR_10",
        "mAR_100",
        "mAR_small",
        "mAR_medium",
        "mAR_large",
    ]
    return {k: float(v) for k, v in zip(keys, stats)}


def bootstrap_coco_ci(
    gt_payload: dict[str, Any],
    detections: list[dict[str, Any]],
    num_samples: int,
    seed: int,
    metric_keys: list[str] | None = None,
) -> dict[str, dict[str, float]]:
    """
    Percentile bootstrap over images with replacement.

    We rebuild resampled COCO payloads to preserve duplicate image draws,
    rather than deduplicating sampled IDs.
    """
    metric_keys = metric_keys or ["mAP", "mAP50"]
    if num_samples <= 1:
        return {k: {"mean": 0.0, "ci_low": 0.0, "ci_high": 0.0} for k in metric_keys}

    images = gt_payload.get("images", [])
    if not images:
        return {k: {"mean": 0.0, "ci_low": 0.0, "ci_high": 0.0} for k in metric_keys}

    img_ids = [int(img["id"]) for img in images]
    anns_by_img: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for ann in gt_payload.get("annotations", []):
        anns_by_img[int(ann["image_id"])].append(ann)

    det_by_img: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for det in detections:
        det_by_img[int(det["image_id"])].append(det)

    img_by_id = {int(img["id"]): img for img in images}
    rng = random.Random(seed)

    sample_metrics: dict[str, list[float]] = {k: [] for k in metric_keys}
    for _ in range(num_samples):
        sampled = [img_ids[rng.randrange(len(img_ids))] for _ in range(len(img_ids))]

        new_images = []
        new_annotations = []
        new_dets = []

        ann_id = 1
        for new_img_id, orig_img_id in enumerate(sampled, start=1):
            img = dict(img_by_id[orig_img_id])
            img["id"] = new_img_id
            new_images.append(img)

            for ann in anns_by_img.get(orig_img_id, []):
                new_ann = dict(ann)
                new_ann["id"] = ann_id
                ann_id += 1
                new_ann["image_id"] = new_img_id
                new_annotations.append(new_ann)

            for det in det_by_img.get(orig_img_id, []):
                new_det = dict(det)
                new_det["image_id"] = new_img_id
                new_dets.append(new_det)

        boot_gt = {
            "info": gt_payload.get("info", {}),
            "licenses": gt_payload.get("licenses", []),
            "images": new_images,
            "annotations": new_annotations,
            "categories": gt_payload.get("categories", []),
        }
        metrics = run_coco_eval(boot_gt, new_dets)
        for key in metric_keys:
            sample_metrics[key].append(float(metrics[key]))

    ci: dict[str, dict[str, float]] = {}
    for key, values in sample_metrics.items():
        arr = np.asarray(values, dtype=float)
        ci[key] = {
            "mean": float(np.mean(arr)),
            "ci_low": float(np.quantile(arr, 0.025)),
            "ci_high": float(np.quantile(arr, 0.975)),
        }

    return ci
