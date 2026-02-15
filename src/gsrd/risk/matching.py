from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any


def iou_xywh(a: list[float], b: list[float]) -> float:
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    iw = max(0.0, inter_x2 - inter_x1)
    ih = max(0.0, inter_y2 - inter_y1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    union = aw * ah + bw * bh - inter
    return inter / max(union, 1e-12)


@dataclass(frozen=True)
class GTObject:
    bbox: list[float]
    class_name: str


def build_gt_index(gt_payload: dict[str, Any], category_id_to_name: dict[int, str]) -> dict[int, list[GTObject]]:
    by_img: dict[int, list[GTObject]] = defaultdict(list)
    for ann in gt_payload.get("annotations", []):
        image_id = int(ann["image_id"])
        cid = int(ann["category_id"])
        cls = category_id_to_name[cid]
        by_img[image_id].append(GTObject(bbox=[float(x) for x in ann["bbox"]], class_name=cls))
    return by_img


def detection_correctness(
    det_bbox: list[float],
    det_classes: set[str],
    gt_objects: list[GTObject],
    iou_thr: float,
    hierarchy_aware: bool,
) -> bool:
    best_iou = 0.0
    best_cls: str | None = None
    for gt in gt_objects:
        iou = iou_xywh(det_bbox, gt.bbox)
        if iou > best_iou:
            best_iou = iou
            best_cls = gt.class_name

    if best_cls is None or best_iou < iou_thr:
        return False

    if hierarchy_aware:
        return best_cls in det_classes
    return best_cls in det_classes and len(det_classes) == 1
