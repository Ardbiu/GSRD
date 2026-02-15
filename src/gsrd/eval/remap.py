from __future__ import annotations

from typing import Any

from gsrd.vocab.taxonomy import normalize_label

UNKNOWN_TERM = "__unknown__"


def remap_ground_truth(
    gt_payload: dict[str, Any],
    class_to_term: dict[str, str],
    include_unknown: bool = True,
) -> tuple[dict[str, Any], dict[str, int]]:
    """Remap canonical classes into vocabulary terms for granularity-specific evaluation."""
    cat_id_to_name = {int(c["id"]): normalize_label(str(c["name"])) for c in gt_payload.get("categories", [])}

    mapped_terms = sorted(set(class_to_term.values()))
    if include_unknown:
        mapped_terms.append(UNKNOWN_TERM)
    term_to_id = {term: idx + 1 for idx, term in enumerate(mapped_terms)}

    out_categories = [{"id": cid, "name": term} for term, cid in term_to_id.items()]

    out_annotations = []
    for ann in gt_payload.get("annotations", []):
        raw_name = cat_id_to_name.get(int(ann["category_id"]))
        if raw_name is None:
            continue
        term = class_to_term.get(raw_name)
        if term is None:
            continue
        new_ann = dict(ann)
        new_ann["category_id"] = term_to_id[term]
        out_annotations.append(new_ann)

    out_gt = {
        "info": gt_payload.get("info", {}),
        "licenses": gt_payload.get("licenses", []),
        "images": gt_payload.get("images", []),
        "annotations": out_annotations,
        "categories": out_categories,
    }
    return out_gt, term_to_id


def remap_predictions(
    prediction_rows: list[dict[str, Any]],
    term_to_id: dict[str, int],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in prediction_rows:
        image_id = int(row["image_id"])
        for det in row.get("detections", []):
            term = normalize_label(str(det.get("term", "")))
            if term not in term_to_id:
                if UNKNOWN_TERM in term_to_id:
                    term = UNKNOWN_TERM
                else:
                    continue
            out.append(
                {
                    "image_id": image_id,
                    "category_id": int(term_to_id[term]),
                    "bbox": [float(x) for x in det["bbox"]],
                    "score": float(det["score"]),
                }
            )
    return out
