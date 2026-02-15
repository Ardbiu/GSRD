from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any

from PIL import Image
from tqdm import tqdm

from gsrd.config.schema import DetectorConfig, InferenceConfig
from gsrd.data.datasets import DetectionDataset, get_dataset_spec
from gsrd.detectors.factory import create_detector
from gsrd.inference.cache import merge_shards, shard_exists, shard_file_name, write_metadata, write_shard
from gsrd.inference.schema import PredictionArtifactSpec, make_prediction_metadata
from gsrd.utils.hashing import stable_hash
from gsrd.utils.io import ensure_dir, iter_jsonl, read_json, write_json, write_jsonl
from gsrd.utils.system import hardware_summary

LOGGER = logging.getLogger(__name__)


def _resolve_shards(cfg: InferenceConfig, slurm_task_var: str, slurm_count_var: str) -> tuple[int, int]:
    num_shards = cfg.num_shards
    shard_index = cfg.shard_index
    if cfg.split_from_env:
        if slurm_task_var in os.environ and slurm_count_var in os.environ:
            shard_index = int(os.environ[slurm_task_var])
            num_shards = int(os.environ[slurm_count_var])
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError(f"Invalid shard index {shard_index} for num_shards={num_shards}")
    return num_shards, shard_index


def _load_vocab_lists(cache_root: Path, dataset_name: str) -> dict[str, list[dict[str, Any]]]:
    vocab_path = cache_root / "vocabs" / f"{dataset_name}_vocabs.json"
    if not vocab_path.exists():
        raise FileNotFoundError(f"Vocabulary file missing: {vocab_path}. Run `gsrd vocab generate` first.")
    payload = read_json(vocab_path)
    granularities = payload.get("granularities")
    if not isinstance(granularities, dict):
        raise ValueError(f"Malformed vocabulary file: {vocab_path}")
    return {k: v for k, v in granularities.items() if isinstance(v, list)}


def _collect_union_terms(granularity_lists: dict[str, list[dict[str, Any]]]) -> list[str]:
    terms: set[str] = set()
    for vocab_lists in granularity_lists.values():
        for vocab in vocab_lists:
            for term in vocab.get("terms", []):
                terms.add(_normalize_phrase(str(term)))
    return sorted(terms)


def _shared_cache_dir(cache_root: Path, detector_name: str, dataset_name: str) -> Path:
    return ensure_dir(cache_root / "shared_predictions" / detector_name / dataset_name)


def _shared_shard_path(
    cache_root: Path,
    detector_name: str,
    dataset_name: str,
    shard_index: int,
    num_shards: int,
) -> Path:
    return _shared_cache_dir(cache_root, detector_name, dataset_name) / shard_file_name(shard_index, num_shards)


def _shared_meta_path(cache_root: Path, detector_name: str, dataset_name: str) -> Path:
    return _shared_cache_dir(cache_root, detector_name, dataset_name) / "meta.json"


def _shared_shard_exists(
    cache_root: Path,
    detector_name: str,
    dataset_name: str,
    shard_index: int,
    num_shards: int,
) -> bool:
    path = _shared_shard_path(cache_root, detector_name, dataset_name, shard_index, num_shards)
    return path.exists() and path.stat().st_size > 0


def _normalize_phrase(term: str) -> str:
    normalized = term.lower().strip()
    for prefix in ("a ", "an ", "the "):
        if normalized.startswith(prefix):
            normalized = normalized[len(prefix):]
    return " ".join(normalized.replace("_", " ").split())


def _nearest_term(
    term: str,
    vocab_terms: list[str],
    class_to_term: dict[str, str],
    normalized_vocab_lookup: dict[str, str],
) -> str:
    normalized = _normalize_phrase(term)
    if normalized in normalized_vocab_lookup:
        return normalized_vocab_lookup[normalized]
    if normalized in class_to_term:
        return class_to_term[normalized]

    # Direct class token containment fallback.
    for cls, mapped_term in class_to_term.items():
        cls_norm = _normalize_phrase(cls)
        if cls_norm and (cls_norm in normalized or normalized in cls_norm):
            return mapped_term

    # Substring / token overlap fallback for phrase outputs from grounding models.
    candidates = []
    for vt in vocab_terms:
        overlap = len(set(_normalize_phrase(vt).split()) & set(normalized.split()))
        candidates.append((overlap, vt))
    candidates.sort(reverse=True)
    best_overlap, best_term = candidates[0]
    if best_overlap <= 0:
        return vocab_terms[0]
    return best_term


def _image_to_row(
    image_id: int,
    file_name: str,
    detections: list[Any],
    vocab_terms: list[str],
    class_to_term: dict[str, str],
    save_logits: bool,
) -> dict[str, Any]:
    rows = []
    normalized_vocab = [t.lower().strip() for t in vocab_terms]
    normalized_vocab_lookup = {_normalize_phrase(v): v for v in normalized_vocab}
    normalized_class_to_term = {_normalize_phrase(k): str(v).lower().strip() for k, v in class_to_term.items()}
    for det in detections:
        x1, y1, x2, y2 = det.bbox_xyxy
        w = max(0.0, x2 - x1)
        h = max(0.0, y2 - y1)
        if w <= 0 or h <= 0:
            continue
        raw_term = str(det.term).lower().strip()
        term = _nearest_term(
            raw_term,
            normalized_vocab,
            normalized_class_to_term,
            normalized_vocab_lookup,
        )
        entry: dict[str, Any] = {
            "term": term,
            "score": float(det.score),
            "bbox": [float(x1), float(y1), float(w), float(h)],
        }
        if save_logits:
            entry["logit"] = float(det.score)
        rows.append(entry)
    return {"image_id": image_id, "file_name": file_name, "detections": rows}


def _project_rows_to_vocab(
    source_rows: list[dict[str, Any]],
    vocab_terms: list[str],
    class_to_term: dict[str, str],
    save_logits: bool,
) -> list[dict[str, Any]]:
    normalized_vocab = [t.lower().strip() for t in vocab_terms]
    normalized_vocab_lookup = {_normalize_phrase(v): v for v in normalized_vocab}
    normalized_class_to_term = {_normalize_phrase(k): str(v).lower().strip() for k, v in class_to_term.items()}

    projected: list[dict[str, Any]] = []
    for row in source_rows:
        detections: list[dict[str, Any]] = []
        for det in row.get("detections", []):
            term = _nearest_term(
                str(det.get("term", "")),
                normalized_vocab,
                normalized_class_to_term,
                normalized_vocab_lookup,
            )
            out_det: dict[str, Any] = {
                "term": term,
                "score": float(det["score"]),
                "bbox": [float(x) for x in det["bbox"]],
            }
            if save_logits:
                out_det["logit"] = float(det.get("logit", det["score"]))
            detections.append(out_det)
        projected.append(
            {
                "image_id": int(row["image_id"]),
                "file_name": str(row.get("file_name", "")),
                "detections": detections,
            }
        )
    return projected


def _filtered_images(
    dataset: DetectionDataset,
    num_shards: int,
    shard_index: int,
    max_images: int | None,
) -> list[Any]:
    rows = [item for idx, item in enumerate(dataset.images) if idx % num_shards == shard_index]
    if max_images is not None and max_images > 0:
        return rows[:max_images]
    return rows


def run_inference_for_dataset(
    cache_root: Path,
    data_root: Path,
    dataset_name: str,
    detector_cfg: DetectorConfig,
    inference_cfg: InferenceConfig,
    slurm_task_var: str,
    slurm_count_var: str,
) -> list[Path]:
    spec = get_dataset_spec(cache_root, dataset_name)
    dataset = DetectionDataset(spec)
    granularity_lists = _load_vocab_lists(cache_root, dataset_name)

    num_shards, shard_index = _resolve_shards(inference_cfg, slurm_task_var, slurm_count_var)
    selected_images = _filtered_images(dataset, num_shards, shard_index, inference_cfg.max_images)
    LOGGER.info(
        "Dataset=%s | shard %d/%d has %d images",
        dataset_name,
        shard_index,
        num_shards,
        len(selected_images),
    )

    outputs: list[Path] = []

    with create_detector(detector_cfg) as detector:
        if inference_cfg.prompt_mode == "shared_union":
            union_terms = _collect_union_terms(granularity_lists)
            if not union_terms:
                raise ValueError(f"No vocabulary terms found for dataset={dataset_name}")

            shared_shard = _shared_shard_path(
                cache_root=cache_root,
                detector_name=detector_cfg.name,
                dataset_name=dataset_name,
                shard_index=shard_index,
                num_shards=num_shards,
            )
            if inference_cfg.skip_existing and _shared_shard_exists(
                cache_root=cache_root,
                detector_name=detector_cfg.name,
                dataset_name=dataset_name,
                shard_index=shard_index,
                num_shards=num_shards,
            ):
                LOGGER.info("Reusing shared union shard: %s", shared_shard)
                shared_rows = list(iter_jsonl(shared_shard))
            else:
                start = time.time()
                identity_map = {term: term for term in union_terms}
                shared_rows = []
                for image_entry in tqdm(
                    selected_images,
                    desc=f"{detector_cfg.name}:{dataset_name}:shared-union",
                ):
                    image_path = spec.image_dir / image_entry.file_name
                    if not image_path.exists():
                        raise FileNotFoundError(f"Image missing: {image_path}")
                    with Image.open(image_path) as image:
                        rgb = image.convert("RGB")
                        detections = detector.predict(rgb, union_terms)
                    shared_rows.append(
                        _image_to_row(
                            image_entry.image_id,
                            image_entry.file_name,
                            detections,
                            union_terms,
                            identity_map,
                            inference_cfg.save_logits,
                        )
                    )
                runtime_sec = max(1e-6, time.time() - start)
                write_jsonl(shared_shard, shared_rows)
                shared_format_version = make_prediction_metadata(
                    detector_name=detector_cfg.name,
                    detector_version=detector.version(),
                    dataset_name=dataset_name,
                    granularity="shared_union",
                    vocab={
                        "vocab_id": stable_hash({"mode": "shared_union", "terms": union_terms}),
                        "terms": union_terms,
                        "class_to_term": {term: term for term in union_terms},
                        "term_to_classes": {term: [term] for term in union_terms},
                    },
                    num_shards=num_shards,
                )["format_version"]
                shared_meta = {
                    "format_version": shared_format_version,
                    "detector": {
                        "name": detector_cfg.name,
                        "version": detector.version(),
                    },
                    "dataset": dataset_name,
                    "mode": "shared_union",
                    "terms": union_terms,
                    "num_shards": num_shards,
                    "shard_stats": {
                        "shard_index": shard_index,
                        "num_images": len(shared_rows),
                        "runtime_sec": runtime_sec,
                        "images_per_sec": len(shared_rows) / runtime_sec,
                        "count_for_compute": True,
                    },
                }
                write_json(
                    _shared_meta_path(cache_root, detector_cfg.name, dataset_name),
                    shared_meta,
                )
                outputs.append(shared_shard)

            for granularity, vocab_lists in granularity_lists.items():
                for vocab in vocab_lists:
                    artifact = PredictionArtifactSpec(
                        detector=detector_cfg.name,
                        dataset=dataset_name,
                        granularity=granularity,
                        vocab_id=vocab["vocab_id"],
                    )
                    if inference_cfg.skip_existing and shard_exists(
                        cache_root, artifact, shard_index, num_shards
                    ):
                        LOGGER.info(
                            "Skipping existing derived shard for %s/%s/%s",
                            dataset_name,
                            granularity,
                            vocab["vocab_id"],
                        )
                        continue

                    start = time.time()
                    projected_rows = _project_rows_to_vocab(
                        source_rows=shared_rows,
                        vocab_terms=vocab["terms"],
                        class_to_term=vocab["class_to_term"],
                        save_logits=inference_cfg.save_logits,
                    )
                    shard_path = write_shard(cache_root, artifact, shard_index, num_shards, projected_rows)
                    runtime_sec = max(1e-6, time.time() - start)
                    meta = make_prediction_metadata(
                        detector_name=detector_cfg.name,
                        detector_version=detector.version(),
                        dataset_name=dataset_name,
                        granularity=granularity,
                        vocab=vocab,
                        num_shards=num_shards,
                    )
                    meta["inference_mode"] = "shared_union_derived"
                    meta["source_shared_predictions"] = {
                        "path": str(shared_shard),
                        "num_terms": len(union_terms),
                    }
                    meta["shard_stats"] = {
                        "shard_index": shard_index,
                        "num_images": len(projected_rows),
                        "runtime_sec": runtime_sec,
                        "images_per_sec": len(projected_rows) / runtime_sec,
                        "count_for_compute": False,
                    }
                    write_metadata(cache_root, artifact, meta)
                    outputs.append(shard_path)
            return outputs

        for granularity, vocab_lists in granularity_lists.items():
            for vocab in vocab_lists:
                artifact = PredictionArtifactSpec(
                    detector=detector_cfg.name,
                    dataset=dataset_name,
                    granularity=granularity,
                    vocab_id=vocab["vocab_id"],
                )

                if inference_cfg.skip_existing and shard_exists(
                    cache_root, artifact, shard_index, num_shards
                ):
                    LOGGER.info(
                        "Skipping existing shard for %s/%s/%s", dataset_name, granularity, vocab["vocab_id"]
                    )
                    continue

                start = time.time()
                rows = []
                for image_entry in tqdm(
                    selected_images,
                    desc=f"{detector_cfg.name}:{dataset_name}:{granularity}:{vocab['vocab_id'][:6]}",
                ):
                    image_path = spec.image_dir / image_entry.file_name
                    if not image_path.exists():
                        raise FileNotFoundError(f"Image missing: {image_path}")
                    with Image.open(image_path) as image:
                        rgb = image.convert("RGB")
                        detections = detector.predict(rgb, vocab["terms"])
                    row = _image_to_row(
                        image_entry.image_id,
                        image_entry.file_name,
                        detections,
                        vocab["terms"],
                        vocab["class_to_term"],
                        inference_cfg.save_logits,
                    )
                    rows.append(row)

                shard_path = write_shard(cache_root, artifact, shard_index, num_shards, rows)
                runtime_sec = max(1e-6, time.time() - start)
                meta = make_prediction_metadata(
                    detector_name=detector_cfg.name,
                    detector_version=detector.version(),
                    dataset_name=dataset_name,
                    granularity=granularity,
                    vocab=vocab,
                    num_shards=num_shards,
                )
                meta["inference_mode"] = "per_vocab"
                meta["shard_stats"] = {
                    "shard_index": shard_index,
                    "num_images": len(rows),
                    "runtime_sec": runtime_sec,
                    "images_per_sec": len(rows) / runtime_sec,
                    "count_for_compute": True,
                }
                write_metadata(cache_root, artifact, meta)
                outputs.append(shard_path)

    return outputs


def merge_inference_artifacts(
    cache_root: Path,
    detector_name: str,
    dataset_name: str,
    num_shards: int,
) -> list[Path]:
    root = cache_root / "predictions" / detector_name / dataset_name
    if not root.exists():
        raise FileNotFoundError(f"No predictions found at {root}")

    merged_paths: list[Path] = []
    for gran_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for vocab_dir in sorted(p for p in gran_dir.iterdir() if p.is_dir()):
            artifact = PredictionArtifactSpec(
                detector=detector_name,
                dataset=dataset_name,
                granularity=gran_dir.name,
                vocab_id=vocab_dir.name,
            )
            merged = merge_shards(cache_root, artifact, num_shards)
            meta_path = vocab_dir / "meta.json"
            meta = read_json(meta_path) if meta_path.exists() else {}
            meta["merged_file"] = str(merged)
            write_json(meta_path, meta)
            merged_paths.append(merged)
    return merged_paths


def compute_inference_summary(cache_root: Path, output_path: Path) -> Path:
    pred_root = cache_root / "predictions"
    shared_root = cache_root / "shared_predictions"
    totals = {
        "total_images_processed": 0,
        "total_runtime_sec": 0.0,
        "artifacts": 0,
        "derived_artifacts": 0,
    }

    def accumulate_meta(meta_path: Path) -> None:
        meta = read_json(meta_path)
        shard = meta.get("shard_stats", {})
        if bool(shard.get("count_for_compute", True)):
            totals["total_images_processed"] += int(shard.get("num_images", 0))
            totals["total_runtime_sec"] += float(shard.get("runtime_sec", 0.0))
            totals["artifacts"] += 1
        else:
            totals["derived_artifacts"] += 1

    if pred_root.exists():
        for meta_path in pred_root.rglob("meta.json"):
            accumulate_meta(meta_path)
    if shared_root.exists():
        for meta_path in shared_root.rglob("meta.json"):
            accumulate_meta(meta_path)

    totals["avg_images_per_sec"] = (
        totals["total_images_processed"] / totals["total_runtime_sec"]
        if totals["total_runtime_sec"] > 0
        else 0.0
    )
    totals["approx_gpu_hours"] = totals["total_runtime_sec"] / 3600.0
    hw = hardware_summary()
    totals["gpu_type"] = ", ".join(hw.get("gpus", [])) if hw.get("gpus") else "cpu"
    totals["gpu_count"] = int(hw.get("gpu_count", 0))
    totals["summary_hash"] = stable_hash(totals)

    write_json(output_path, totals)
    return output_path
