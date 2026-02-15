from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from gsrd.config.schema import EvaluationConfig
from gsrd.data.datasets import get_dataset_spec
from gsrd.eval.coco_eval import bootstrap_coco_ci, run_coco_eval
from gsrd.eval.remap import remap_ground_truth, remap_predictions
from gsrd.eval.statistics import novelty_summary, shift_interaction, summarize_granularity
from gsrd.utils.io import ensure_dir, iter_jsonl, read_json, write_json

LOGGER = logging.getLogger(__name__)


def _iter_prediction_artifacts(cache_root: Path, detector: str, dataset: str) -> list[Path]:
    root = cache_root / "predictions" / detector / dataset
    if not root.exists():
        return []
    metas = sorted(root.rglob("meta.json"))
    return [m for m in metas if (m.parent / "merged.jsonl.gz").exists()]


def _load_domain_metadata(path: Path | None) -> dict[int, dict[str, str]]:
    if path is None or not path.exists():
        return {}
    raw = read_json(path)
    return {int(k): {str(kk): str(vv) for kk, vv in v.items()} for k, v in raw.items()}


def _domain_breakdown(
    gt_payload: dict[str, Any],
    detections: list[dict[str, Any]],
    domain_meta: dict[int, dict[str, str]],
) -> list[dict[str, Any]]:
    if not domain_meta:
        return []
    rows: list[dict[str, Any]] = []
    for domain_key in ["timeofday", "weather", "scene"]:
        values = sorted({v.get(domain_key, "unknown") for v in domain_meta.values()})
        for domain_value in values:
            image_ids = [
                img_id for img_id, meta in domain_meta.items() if meta.get(domain_key, "unknown") == domain_value
            ]
            if len(image_ids) < 25:
                continue
            metrics = run_coco_eval(gt_payload, detections, image_ids=image_ids)
            rows.append(
                {
                    "domain_key": domain_key,
                    "domain_value": domain_value,
                    "num_images": len(image_ids),
                    "mAP": metrics["mAP"],
                    "mAP50": metrics["mAP50"],
                }
            )
    return rows


def evaluate_detector_dataset(
    cache_root: Path,
    outputs_root: Path,
    detector_name: str,
    dataset_name: str,
    eval_cfg: EvaluationConfig,
) -> dict[str, Path]:
    spec = get_dataset_spec(cache_root, dataset_name)
    gt_payload = read_json(spec.annotation_file)
    domain_meta = _load_domain_metadata(spec.domain_metadata_file)

    out_dir = ensure_dir(outputs_root / "eval" / detector_name / dataset_name)
    artifact_rows: list[dict[str, Any]] = []
    domain_rows: list[dict[str, Any]] = []

    for meta_path in _iter_prediction_artifacts(cache_root, detector_name, dataset_name):
        meta = read_json(meta_path)
        merged = meta_path.parent / "merged.jsonl.gz"
        pred_rows = list(iter_jsonl(merged))

        vocab = meta["vocab"]
        granularity = str(meta["granularity"])
        vocab_id = str(vocab["vocab_id"])

        class_to_term = {str(k): str(v) for k, v in vocab["class_to_term"].items()}
        remapped_gt, term_to_id = remap_ground_truth(gt_payload, class_to_term)
        remapped_pred = remap_predictions(pred_rows, term_to_id)

        metrics = run_coco_eval(remapped_gt, remapped_pred)
        ci = bootstrap_coco_ci(
            remapped_gt,
            remapped_pred,
            num_samples=eval_cfg.bootstrap_samples,
            seed=eval_cfg.bootstrap_seed,
        )

        artifact_row = {
            "detector": detector_name,
            "dataset": dataset_name,
            "granularity": granularity,
            "vocab_id": vocab_id,
            **metrics,
            "mAP_ci_low": ci["mAP"]["ci_low"],
            "mAP_ci_high": ci["mAP"]["ci_high"],
            "mAP50_ci_low": ci["mAP50"]["ci_low"],
            "mAP50_ci_high": ci["mAP50"]["ci_high"],
            "num_predictions": len(remapped_pred),
        }
        artifact_rows.append(artifact_row)

        for dom in _domain_breakdown(remapped_gt, remapped_pred, domain_meta):
            domain_rows.append(
                {
                    "detector": detector_name,
                    "dataset": dataset_name,
                    "granularity": granularity,
                    "vocab_id": vocab_id,
                    **dom,
                }
            )

    if not artifact_rows:
        raise FileNotFoundError(
            "No merged prediction artifacts found. Run inference and merge shards first."
        )

    artifact_df = pd.DataFrame(artifact_rows)
    domain_df = pd.DataFrame(domain_rows)

    artifact_csv = out_dir / "artifact_metrics.csv"
    artifact_json = out_dir / "artifact_metrics.json"
    artifact_df.to_csv(artifact_csv, index=False)
    write_json(artifact_json, artifact_rows)

    domain_csv = out_dir / "domain_breakdown.csv"
    if not domain_df.empty:
        domain_df.to_csv(domain_csv, index=False)
        write_json(out_dir / "domain_breakdown.json", domain_rows)

    granularity_df = summarize_granularity(artifact_df)

    gran_csv = out_dir / "granularity_summary.csv"
    gran_json = out_dir / "granularity_summary.json"
    granularity_df.to_csv(gran_csv, index=False)
    write_json(gran_json, granularity_df.to_dict(orient="records"))

    robustness_rows = []
    for (detector, dataset), group in granularity_df.groupby(["detector", "dataset"], as_index=False):
        if group.empty:
            continue
        robustness_rows.append(
            {
                "detector": detector,
                "dataset": dataset,
                "worst_case_drop_mAP": float(group["drop_from_standard_mAP"].max()),
                "mean_drop_mAP": float(group["drop_from_standard_mAP"].mean()),
                "granularity_variance_mAP": float(group["mAP_mean"].var(ddof=0)),
                "robustness_to_standard_min": float(group["robustness_to_standard"].min()),
            }
        )
    robustness_csv = out_dir / "robustness_summary.csv"
    write_json(out_dir / "robustness_summary.json", robustness_rows)
    pd.DataFrame(robustness_rows).to_csv(robustness_csv, index=False)

    novelty_df = novelty_summary(artifact_df, granularity_df)
    novelty_csv = out_dir / "novelty_summary.csv"
    novelty_df.to_csv(novelty_csv, index=False)
    write_json(out_dir / "novelty_summary.json", novelty_df.to_dict(orient="records"))

    return {
        "artifact_csv": artifact_csv,
        "artifact_json": artifact_json,
        "domain_csv": domain_csv,
        "granularity_csv": gran_csv,
        "granularity_json": gran_json,
        "robustness_csv": robustness_csv,
        "novelty_csv": novelty_csv,
    }


def build_shift_interaction_summary(
    outputs_root: Path,
    detector_names: list[str],
    n_bootstrap: int = 500,
    seed: int = 123,
) -> Path:
    frames = []
    for detector in detector_names:
        coco_path = outputs_root / "eval" / detector / "coco_val2017" / "artifact_metrics.csv"
        bdd_path = outputs_root / "eval" / detector / "bdd100k_val" / "artifact_metrics.csv"
        if not (coco_path.exists() and bdd_path.exists()):
            continue
        id_df = pd.read_csv(coco_path)
        ood_df = pd.read_csv(bdd_path)
        frames.append(
            shift_interaction(
                id_artifact=id_df,
                ood_artifact=ood_df,
                n_bootstrap=n_bootstrap,
                seed=seed,
            )
        )

    out_dir = ensure_dir(outputs_root / "eval")
    out_path = out_dir / "shift_interaction_summary.csv"

    if frames:
        result = pd.concat(frames, ignore_index=True)
    else:
        result = pd.DataFrame(
            columns=[
                "detector",
                "granularity",
                "mAP_id",
                "mAP_ood",
                "absolute_drop",
                "absolute_drop_ci95_low",
                "absolute_drop_ci95_high",
                "relative_drop_pct",
                "interaction_vs_standard",
            ]
        )

    result.to_csv(out_path, index=False)
    write_json(out_dir / "shift_interaction_summary.json", result.to_dict(orient="records"))
    return out_path
