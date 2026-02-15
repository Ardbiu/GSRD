from __future__ import annotations

import logging
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from gsrd.config.schema import RiskConfig
from gsrd.data.datasets import get_dataset_spec
from gsrd.risk.control import (
    ThresholdStats,
    clopper_pearson_ucb,
    enforce_monotone_nonincreasing,
    hoeffding_ucb,
    select_threshold,
)
from gsrd.risk.matching import build_gt_index, detection_correctness
from gsrd.utils.hashing import stable_hash
from gsrd.utils.io import ensure_dir, iter_jsonl, read_json, write_json
from gsrd.vocab.taxonomy import normalize_label

LOGGER = logging.getLogger(__name__)


def _iter_meta_paths(cache_root: Path, detector_name: str, dataset_name: str) -> list[Path]:
    root = cache_root / "predictions" / detector_name / dataset_name
    if not root.exists():
        return []
    return sorted(p for p in root.rglob("meta.json") if (p.parent / "merged.jsonl.gz").exists())


def _split_images(image_ids: list[int], fraction: float, seed: int) -> tuple[set[int], set[int]]:
    rng = random.Random(seed)
    ids = sorted(set(image_ids))
    rng.shuffle(ids)
    n_cal = max(1, int(len(ids) * fraction))
    cal = set(ids[:n_cal])
    test = set(ids[n_cal:])
    if not test:
        test = cal
    return cal, test


def _candidate_thresholds(pred_rows: list[dict[str, Any]], n_grid: int = 50) -> list[float]:
    scores = [float(det["score"]) for row in pred_rows for det in row.get("detections", [])]
    if not scores:
        return [1.0]
    quantiles = np.linspace(0.0, 1.0, num=n_grid)
    vals = sorted({float(np.quantile(scores, q)) for q in quantiles})
    vals.append(1.0)
    return sorted(set(vals))


def _load_domain_groups(
    domain_metadata_file: Path | None,
    image_ids: set[int],
    group_key: str,
) -> dict[int, str]:
    if domain_metadata_file is not None and domain_metadata_file.exists():
        raw = read_json(domain_metadata_file)
        groups = {
            int(img_id): str(attrs.get(group_key, "unknown"))
            for img_id, attrs in raw.items()
            if int(img_id) in image_ids
        }
        if groups:
            return groups

    # Fallback deterministic pseudo-groups to keep the method operational when
    # no domain metadata is available.
    return {img_id: f"bucket_{img_id % 4}" for img_id in image_ids}


def _ucb_from_counts(num_errors: int, n: int, empirical_risk: float, delta: float, method: str) -> float:
    if method == "clopper_pearson":
        return clopper_pearson_ucb(num_errors=num_errors, n=n, delta=delta)
    return hoeffding_ucb(empirical_risk, n=n, delta=delta)


def _evaluate_threshold(
    pred_rows: list[dict[str, Any]],
    gt_index: dict[int, Any],
    term_to_classes: dict[str, list[str]],
    image_filter: set[int],
    threshold: float,
    hierarchy_aware: bool,
    iou_thr: float,
) -> tuple[int, int, int]:
    kept = 0
    errors = 0
    total = 0
    for row in pred_rows:
        image_id = int(row["image_id"])
        if image_id not in image_filter:
            continue
        gts = gt_index.get(image_id, [])
        for det in row.get("detections", []):
            total += 1
            score = float(det["score"])
            if score < threshold:
                continue
            kept += 1
            term = normalize_label(str(det.get("term", "")))
            classes = {normalize_label(c) for c in term_to_classes.get(term, [term])}
            if not detection_correctness(
                det_bbox=[float(x) for x in det["bbox"]],
                det_classes=classes,
                gt_objects=gts,
                iou_thr=iou_thr,
                hierarchy_aware=hierarchy_aware,
            ):
                errors += 1
    return kept, errors, total


def _evaluate_threshold_groupwise(
    pred_rows: list[dict[str, Any]],
    gt_index: dict[int, Any],
    term_to_classes: dict[str, list[str]],
    image_filter: set[int],
    image_to_group: dict[int, str],
    threshold: float,
    hierarchy_aware: bool,
    iou_thr: float,
) -> dict[str, tuple[int, int, int]]:
    stats: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])  # kept, errors, total

    for row in pred_rows:
        image_id = int(row["image_id"])
        if image_id not in image_filter:
            continue
        group = image_to_group.get(image_id, "unknown")
        gts = gt_index.get(image_id, [])
        for det in row.get("detections", []):
            stats[group][2] += 1
            score = float(det["score"])
            if score < threshold:
                continue
            stats[group][0] += 1
            term = normalize_label(str(det.get("term", "")))
            classes = {normalize_label(c) for c in term_to_classes.get(term, [term])}
            if not detection_correctness(
                det_bbox=[float(x) for x in det["bbox"]],
                det_classes=classes,
                gt_objects=gts,
                iou_thr=iou_thr,
                hierarchy_aware=hierarchy_aware,
            ):
                stats[group][1] += 1

    return {g: (v[0], v[1], v[2]) for g, v in stats.items()}


def _calibrate_one(
    pred_rows: list[dict[str, Any]],
    gt_index: dict[int, Any],
    term_to_classes: dict[str, list[str]],
    cal_ids: set[int],
    test_ids: set[int],
    alphas: list[float],
    iou_thr: float,
    hierarchy_aware: bool,
    delta: float,
    ucb_method: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    thresholds = _candidate_thresholds(pred_rows)

    raw_stats: list[ThresholdStats] = []
    risks = []
    for t in thresholds:
        kept, errors, total = _evaluate_threshold(
            pred_rows,
            gt_index,
            term_to_classes,
            cal_ids,
            threshold=t,
            hierarchy_aware=hierarchy_aware,
            iou_thr=iou_thr,
        )
        # For zero kept predictions, empirical risk is undefined; use 0.0 here so
        # the monotone envelope is not dominated by the empty-coverage endpoint.
        # Conservativeness is still enforced through UCB (n=0 -> ucb_risk=1.0).
        risk = float(errors / kept) if kept > 0 else 0.0
        coverage = float(kept / max(1, total))
        risks.append(risk)
        raw_stats.append(
            ThresholdStats(
                threshold=t,
                num_kept=kept,
                num_errors=errors,
                risk=risk,
                coverage=coverage,
                ucb_risk=1.0,
            )
        )

    monotone = enforce_monotone_nonincreasing(risks)
    fixed_stats: list[ThresholdStats] = []
    for stat, risk_hat in zip(raw_stats, monotone):
        ucb = _ucb_from_counts(stat.num_errors, stat.num_kept, risk_hat, delta, ucb_method)
        fixed_stats.append(
            ThresholdStats(
                threshold=stat.threshold,
                num_kept=stat.num_kept,
                num_errors=stat.num_errors,
                risk=risk_hat,
                coverage=stat.coverage,
                ucb_risk=ucb,
            )
        )

    curve_rows = [
        {
            "threshold": s.threshold,
            "risk": s.risk,
            "coverage": s.coverage,
            "ucb_risk": s.ucb_risk,
            "num_kept": s.num_kept,
            "num_errors": s.num_errors,
        }
        for s in fixed_stats
    ]

    result_rows = []
    for alpha in alphas:
        choice = select_threshold(fixed_stats, alpha=alpha)
        kept, errors, total = _evaluate_threshold(
            pred_rows,
            gt_index,
            term_to_classes,
            test_ids,
            threshold=choice.threshold,
            hierarchy_aware=hierarchy_aware,
            iou_thr=iou_thr,
        )
        achieved_risk = float(errors / kept) if kept > 0 else 1.0
        achieved_coverage = float(kept / max(1, total))
        result_rows.append(
            {
                "alpha": alpha,
                "selected_threshold": choice.threshold,
                "calib_ucb_risk": choice.ucb_risk,
                "calib_empirical_risk": choice.risk,
                "calib_coverage": choice.coverage,
                "test_achieved_risk": achieved_risk,
                "test_achieved_coverage": achieved_coverage,
                "test_num_kept": kept,
                "test_num_errors": errors,
                "ucb_method": ucb_method,
                "delta": delta,
                "calib_guarantee_margin": float(alpha - choice.ucb_risk),
                "test_margin": float(alpha - achieved_risk),
                "target_met_on_test": bool(achieved_risk <= alpha),
            }
        )

    return curve_rows, result_rows


def _calibrate_groupwise(
    pred_rows: list[dict[str, Any]],
    gt_index: dict[int, Any],
    term_to_classes: dict[str, list[str]],
    cal_ids: set[int],
    test_ids: set[int],
    image_to_group: dict[int, str],
    alphas: list[float],
    iou_thr: float,
    hierarchy_aware: bool,
    delta: float,
    ucb_method: str,
    min_group_calibration_samples: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    thresholds = _candidate_thresholds(pred_rows)
    curve_rows: list[dict[str, Any]] = []

    for t in thresholds:
        gstats = _evaluate_threshold_groupwise(
            pred_rows,
            gt_index,
            term_to_classes,
            cal_ids,
            image_to_group,
            threshold=t,
            hierarchy_aware=hierarchy_aware,
            iou_thr=iou_thr,
        )
        total_kept = sum(v[0] for v in gstats.values())
        total_errors = sum(v[1] for v in gstats.values())
        total_preds = sum(v[2] for v in gstats.values())

        eligible = {
            g: vals
            for g, vals in gstats.items()
            if vals[2] >= min_group_calibration_samples
        }
        if not eligible:
            eligible = gstats

        worst_group = "unknown"
        worst_ucb = -1.0
        worst_empirical = 1.0
        for g, (kept, errors, _) in eligible.items():
            risk = float(errors / kept) if kept > 0 else 1.0
            ucb = _ucb_from_counts(errors, kept, risk, delta, ucb_method)
            if ucb > worst_ucb:
                worst_ucb = ucb
                worst_group = g
                worst_empirical = risk

        coverage = float(total_kept / max(1, total_preds))
        overall_risk = float(total_errors / total_kept) if total_kept > 0 else 0.0
        curve_rows.append(
            {
                "threshold": t,
                "risk": overall_risk,
                "coverage": coverage,
                "ucb_risk": worst_ucb,
                "num_kept": total_kept,
                "num_errors": total_errors,
                "worst_group": worst_group,
                "worst_group_risk": worst_empirical,
                "worst_group_ucb": worst_ucb,
            }
        )

    result_rows: list[dict[str, Any]] = []
    for alpha in alphas:
        feasible = [row for row in curve_rows if row["worst_group_ucb"] <= alpha]
        chosen = min(feasible, key=lambda x: x["threshold"]) if feasible else max(
            curve_rows, key=lambda x: x["threshold"]
        )

        test_stats = _evaluate_threshold_groupwise(
            pred_rows,
            gt_index,
            term_to_classes,
            test_ids,
            image_to_group,
            threshold=float(chosen["threshold"]),
            hierarchy_aware=hierarchy_aware,
            iou_thr=iou_thr,
        )
        total_kept = sum(v[0] for v in test_stats.values())
        total_errors = sum(v[1] for v in test_stats.values())
        total_preds = sum(v[2] for v in test_stats.values())

        worst_group = "unknown"
        worst_group_risk = -1.0
        for g, (kept, errors, total) in test_stats.items():
            if total <= 0:
                continue
            risk = float(errors / kept) if kept > 0 else 1.0
            if risk > worst_group_risk:
                worst_group_risk = risk
                worst_group = g

        achieved_risk = float(total_errors / total_kept) if total_kept > 0 else 1.0
        achieved_coverage = float(total_kept / max(1, total_preds))

        result_rows.append(
            {
                "alpha": alpha,
                "selected_threshold": chosen["threshold"],
                "calib_ucb_risk": chosen["worst_group_ucb"],
                "calib_empirical_risk": chosen["worst_group_risk"],
                "calib_coverage": chosen["coverage"],
                "test_achieved_risk": achieved_risk,
                "test_achieved_coverage": achieved_coverage,
                "test_num_kept": total_kept,
                "test_num_errors": total_errors,
                "ucb_method": ucb_method,
                "delta": delta,
                "calib_guarantee_margin": float(alpha - float(chosen["worst_group_ucb"])),
                "test_margin": float(alpha - achieved_risk),
                "target_met_on_test": bool(achieved_risk <= alpha),
                "worst_group_test": worst_group,
                "worst_group_test_risk": float(max(0.0, worst_group_risk)),
                "target_met_worst_group": bool(max(0.0, worst_group_risk) <= alpha),
            }
        )

    return curve_rows, result_rows


def run_risk_control(
    cache_root: Path,
    outputs_root: Path,
    detector_name: str,
    dataset_name: str,
    risk_cfg: RiskConfig,
    seed: int,
    iou_thr: float = 0.5,
) -> dict[str, Path]:
    spec = get_dataset_spec(cache_root, dataset_name)
    gt_payload = read_json(spec.annotation_file)
    id_to_name = {int(k): normalize_label(v) for k, v in spec.category_id_to_name.items()}
    gt_index = build_gt_index(gt_payload, id_to_name)

    out_dir = ensure_dir(outputs_root / "risk" / detector_name / dataset_name)
    split_dir = ensure_dir(out_dir / "splits")
    all_curves: list[dict[str, Any]] = []
    all_results: list[dict[str, Any]] = []

    for meta_path in _iter_meta_paths(cache_root, detector_name, dataset_name):
        meta = read_json(meta_path)
        pred_rows = list(iter_jsonl(meta_path.parent / "merged.jsonl.gz"))
        image_ids = [int(row["image_id"]) for row in pred_rows]
        cal_ids, test_ids = _split_images(image_ids, risk_cfg.calibration_fraction, seed=seed)

        term_to_classes = {
            normalize_label(k): [normalize_label(v) for v in vals]
            for k, vals in meta["vocab"]["term_to_classes"].items()
        }

        granularity = str(meta["granularity"])
        vocab_id = str(meta["vocab"]["vocab_id"])

        all_ids = set(cal_ids) | set(test_ids)
        image_to_group = _load_domain_groups(spec.domain_metadata_file, all_ids, risk_cfg.group_key)

        split_payload = {
            "detector": detector_name,
            "dataset": dataset_name,
            "granularity": granularity,
            "vocab_id": vocab_id,
            "seed": seed,
            "calibration_fraction": risk_cfg.calibration_fraction,
            "group_key": risk_cfg.group_key,
            "num_calibration_images": len(cal_ids),
            "num_test_images": len(test_ids),
            "calibration_image_ids": sorted(cal_ids),
            "test_image_ids": sorted(test_ids),
        }
        split_name = stable_hash(
            {"detector": detector_name, "dataset": dataset_name, "granularity": granularity, "vocab_id": vocab_id}
        )
        write_json(split_dir / f"split_{split_name}.json", split_payload)

        if risk_cfg.strategy == "both":
            strategies = ["flat", "hierarchy"]
        elif risk_cfg.strategy == "all":
            strategies = ["flat", "hierarchy", "groupwise"]
        else:
            strategies = [risk_cfg.strategy]

        for strategy in strategies:
            hierarchy_aware = strategy == "hierarchy"
            strategy_iou = risk_cfg.hierarchy_relax_iou if hierarchy_aware else iou_thr

            if strategy == "groupwise":
                curves, results = _calibrate_groupwise(
                    pred_rows,
                    gt_index,
                    term_to_classes,
                    cal_ids,
                    test_ids,
                    image_to_group=image_to_group,
                    alphas=risk_cfg.alphas,
                    iou_thr=risk_cfg.hierarchy_relax_iou,
                    hierarchy_aware=True,
                    delta=risk_cfg.delta,
                    ucb_method=risk_cfg.ucb_method,
                    min_group_calibration_samples=risk_cfg.min_group_calibration_samples,
                )
            else:
                curves, results = _calibrate_one(
                    pred_rows,
                    gt_index,
                    term_to_classes,
                    cal_ids,
                    test_ids,
                    alphas=risk_cfg.alphas,
                    iou_thr=strategy_iou,
                    hierarchy_aware=hierarchy_aware,
                    delta=risk_cfg.delta,
                    ucb_method=risk_cfg.ucb_method,
                )

            for item in curves:
                all_curves.append(
                    {
                        "detector": detector_name,
                        "dataset": dataset_name,
                        "granularity": granularity,
                        "vocab_id": vocab_id,
                        "strategy": strategy,
                        **item,
                    }
                )

            for item in results:
                all_results.append(
                    {
                        "detector": detector_name,
                        "dataset": dataset_name,
                        "granularity": granularity,
                        "vocab_id": vocab_id,
                        "strategy": strategy,
                        **item,
                    }
                )

    curve_df = pd.DataFrame(all_curves)
    result_df = pd.DataFrame(all_results)

    curve_csv = out_dir / "risk_curves.csv"
    result_csv = out_dir / "risk_results.csv"
    curve_df.to_csv(curve_csv, index=False)
    result_df.to_csv(result_csv, index=False)

    write_json(out_dir / "risk_curves.json", all_curves)
    write_json(out_dir / "risk_results.json", all_results)

    summary_rows = []
    if not result_df.empty:
        grouped = result_df.groupby(["detector", "dataset", "granularity", "strategy", "alpha"], as_index=False)
        for _, grp in grouped:
            row = grp.iloc[0]
            entry = {
                "detector": row["detector"],
                "dataset": row["dataset"],
                "granularity": row["granularity"],
                "strategy": row["strategy"],
                "alpha": float(row["alpha"]),
                "ucb_method": row["ucb_method"],
                "delta": float(row["delta"]),
                "mean_test_risk": float(grp["test_achieved_risk"].mean()),
                "mean_test_coverage": float(grp["test_achieved_coverage"].mean()),
                "target_met_rate": float(grp["target_met_on_test"].mean()),
                "mean_guarantee_margin": float(grp["calib_guarantee_margin"].mean()),
            }
            if "target_met_worst_group" in grp.columns:
                entry["worst_group_target_met_rate"] = float(grp["target_met_worst_group"].mean())
                entry["worst_group_test_risk"] = float(grp["worst_group_test_risk"].mean())
            summary_rows.append(entry)

    summary_df = pd.DataFrame(summary_rows)
    summary_csv = out_dir / "risk_summary.csv"
    summary_df.to_csv(summary_csv, index=False)
    write_json(out_dir / "risk_summary.json", summary_rows)

    return {
        "risk_curve_csv": curve_csv,
        "risk_results_csv": result_csv,
        "risk_summary_csv": summary_csv,
    }
