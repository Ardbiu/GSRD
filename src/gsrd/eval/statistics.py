from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import pandas as pd


def mean_ci95(values: Iterable[float]) -> tuple[float, float, float, float]:
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return 0.0, 0.0, 0.0, 0.0
    mean = float(np.mean(arr))
    if arr.size == 1:
        return mean, 0.0, mean, mean
    std = float(np.std(arr, ddof=1))
    half = 1.96 * std / math.sqrt(arr.size)
    return mean, std, mean - half, mean + half


def summarize_granularity(artifact_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | str | int]] = []

    for (detector, dataset, granularity), group in artifact_df.groupby(
        ["detector", "dataset", "granularity"], as_index=False
    ):
        map_mean, map_std, map_lo, map_hi = mean_ci95(group["mAP"].tolist())
        map50_mean, map50_std, map50_lo, map50_hi = mean_ci95(group["mAP50"].tolist())
        rows.append(
            {
                "detector": str(detector),
                "dataset": str(dataset),
                "granularity": str(granularity),
                "mAP_mean": map_mean,
                "mAP_std": map_std,
                "mAP_ci95_low": map_lo,
                "mAP_ci95_high": map_hi,
                "mAP50_mean": map50_mean,
                "mAP50_std": map50_std,
                "mAP50_ci95_low": map50_lo,
                "mAP50_ci95_high": map50_hi,
                "n_vocab": int(group["vocab_id"].nunique()),
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    out["granularity_variance_mAP"] = out.groupby(["detector", "dataset"])["mAP_mean"].transform("var")

    standard_map = (
        out[out["granularity"] == "standard"]
        .set_index(["detector", "dataset"])["mAP_mean"]
        .to_dict()
    )
    drop, robust = [], []
    for _, row in out.iterrows():
        base = float(standard_map.get((row["detector"], row["dataset"]), row["mAP_mean"]))
        d = float(base - row["mAP_mean"])
        r = float(1.0 - (d / max(base, 1e-9)))
        drop.append(d)
        robust.append(r)
    out["drop_from_standard_mAP"] = drop
    out["robustness_to_standard"] = robust
    return out


def bootstrap_mean_diff(
    a: list[float],
    b: list[float],
    n_bootstrap: int,
    seed: int,
) -> dict[str, float]:
    if not a or not b:
        return {"mean": 0.0, "ci_low": 0.0, "ci_high": 0.0}

    rng = np.random.default_rng(seed)
    a_arr = np.asarray(a, dtype=float)
    b_arr = np.asarray(b, dtype=float)

    diffs = []
    for _ in range(max(10, n_bootstrap)):
        a_s = rng.choice(a_arr, size=a_arr.size, replace=True)
        b_s = rng.choice(b_arr, size=b_arr.size, replace=True)
        diffs.append(float(np.mean(a_s) - np.mean(b_s)))
    diff_arr = np.asarray(diffs, dtype=float)

    return {
        "mean": float(np.mean(a_arr) - np.mean(b_arr)),
        "ci_low": float(np.quantile(diff_arr, 0.025)),
        "ci_high": float(np.quantile(diff_arr, 0.975)),
    }


def shift_interaction(
    id_artifact: pd.DataFrame,
    ood_artifact: pd.DataFrame,
    n_bootstrap: int,
    seed: int,
) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    detectors = sorted(set(id_artifact["detector"]).intersection(set(ood_artifact["detector"])))
    for detector in detectors:
        id_det = id_artifact[id_artifact["detector"] == detector]
        ood_det = ood_artifact[ood_artifact["detector"] == detector]

        per_gran = {}
        for granularity in sorted(set(id_det["granularity"]).intersection(set(ood_det["granularity"]))):
            id_vals = id_det[id_det["granularity"] == granularity]["mAP"].tolist()
            ood_vals = ood_det[ood_det["granularity"] == granularity]["mAP"].tolist()
            diff = bootstrap_mean_diff(id_vals, ood_vals, n_bootstrap=n_bootstrap, seed=seed)
            per_gran[granularity] = diff

            rows.append(
                {
                    "detector": detector,
                    "granularity": granularity,
                    "mAP_id": float(np.mean(id_vals)) if id_vals else 0.0,
                    "mAP_ood": float(np.mean(ood_vals)) if ood_vals else 0.0,
                    "absolute_drop": diff["mean"],
                    "absolute_drop_ci95_low": diff["ci_low"],
                    "absolute_drop_ci95_high": diff["ci_high"],
                    "relative_drop_pct": 100.0 * diff["mean"] / max(1e-9, float(np.mean(id_vals)) if id_vals else 1e-9),
                }
            )

        standard = per_gran.get("standard")
        if standard is None:
            continue
        for granularity, item in per_gran.items():
            interaction = item["mean"] - standard["mean"]
            rows.append(
                {
                    "detector": detector,
                    "granularity": granularity,
                    "mAP_id": 0.0,
                    "mAP_ood": 0.0,
                    "absolute_drop": item["mean"],
                    "absolute_drop_ci95_low": item["ci_low"],
                    "absolute_drop_ci95_high": item["ci_high"],
                    "relative_drop_pct": 0.0,
                    "interaction_vs_standard": interaction,
                }
            )

    if not rows:
        return pd.DataFrame(
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
    return pd.DataFrame(rows)


def novelty_summary(artifact_df: pd.DataFrame, granularity_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []

    for (detector, dataset), group in granularity_df.groupby(["detector", "dataset"], as_index=False):
        vals = group["mAP_mean"].tolist()
        if len(vals) >= 2:
            pairwise = []
            for i in range(len(vals)):
                for j in range(i + 1, len(vals)):
                    pairwise.append(abs(vals[i] - vals[j]))
            gsi = float(np.mean(pairwise))
        else:
            gsi = 0.0

        standard = group[group["granularity"] == "standard"]["mAP_mean"]
        standard_map = float(standard.iloc[0]) if not standard.empty else float(np.mean(vals) if vals else 0.0)
        rows.append(
            {
                "detector": str(detector),
                "dataset": str(dataset),
                "granularity": "__overall__",
                "metric": "granularity_sensitivity_index",
                "value": gsi,
            }
        )
        rows.append(
            {
                "detector": str(detector),
                "dataset": str(dataset),
                "granularity": "__overall__",
                "metric": "normalized_gsi_to_standard",
                "value": float(gsi / max(1e-9, standard_map)),
            }
        )

    for (detector, dataset, granularity), group in artifact_df.groupby(
        ["detector", "dataset", "granularity"], as_index=False
    ):
        map_vals = group["mAP"].astype(float).to_numpy()
        if map_vals.size == 0:
            continue
        rows.extend(
            [
                {
                    "detector": str(detector),
                    "dataset": str(dataset),
                    "granularity": str(granularity),
                    "metric": "vocab_instability_cv",
                    "value": float(np.std(map_vals) / max(1e-9, np.mean(np.abs(map_vals)))),
                },
                {
                    "detector": str(detector),
                    "dataset": str(dataset),
                    "granularity": str(granularity),
                    "metric": "vocab_worst10_map",
                    "value": float(np.quantile(map_vals, 0.10)),
                },
                {
                    "detector": str(detector),
                    "dataset": str(dataset),
                    "granularity": str(granularity),
                    "metric": "vocab_median_map",
                    "value": float(np.quantile(map_vals, 0.50)),
                },
            ]
        )

    return pd.DataFrame(rows)
