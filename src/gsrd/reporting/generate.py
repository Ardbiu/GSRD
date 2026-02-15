from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from gsrd.config.schema import ReportingConfig
from gsrd.utils.io import ensure_dir, write_json

LOGGER = logging.getLogger(__name__)

GRAN_ORDER = ["coarse", "standard", "fine", "mixed", "counterfactual"]


def _load_eval_frames(
    outputs_root: Path,
    detectors: list[str],
    datasets: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    artifact_frames = []
    gran_frames = []
    domain_frames = []
    novelty_frames = []

    for detector in detectors:
        for dataset in datasets:
            base = outputs_root / "eval" / detector / dataset
            artifact_path = base / "artifact_metrics.csv"
            gran_path = base / "granularity_summary.csv"
            domain_path = base / "domain_breakdown.csv"
            novelty_path = base / "novelty_summary.csv"
            if artifact_path.exists():
                artifact_frames.append(pd.read_csv(artifact_path))
            if gran_path.exists():
                gran_frames.append(pd.read_csv(gran_path))
            if domain_path.exists():
                domain_frames.append(pd.read_csv(domain_path))
            if novelty_path.exists():
                novelty_frames.append(pd.read_csv(novelty_path))

    interaction_path = outputs_root / "eval" / "shift_interaction_summary.csv"
    interaction_df = pd.read_csv(interaction_path) if interaction_path.exists() else pd.DataFrame()

    artifact_df = pd.concat(artifact_frames, ignore_index=True) if artifact_frames else pd.DataFrame()
    gran_df = pd.concat(gran_frames, ignore_index=True) if gran_frames else pd.DataFrame()
    domain_df = pd.concat(domain_frames, ignore_index=True) if domain_frames else pd.DataFrame()
    novelty_df = pd.concat(novelty_frames, ignore_index=True) if novelty_frames else pd.DataFrame()
    return artifact_df, gran_df, domain_df, interaction_df, novelty_df


def _load_risk_frames(
    outputs_root: Path,
    detectors: list[str],
    datasets: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    curve_frames = []
    result_frames = []
    summary_frames = []
    for detector in detectors:
        for dataset in datasets:
            base = outputs_root / "risk" / detector / dataset
            c = base / "risk_curves.csv"
            r = base / "risk_results.csv"
            s = base / "risk_summary.csv"
            if c.exists():
                curve_frames.append(pd.read_csv(c))
            if r.exists():
                result_frames.append(pd.read_csv(r))
            if s.exists():
                summary_frames.append(pd.read_csv(s))
    curves = pd.concat(curve_frames, ignore_index=True) if curve_frames else pd.DataFrame()
    results = pd.concat(result_frames, ignore_index=True) if result_frames else pd.DataFrame()
    summaries = pd.concat(summary_frames, ignore_index=True) if summary_frames else pd.DataFrame()
    return curves, results, summaries


def _save_fig(fig_path: Path) -> None:
    plt.tight_layout()
    plt.savefig(fig_path)
    plt.savefig(fig_path.with_suffix(".pdf"))
    plt.close()


def _plot_granularity(gran_df: pd.DataFrame, out_dir: Path, cfg: ReportingConfig) -> Path | None:
    if gran_df.empty:
        return None
    fig_path = out_dir / "granularity_sensitivity.png"

    plt.figure(figsize=(8.5, 4.8), dpi=cfg.figure_dpi)
    for (detector, dataset), group in gran_df.groupby(["detector", "dataset"]):
        ordered = group.copy()
        ordered["granularity"] = pd.Categorical(ordered["granularity"], categories=GRAN_ORDER, ordered=True)
        ordered = ordered.sort_values("granularity")
        x = range(len(ordered))
        label = f"{detector} | {dataset}"
        plt.plot(x, ordered["mAP_mean"], marker="o", linewidth=2, label=label)
        if "mAP_ci95_low" in ordered.columns and "mAP_ci95_high" in ordered.columns:
            plt.fill_between(x, ordered["mAP_ci95_low"], ordered["mAP_ci95_high"], alpha=0.18)

    plt.title(f"{cfg.title_prefix}: Granularity Sensitivity with 95% CI")
    plt.xlabel("Vocabulary Granularity")
    plt.ylabel("mAP")
    plt.xticks(range(len(GRAN_ORDER)), GRAN_ORDER)
    plt.legend(loc="best", fontsize=8)
    _save_fig(fig_path)
    return fig_path


def _plot_risk_coverage(
    curves_df: pd.DataFrame,
    risk_results_df: pd.DataFrame,
    out_dir: Path,
    cfg: ReportingConfig,
) -> Path | None:
    if curves_df.empty:
        return None

    fig_path = out_dir / "risk_vs_coverage.png"

    plt.figure(figsize=(8.5, 4.8), dpi=cfg.figure_dpi)
    line_df = (
        curves_df.groupby(["strategy", "granularity", "threshold"], as_index=False)
        .agg(risk=("risk", "mean"), coverage=("coverage", "mean"))
        .sort_values("threshold")
    )
    sns.lineplot(
        data=line_df,
        x="coverage",
        y="risk",
        hue="strategy",
        style="granularity",
        linewidth=2,
    )

    if not risk_results_df.empty:
        alpha_pts = (
            risk_results_df.groupby(["strategy", "alpha"], as_index=False)
            .agg(test_achieved_risk=("test_achieved_risk", "mean"), test_achieved_coverage=("test_achieved_coverage", "mean"))
        )
        sns.scatterplot(
            data=alpha_pts,
            x="test_achieved_coverage",
            y="test_achieved_risk",
            hue="strategy",
            style="alpha",
            legend=False,
            s=80,
            edgecolor="black",
        )

    plt.title(f"{cfg.title_prefix}: Risk vs Coverage")
    plt.xlabel("Coverage")
    plt.ylabel("Risk")
    plt.ylim(0, 1)
    plt.xlim(0, 1)
    _save_fig(fig_path)
    return fig_path


def _plot_shift_domains(domain_df: pd.DataFrame, out_dir: Path, cfg: ReportingConfig) -> Path | None:
    if domain_df.empty:
        return None

    df = domain_df.copy()
    # Keep one domain axis for readability in a main figure.
    if "timeofday" in set(df["domain_key"]):
        df = df[df["domain_key"] == "timeofday"]

    fig_path = out_dir / "id_to_ood_domain_breakdown.png"
    plt.figure(figsize=(9, 5), dpi=cfg.figure_dpi)
    sns.barplot(data=df, x="domain_value", y="mAP50", hue="granularity")
    plt.title(f"{cfg.title_prefix}: OOD Domain Breakdown (mAP50)")
    plt.xlabel("Domain")
    plt.ylabel("mAP50")
    plt.xticks(rotation=20, ha="right")
    _save_fig(fig_path)
    return fig_path


def _plot_id_ood_degradation(interaction_df: pd.DataFrame, out_dir: Path, cfg: ReportingConfig) -> Path | None:
    if interaction_df.empty:
        return None
    df = interaction_df.dropna(subset=["mAP_id", "mAP_ood"]).copy()
    df = df[df["mAP_id"] > 0]
    if df.empty:
        return None

    fig_path = out_dir / "id_to_ood_degradation.png"
    plt.figure(figsize=(8.8, 4.8), dpi=cfg.figure_dpi)
    sns.barplot(data=df, x="granularity", y="absolute_drop", hue="detector", order=GRAN_ORDER)

    if {"absolute_drop_ci95_low", "absolute_drop_ci95_high"}.issubset(df.columns):
        x_positions = {g: i for i, g in enumerate(GRAN_ORDER)}
        for _, row in df.iterrows():
            g = row["granularity"]
            if g not in x_positions:
                continue
            x = x_positions[g]
            lo = float(row["absolute_drop_ci95_low"])
            hi = float(row["absolute_drop_ci95_high"])
            y = float(row["absolute_drop"])
            plt.plot([x, x], [lo, hi], color="black", linewidth=1)
            plt.plot([x - 0.07, x + 0.07], [lo, lo], color="black", linewidth=1)
            plt.plot([x - 0.07, x + 0.07], [hi, hi], color="black", linewidth=1)

    plt.title(f"{cfg.title_prefix}: ID to OOD Degradation by Granularity")
    plt.xlabel("Granularity")
    plt.ylabel("mAP Drop (ID - OOD)")
    _save_fig(fig_path)
    return fig_path


def _plot_shift_granularity_heatmap(interaction_df: pd.DataFrame, out_dir: Path, cfg: ReportingConfig) -> Path | None:
    if interaction_df.empty:
        return None

    if "interaction_vs_standard" in interaction_df.columns:
        df = interaction_df.dropna(subset=["interaction_vs_standard"]).copy()
        value_col = "interaction_vs_standard"
        title = f"{cfg.title_prefix}: Shift × Granularity Interaction (vs Standard)"
    else:
        df = interaction_df.copy()
        value_col = "absolute_drop"
        title = f"{cfg.title_prefix}: Shift × Granularity Interaction"

    if df.empty:
        return None

    pivot = df.pivot_table(index="detector", columns="granularity", values=value_col, aggfunc="mean")
    if pivot.empty:
        return None

    fig_path = out_dir / "shift_granularity_interaction_heatmap.png"
    plt.figure(figsize=(8.2, 4.6), dpi=cfg.figure_dpi)
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="RdYlBu_r", center=0.0)
    plt.title(title)
    plt.xlabel("Granularity")
    plt.ylabel("Detector")
    _save_fig(fig_path)
    return fig_path


def _write_tables(
    gran_df: pd.DataFrame,
    risk_results_df: pd.DataFrame,
    risk_summary_df: pd.DataFrame,
    interaction_df: pd.DataFrame,
    novelty_df: pd.DataFrame,
    out_dir: Path,
) -> dict[str, Path]:
    table_paths: dict[str, Path] = {}

    if not gran_df.empty:
        perf = gran_df[
            [
                "detector",
                "dataset",
                "granularity",
                "mAP_mean",
                "mAP_ci95_low",
                "mAP_ci95_high",
                "drop_from_standard_mAP",
                "robustness_to_standard",
            ]
        ].copy()
        perf["mAP_95ci"] = perf.apply(
            lambda r: f"{r['mAP_mean']:.4f} [{r['mAP_ci95_low']:.4f}, {r['mAP_ci95_high']:.4f}]",
            axis=1,
        )
        perf_csv = out_dir / "table_performance_by_granularity.csv"
        perf_tex = out_dir / "table_performance_by_granularity.tex"
        perf.to_csv(perf_csv, index=False)
        perf.to_latex(perf_tex, index=False)
        table_paths["performance_csv"] = perf_csv
        table_paths["performance_tex"] = perf_tex

        worst = (
            gran_df.groupby(["detector", "dataset"], as_index=False)
            .agg(
                worst_case_drop_mAP=("drop_from_standard_mAP", "max"),
                mean_drop_mAP=("drop_from_standard_mAP", "mean"),
                min_robustness_to_standard=("robustness_to_standard", "min"),
            )
            .sort_values(["detector", "dataset"])
        )
        worst_csv = out_dir / "table_worst_case_drop.csv"
        worst_tex = out_dir / "table_worst_case_drop.tex"
        worst.to_csv(worst_csv, index=False)
        worst.to_latex(worst_tex, index=False, float_format="%.4f")
        table_paths["worst_drop_csv"] = worst_csv
        table_paths["worst_drop_tex"] = worst_tex

    if not risk_results_df.empty:
        calib = (
            risk_results_df.groupby(["detector", "dataset", "granularity", "strategy", "alpha"], as_index=False)
            .agg(
                achieved_risk=("test_achieved_risk", "mean"),
                achieved_coverage=("test_achieved_coverage", "mean"),
                target_met_rate=("target_met_on_test", "mean"),
                calib_margin=("calib_guarantee_margin", "mean"),
            )
            .sort_values(["detector", "dataset", "strategy", "alpha", "granularity"])
        )
        calib_csv = out_dir / "table_calibration_effectiveness.csv"
        calib_tex = out_dir / "table_calibration_effectiveness.tex"
        calib.to_csv(calib_csv, index=False)
        calib.to_latex(calib_tex, index=False, float_format="%.4f")
        table_paths["calib_csv"] = calib_csv
        table_paths["calib_tex"] = calib_tex

    if not risk_summary_df.empty:
        summary_csv = out_dir / "table_risk_summary.csv"
        summary_tex = out_dir / "table_risk_summary.tex"
        risk_summary_df.to_csv(summary_csv, index=False)
        risk_summary_df.to_latex(summary_tex, index=False, float_format="%.4f")
        table_paths["risk_summary_csv"] = summary_csv
        table_paths["risk_summary_tex"] = summary_tex

    if not interaction_df.empty:
        shift = interaction_df[
            [
                "detector",
                "granularity",
                "mAP_id",
                "mAP_ood",
                "absolute_drop",
                "absolute_drop_ci95_low",
                "absolute_drop_ci95_high",
                "relative_drop_pct",
            ]
        ].drop_duplicates()
        shift = shift[shift["mAP_id"] > 0]
        shift_csv = out_dir / "table_shift_interaction.csv"
        shift_tex = out_dir / "table_shift_interaction.tex"
        shift.to_csv(shift_csv, index=False)
        shift.to_latex(shift_tex, index=False, float_format="%.4f")
        table_paths["shift_csv"] = shift_csv
        table_paths["shift_tex"] = shift_tex

    if not novelty_df.empty:
        novelty_csv = out_dir / "table_novelty_metrics.csv"
        novelty_tex = out_dir / "table_novelty_metrics.tex"
        novelty_df.to_csv(novelty_csv, index=False)
        novelty_df.to_latex(novelty_tex, index=False, float_format="%.4f")
        table_paths["novelty_csv"] = novelty_csv
        table_paths["novelty_tex"] = novelty_tex

    return table_paths


def _headline_numbers(
    gran_df: pd.DataFrame,
    risk_results_df: pd.DataFrame,
    interaction_df: pd.DataFrame,
) -> dict[str, Any]:
    stats: dict[str, Any] = {}
    if not gran_df.empty:
        best = gran_df.sort_values("mAP_mean", ascending=False).iloc[0]
        stats["best_map"] = {
            "detector": best["detector"],
            "dataset": best["dataset"],
            "granularity": best["granularity"],
            "mAP_mean": float(best["mAP_mean"]),
        }
        stats["max_worst_case_drop"] = float(gran_df["drop_from_standard_mAP"].max())
    if not risk_results_df.empty:
        stats["risk_target_success_rate"] = float(risk_results_df["target_met_on_test"].mean())
        stats["mean_test_coverage"] = float(risk_results_df["test_achieved_coverage"].mean())
    if not interaction_df.empty and "absolute_drop" in interaction_df.columns:
        stats["mean_id_to_ood_drop"] = float(interaction_df["absolute_drop"].mean())
    return stats


def generate_report(
    outputs_root: Path,
    detectors: list[str],
    datasets: list[str],
    cfg: ReportingConfig,
) -> dict[str, Path]:
    sns.set_theme(style=cfg.style)

    report_root = ensure_dir(outputs_root / "report")
    fig_dir = ensure_dir(report_root / "figures")
    table_dir = ensure_dir(report_root / "tables")

    artifact_df, gran_df, domain_df, interaction_df, novelty_df = _load_eval_frames(
        outputs_root, detectors, datasets
    )
    curves_df, risk_results_df, risk_summary_df = _load_risk_frames(outputs_root, detectors, datasets)

    plots = {
        "granularity_fig": _plot_granularity(gran_df, fig_dir, cfg),
        "risk_fig": _plot_risk_coverage(curves_df, risk_results_df, fig_dir, cfg),
        "domain_fig": _plot_shift_domains(domain_df, fig_dir, cfg),
        "degradation_fig": _plot_id_ood_degradation(interaction_df, fig_dir, cfg),
        "interaction_fig": _plot_shift_granularity_heatmap(interaction_df, fig_dir, cfg),
    }

    tables = _write_tables(gran_df, risk_results_df, risk_summary_df, interaction_df, novelty_df, table_dir)
    headlines = _headline_numbers(gran_df, risk_results_df, interaction_df)

    summary_path = report_root / "final_report.md"
    lines = ["# GSRD Final Report", "", "## Headline Numbers"]
    if headlines:
        for key, value in headlines.items():
            lines.append(f"- **{key}**: {value}")
    else:
        lines.append("- No headline numbers were available. Run evaluation and risk stages first.")

    lines.extend(["", "## Generated Figures (PNG + PDF)"])
    for key, path in plots.items():
        if path is not None:
            lines.append(f"- {key}: `{path}` and `{path.with_suffix('.pdf')}`")

    lines.extend(["", "## Generated Tables"])
    for key, path in tables.items():
        lines.append(f"- {key}: `{path}`")

    lines.extend(["", "## Experimental Notes"])
    lines.append("- Granularity summaries include vocabulary-list uncertainty (95% CI).")
    lines.append("- Counterfactual vocabulary stress tests quantify brittleness to lexical perturbations.")
    lines.append("- Novelty tables include GSI and vocabulary-instability metrics.")
    lines.append("- Shift interaction includes uncertainty over ID-to-OOD drop estimates.")
    lines.append("- Risk calibration reports target satisfaction rates and guarantee margins.")

    summary_path.write_text("\n".join(lines), encoding="utf-8")

    manifest = {
        "summary_markdown": str(summary_path),
        "figures": {
            key: {"png": str(path), "pdf": str(path.with_suffix('.pdf'))}
            for key, path in plots.items()
            if path is not None
        },
        "tables": {k: str(v) for k, v in tables.items()},
        "headline_numbers": headlines,
        "counts": {
            "n_artifact_rows": int(len(artifact_df)),
            "n_granularity_rows": int(len(gran_df)),
            "n_risk_rows": int(len(risk_results_df)),
        },
    }
    manifest_path = report_root / "report_manifest.json"
    write_json(manifest_path, manifest)

    LOGGER.info("Report generated at %s", summary_path)
    return {"summary": summary_path, "manifest": manifest_path}
