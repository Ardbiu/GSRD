from __future__ import annotations

import datetime as dt
import logging
from pathlib import Path
from typing import Iterable

from gsrd.config.schema import DetectorConfig, RunConfig
from gsrd.utils.io import ensure_dir, write_json
from gsrd.utils.system import git_commit_or_none, hardware_summary

LOGGER = logging.getLogger(__name__)


def _enabled_detectors(config: RunConfig) -> list[DetectorConfig]:
    dets = [d for d in config.detectors if d.enabled]
    if not dets:
        raise ValueError("No enabled detectors in config.detectors")
    return dets


def _selected_datasets(cache_root: Path, requested: list[str] | None = None) -> list[str]:
    from gsrd.data.datasets import list_dataset_names

    available = list_dataset_names(cache_root)
    if requested:
        missing = sorted(set(requested) - set(available))
        if missing:
            raise KeyError(f"Requested datasets missing in manifest: {missing}")
        return requested
    return available


def write_run_manifest(config: RunConfig, stage: str) -> Path:
    run_id = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = ensure_dir(config.paths.outputs_root / "runs" / run_id)
    manifest = {
        "run_id": run_id,
        "stage": stage,
        "timestamp_utc": dt.datetime.utcnow().isoformat() + "Z",
        "seed": config.runtime.seed,
        "deterministic": config.runtime.deterministic,
        "config": config.model_dump(mode="json"),
        "git_commit": git_commit_or_none(config.runtime.workdir),
        "hardware": hardware_summary(),
    }
    path = run_dir / "run_manifest.json"
    write_json(path, manifest)
    return path


def stage_data(config: RunConfig) -> Path:
    from gsrd.data.prepare import run_data_prep

    return run_data_prep(
        data_root=config.paths.data_root,
        cache_root=config.paths.cache_root,
        prepare_coco=config.datasets.coco_val,
        prepare_bdd=config.datasets.bdd100k_val,
        prepare_toy=config.datasets.toy,
        toy_num_images=config.datasets.toy_num_images,
        max_images=config.datasets.max_images,
    )


def stage_vocab(config: RunConfig, dataset_names: list[str] | None = None) -> list[Path]:
    from gsrd.vocab.generate import generate_vocabularies

    names = _selected_datasets(config.paths.cache_root, dataset_names)
    outputs = []
    for ds in names:
        outputs.append(
            generate_vocabularies(
                cache_root=config.paths.cache_root,
                dataset_name=ds,
                lists_per_granularity=config.vocab.lists_per_granularity,
                seed=config.runtime.seed,
                mixed_ratio=config.vocab.mixed_ratio,
                include_counterfactual=config.vocab.include_counterfactual,
                counterfactual_lists=config.vocab.counterfactual_lists,
                counterfactual_noise_ratio=config.vocab.counterfactual_noise_ratio,
            )
        )
    return outputs


def stage_inference(
    config: RunConfig,
    dataset_names: list[str] | None = None,
    detector_names: list[str] | None = None,
) -> list[Path]:
    from gsrd.inference.runner import compute_inference_summary, run_inference_for_dataset

    names = _selected_datasets(config.paths.cache_root, dataset_names)
    detectors = _enabled_detectors(config)
    if detector_names:
        detectors = [d for d in detectors if d.name in detector_names]
    outputs = []
    for det in detectors:
        for ds in names:
            outputs.extend(
                run_inference_for_dataset(
                    cache_root=config.paths.cache_root,
                    data_root=config.paths.data_root,
                    dataset_name=ds,
                    detector_cfg=det,
                    inference_cfg=config.inference,
                    slurm_task_var=config.cluster.slurm_array_var,
                    slurm_count_var=config.cluster.slurm_array_count_var,
                )
            )
    summary_path = config.paths.outputs_root / "compute_summary.json"
    compute_inference_summary(config.paths.cache_root, summary_path)
    return outputs


def stage_merge(
    config: RunConfig,
    dataset_names: list[str] | None = None,
    detector_names: list[str] | None = None,
) -> list[Path]:
    from gsrd.inference.runner import merge_inference_artifacts

    names = _selected_datasets(config.paths.cache_root, dataset_names)
    detectors = _enabled_detectors(config)
    if detector_names:
        detectors = [d for d in detectors if d.name in detector_names]

    outputs = []
    for det in detectors:
        for ds in names:
            outputs.extend(
                merge_inference_artifacts(
                    cache_root=config.paths.cache_root,
                    detector_name=det.name,
                    dataset_name=ds,
                    num_shards=config.inference.num_shards,
                )
            )
    return outputs


def stage_eval(
    config: RunConfig,
    dataset_names: list[str] | None = None,
    detector_names: list[str] | None = None,
) -> list[Path]:
    try:
        from gsrd.eval.engine import build_shift_interaction_summary, evaluate_detector_dataset
    except ModuleNotFoundError as exc:  # pragma: no cover - environment-specific
        if exc.name and exc.name.startswith("pycocotools"):
            raise RuntimeError(
                "Evaluation requires pycocotools. Install dependencies with `pip install -e .`."
            ) from exc
        raise

    names = _selected_datasets(config.paths.cache_root, dataset_names)
    detectors = _enabled_detectors(config)
    if detector_names:
        detectors = [d for d in detectors if d.name in detector_names]

    outputs: list[Path] = []
    for det in detectors:
        for ds in names:
            paths = evaluate_detector_dataset(
                cache_root=config.paths.cache_root,
                outputs_root=config.paths.outputs_root,
                detector_name=det.name,
                dataset_name=ds,
                eval_cfg=config.evaluation,
            )
            outputs.extend(paths.values())

    shift = build_shift_interaction_summary(
        config.paths.outputs_root,
        [d.name for d in detectors],
        n_bootstrap=max(200, config.evaluation.bootstrap_samples),
        seed=config.evaluation.bootstrap_seed,
    )
    outputs.append(shift)
    return outputs


def stage_risk(
    config: RunConfig,
    dataset_names: list[str] | None = None,
    detector_names: list[str] | None = None,
) -> list[Path]:
    from gsrd.risk.runner import run_risk_control

    names = _selected_datasets(config.paths.cache_root, dataset_names)
    detectors = _enabled_detectors(config)
    if detector_names:
        detectors = [d for d in detectors if d.name in detector_names]

    outputs: list[Path] = []
    for det in detectors:
        for ds in names:
            paths = run_risk_control(
                cache_root=config.paths.cache_root,
                outputs_root=config.paths.outputs_root,
                detector_name=det.name,
                dataset_name=ds,
                risk_cfg=config.risk,
                seed=config.runtime.seed,
            )
            outputs.extend(paths.values())
    return outputs


def stage_report(
    config: RunConfig,
    dataset_names: list[str] | None = None,
    detector_names: list[str] | None = None,
) -> list[Path]:
    from gsrd.reporting.generate import generate_report

    names = _selected_datasets(config.paths.cache_root, dataset_names)
    detectors = _enabled_detectors(config)
    if detector_names:
        detectors = [d for d in detectors if d.name in detector_names]

    paths = generate_report(
        outputs_root=config.paths.outputs_root,
        detectors=[d.name for d in detectors],
        datasets=names,
        cfg=config.reporting,
    )
    return list(paths.values())


def run_full_pipeline(
    config: RunConfig,
    stages: Iterable[str],
    dataset_names: list[str] | None = None,
    detector_names: list[str] | None = None,
) -> dict[str, list[Path]]:
    stage_list = [s.strip().lower() for s in stages]
    outputs: dict[str, list[Path]] = {}

    for stage in stage_list:
        LOGGER.info("Running stage: %s", stage)
        write_run_manifest(config, stage=stage)
        if stage == "data":
            outputs[stage] = [stage_data(config)]
        elif stage == "vocab":
            outputs[stage] = stage_vocab(config, dataset_names=dataset_names)
        elif stage == "inference":
            outputs[stage] = stage_inference(
                config,
                dataset_names=dataset_names,
                detector_names=detector_names,
            )
        elif stage == "merge":
            outputs[stage] = stage_merge(
                config,
                dataset_names=dataset_names,
                detector_names=detector_names,
            )
        elif stage == "eval":
            outputs[stage] = stage_eval(
                config,
                dataset_names=dataset_names,
                detector_names=detector_names,
            )
        elif stage == "risk":
            outputs[stage] = stage_risk(
                config,
                dataset_names=dataset_names,
                detector_names=detector_names,
            )
        elif stage == "report":
            outputs[stage] = stage_report(
                config,
                dataset_names=dataset_names,
                detector_names=detector_names,
            )
        else:
            raise ValueError(f"Unknown stage: {stage}")

    return outputs
