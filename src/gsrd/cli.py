from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import typer

from gsrd.config.loader import load_config
from gsrd.pipeline.orchestrator import (
    run_full_pipeline,
    stage_data,
    stage_eval,
    stage_inference,
    stage_merge,
    stage_report,
    stage_risk,
    stage_vocab,
    write_run_manifest,
)
from gsrd.utils.logging import configure_logging
from gsrd.utils.seed import set_global_seed

app = typer.Typer(help="GSRD pipeline CLI")
data_app = typer.Typer(help="Dataset preparation commands")
vocab_app = typer.Typer(help="Vocabulary generation commands")
inference_app = typer.Typer(help="Inference commands")
eval_app = typer.Typer(help="Evaluation commands")
risk_app = typer.Typer(help="Risk control commands")
report_app = typer.Typer(help="Reporting commands")

app.add_typer(data_app, name="data")
app.add_typer(vocab_app, name="vocab")
app.add_typer(inference_app, name="inference")
app.add_typer(eval_app, name="eval")
app.add_typer(risk_app, name="risk")
app.add_typer(report_app, name="report")

LOGGER = logging.getLogger(__name__)


def _load(cfg: Path, overrides: list[str]):
    config = load_config(cfg, overrides=overrides)
    config.runtime.workdir = Path(os.path.expandvars(str(config.runtime.workdir))).expanduser()
    config.paths.data_root = Path(os.path.expandvars(str(config.paths.data_root))).expanduser()
    config.paths.cache_root = Path(os.path.expandvars(str(config.paths.cache_root))).expanduser()
    config.paths.outputs_root = Path(os.path.expandvars(str(config.paths.outputs_root))).expanduser()
    config.paths.logs_root = Path(os.path.expandvars(str(config.paths.logs_root))).expanduser()
    configure_logging(config.runtime.log_level, config.paths.logs_root / "gsrd.log")
    if config.runtime.deterministic:
        set_global_seed(config.runtime.seed)
    return config


@app.command("run")
def run(
    config_path: Path = typer.Option(Path("configs/default.yaml"), "--config", "-c"),
    stage: str = typer.Option(
        "data,vocab,inference,merge,eval,risk,report", help="Comma-separated stages"
    ),
    dataset: Optional[list[str]] = typer.Option(None, "--dataset"),
    detector: Optional[list[str]] = typer.Option(None, "--detector"),
    override: list[str] = typer.Option([], "--set", help="Override config: key=value"),
):
    config = _load(config_path, override)
    write_run_manifest(config, stage="run")
    outputs = run_full_pipeline(
        config,
        stages=[s.strip() for s in stage.split(",") if s.strip()],
        dataset_names=dataset,
        detector_names=detector,
    )
    for k, v in outputs.items():
        LOGGER.info("%s -> %d artifacts", k, len(v))


@data_app.command("prepare")
def data_prepare(
    config_path: Path = typer.Option(Path("configs/default.yaml"), "--config", "-c"),
    override: list[str] = typer.Option([], "--set"),
):
    config = _load(config_path, override)
    out = stage_data(config)
    LOGGER.info("Dataset manifest: %s", out)


@vocab_app.command("generate")
def vocab_generate(
    dataset: list[str] = typer.Option(..., "--dataset", help="Dataset(s) from manifest"),
    config_path: Path = typer.Option(Path("configs/default.yaml"), "--config", "-c"),
    override: list[str] = typer.Option([], "--set"),
):
    config = _load(config_path, override)
    outs = stage_vocab(config, dataset_names=dataset)
    for out in outs:
        LOGGER.info("Wrote vocab: %s", out)


@inference_app.command("run")
def inference_run(
    dataset: list[str] = typer.Option(..., "--dataset"),
    detector: Optional[list[str]] = typer.Option(None, "--detector"),
    config_path: Path = typer.Option(Path("configs/default.yaml"), "--config", "-c"),
    override: list[str] = typer.Option([], "--set"),
):
    config = _load(config_path, override)
    outs = stage_inference(config, dataset_names=dataset, detector_names=detector)
    LOGGER.info("Inference artifacts written: %d", len(outs))


@inference_app.command("merge")
def inference_merge(
    dataset: list[str] = typer.Option(..., "--dataset"),
    detector: Optional[list[str]] = typer.Option(None, "--detector"),
    config_path: Path = typer.Option(Path("configs/default.yaml"), "--config", "-c"),
    override: list[str] = typer.Option([], "--set"),
):
    config = _load(config_path, override)
    outs = stage_merge(config, dataset_names=dataset, detector_names=detector)
    LOGGER.info("Merged artifacts: %d", len(outs))


@eval_app.command("run")
def eval_run(
    dataset: list[str] = typer.Option(..., "--dataset"),
    detector: Optional[list[str]] = typer.Option(None, "--detector"),
    config_path: Path = typer.Option(Path("configs/default.yaml"), "--config", "-c"),
    override: list[str] = typer.Option([], "--set"),
):
    config = _load(config_path, override)
    outs = stage_eval(config, dataset_names=dataset, detector_names=detector)
    LOGGER.info("Evaluation artifacts: %d", len(outs))


@risk_app.command("run")
def risk_run(
    dataset: list[str] = typer.Option(..., "--dataset"),
    detector: Optional[list[str]] = typer.Option(None, "--detector"),
    config_path: Path = typer.Option(Path("configs/default.yaml"), "--config", "-c"),
    override: list[str] = typer.Option([], "--set"),
):
    config = _load(config_path, override)
    outs = stage_risk(config, dataset_names=dataset, detector_names=detector)
    LOGGER.info("Risk artifacts: %d", len(outs))


@report_app.command("generate")
def report_generate(
    dataset: list[str] = typer.Option(..., "--dataset"),
    detector: Optional[list[str]] = typer.Option(None, "--detector"),
    config_path: Path = typer.Option(Path("configs/default.yaml"), "--config", "-c"),
    override: list[str] = typer.Option([], "--set"),
):
    config = _load(config_path, override)
    outs = stage_report(config, dataset_names=dataset, detector_names=detector)
    LOGGER.info("Report artifacts: %s", outs)


if __name__ == "__main__":
    app()
