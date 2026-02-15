from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field


class RuntimeConfig(BaseModel):
    seed: int = 20260215
    deterministic: bool = True
    log_level: str = "INFO"
    workdir: Path = Path(".")


class PathsConfig(BaseModel):
    data_root: Path = Path("data")
    cache_root: Path = Path("artifacts/cache")
    outputs_root: Path = Path("artifacts/outputs")
    logs_root: Path = Path("artifacts/logs")


class DatasetPrepareConfig(BaseModel):
    coco_val: bool = True
    bdd100k_val: bool = True
    toy: bool = False
    toy_num_images: int = 24
    max_images: int | None = None


class VocabularyConfig(BaseModel):
    lists_per_granularity: int = 5
    coarse_groups_file: Path | None = None
    fine_expansion_file: Path | None = None
    mixed_ratio: float = 0.5
    min_terms: int = 8
    max_terms: int = 32
    include_counterfactual: bool = True
    counterfactual_lists: int = 3
    counterfactual_noise_ratio: float = 0.2


class DetectorConfig(BaseModel):
    name: str
    enabled: bool = True
    model_id: str | None = None
    device: str = "cuda"
    score_threshold: float = 0.05
    max_detections: int = 300


class InferenceConfig(BaseModel):
    batch_size: int = 1
    num_workers: int = 0
    skip_existing: bool = True
    save_logits: bool = False
    num_shards: int = 1
    shard_index: int = 0
    split_from_env: bool = True


class EvaluationConfig(BaseModel):
    iou_threshold: float = 0.5
    bootstrap_samples: int = 300
    bootstrap_seed: int = 123


class RiskConfig(BaseModel):
    alphas: list[float] = Field(default_factory=lambda: [0.05, 0.1, 0.2])
    calibration_fraction: float = 0.25
    strategy: Literal["flat", "hierarchy", "groupwise", "both", "all"] = "all"
    hierarchy_relax_iou: float = 0.5
    delta: float = 0.05
    ucb_method: Literal["hoeffding", "clopper_pearson"] = "clopper_pearson"
    group_key: str = "timeofday"
    min_group_calibration_samples: int = 20


class ReportingConfig(BaseModel):
    figure_dpi: int = 180
    style: str = "whitegrid"
    title_prefix: str = "GSRD"


class ClusterConfig(BaseModel):
    slurm_array_var: str = "SLURM_ARRAY_TASK_ID"
    slurm_array_count_var: str = "SLURM_ARRAY_TASK_COUNT"


class RunConfig(BaseModel):
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)
    datasets: DatasetPrepareConfig = Field(default_factory=DatasetPrepareConfig)
    vocab: VocabularyConfig = Field(default_factory=VocabularyConfig)
    detectors: list[DetectorConfig] = Field(
        default_factory=lambda: [
            DetectorConfig(
                name="grounding_dino",
                model_id="IDEA-Research/grounding-dino-base",
                enabled=True,
            ),
            DetectorConfig(
                name="owlvit",
                model_id="google/owlvit-base-patch32",
                enabled=False,
            ),
            DetectorConfig(name="mock", enabled=False, model_id="mock-v1", device="cpu"),
        ]
    )
    inference: InferenceConfig = Field(default_factory=InferenceConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    reporting: ReportingConfig = Field(default_factory=ReportingConfig)
    cluster: ClusterConfig = Field(default_factory=ClusterConfig)
