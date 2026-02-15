from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from gsrd.inference.schema import PredictionArtifactSpec
from gsrd.utils.io import ensure_dir, iter_jsonl, write_json, write_jsonl

LOGGER = logging.getLogger(__name__)


def shard_file_name(shard_index: int, num_shards: int) -> str:
    return f"shard-{shard_index:05d}-of-{num_shards:05d}.jsonl.gz"


def write_shard(
    cache_root: Path,
    artifact: PredictionArtifactSpec,
    shard_index: int,
    num_shards: int,
    rows: list[dict[str, Any]],
) -> Path:
    out_dir = ensure_dir(artifact.artifact_dir(cache_root))
    shard_path = out_dir / shard_file_name(shard_index, num_shards)
    write_jsonl(shard_path, rows)
    return shard_path


def shard_exists(
    cache_root: Path, artifact: PredictionArtifactSpec, shard_index: int, num_shards: int
) -> bool:
    path = artifact.artifact_dir(cache_root) / shard_file_name(shard_index, num_shards)
    return path.exists() and path.stat().st_size > 0


def write_metadata(cache_root: Path, artifact: PredictionArtifactSpec, payload: dict[str, Any]) -> Path:
    out_dir = ensure_dir(artifact.artifact_dir(cache_root))
    path = out_dir / "meta.json"
    write_json(path, payload)
    return path


def merge_shards(cache_root: Path, artifact: PredictionArtifactSpec, num_shards: int) -> Path:
    root = artifact.artifact_dir(cache_root)
    rows: list[dict[str, Any]] = []
    for idx in range(num_shards):
        shard = root / shard_file_name(idx, num_shards)
        if not shard.exists():
            raise FileNotFoundError(f"Missing shard: {shard}")
        rows.extend(iter_jsonl(shard))

    rows.sort(key=lambda x: int(x["image_id"]))
    out = root / "merged.jsonl.gz"
    write_jsonl(out, rows)
    LOGGER.info("Merged %d shards -> %s (%d rows)", num_shards, out, len(rows))
    return out
