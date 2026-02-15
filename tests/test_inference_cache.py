from pathlib import Path

from gsrd.inference.cache import merge_shards, write_shard
from gsrd.inference.schema import PredictionArtifactSpec
from gsrd.utils.io import iter_jsonl


def test_merge_shards(tmp_path: Path) -> None:
    artifact = PredictionArtifactSpec(
        detector="mock",
        dataset="toy",
        granularity="standard",
        vocab_id="abc123",
    )
    write_shard(
        tmp_path,
        artifact,
        shard_index=0,
        num_shards=2,
        rows=[{"image_id": 2, "file_name": "b.jpg", "detections": []}],
    )
    write_shard(
        tmp_path,
        artifact,
        shard_index=1,
        num_shards=2,
        rows=[{"image_id": 1, "file_name": "a.jpg", "detections": []}],
    )

    merged = merge_shards(tmp_path, artifact, num_shards=2)
    rows = list(iter_jsonl(merged))
    assert [r["image_id"] for r in rows] == [1, 2]
