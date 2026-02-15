from pathlib import Path

from gsrd.data.prepare import prepare_toy_dataset
from gsrd.utils.io import write_json, read_json
from gsrd.vocab.generate import generate_vocabularies


def test_vocab_generation_reproducible(tmp_path: Path) -> None:
    toy = prepare_toy_dataset(tmp_path / "data")
    cache_root = tmp_path / "cache"
    manifest = {"datasets": {toy["name"]: toy}}
    write_json(cache_root / "datasets" / "dataset_manifest.json", manifest)

    path1 = generate_vocabularies(cache_root, toy["name"], lists_per_granularity=3, seed=42, mixed_ratio=0.5)
    path2 = generate_vocabularies(cache_root, toy["name"], lists_per_granularity=3, seed=42, mixed_ratio=0.5)

    assert read_json(path1) == read_json(path2)
