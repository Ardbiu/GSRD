from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from gsrd.config.schema import RunConfig
from gsrd.utils.io import read_yaml


def _coerce_value(raw: str) -> Any:
    lowered = raw.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered == "none":
        return None
    try:
        if "." in raw:
            return float(raw)
        return int(raw)
    except ValueError:
        pass
    if raw.startswith("[") and raw.endswith("]"):
        values = [v.strip() for v in raw[1:-1].split(",") if v.strip()]
        return [_coerce_value(v) for v in values]
    return raw


def _apply_override(doc: dict[str, Any], key: str, value: Any) -> None:
    parts = key.split(".")
    cursor = doc
    for part in parts[:-1]:
        if part not in cursor or not isinstance(cursor[part], dict):
            cursor[part] = {}
        cursor = cursor[part]
    cursor[parts[-1]] = value


def load_config(config_path: Path, overrides: list[str] | None = None) -> RunConfig:
    payload = read_yaml(config_path)
    payload = deepcopy(payload)
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"Invalid override '{item}'. Expected key=value")
        key, raw = item.split("=", 1)
        _apply_override(payload, key, _coerce_value(raw))
    return RunConfig.model_validate(payload)
