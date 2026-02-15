from __future__ import annotations

import platform
import subprocess
from pathlib import Path
from typing import Any


def git_commit_or_none(workdir: Path) -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=workdir, stderr=subprocess.DEVNULL
        )
        return out.decode("utf-8").strip()
    except Exception:
        return None


def hardware_summary() -> dict[str, Any]:
    summary: dict[str, Any] = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu": platform.processor(),
    }
    try:
        import torch

        summary["torch_cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            summary["gpu_count"] = torch.cuda.device_count()
            summary["gpus"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        else:
            summary["gpu_count"] = 0
            summary["gpus"] = []
    except Exception:
        summary["torch_cuda_available"] = False
        summary["gpu_count"] = 0
        summary["gpus"] = []
    return summary
