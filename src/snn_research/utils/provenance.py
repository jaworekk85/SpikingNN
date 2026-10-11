"""Seed control and environment provenance."""

from __future__ import annotations

import platform
import subprocess
from pathlib import Path

import torch

import snn_research


def seed_everything(seed: int) -> torch.Generator:
    """Seed torch's global RNG (weight init) and return a generator for data/encoding."""
    torch.manual_seed(seed)
    return torch.Generator().manual_seed(seed)


def _git(*args: str) -> str | None:
    try:
        out = subprocess.run(["git", *args], capture_output=True, text=True, check=True,
                             cwd=Path(__file__).resolve().parent)
    except (OSError, subprocess.CalledProcessError):
        return None
    return out.stdout.strip()


def provenance(device: str) -> dict[str, object]:
    status = _git("status", "--porcelain")
    return {
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": None if status is None else bool(status),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "snn_research": snn_research.__version__,
        "platform": platform.platform(),
        "device": device,
    }
