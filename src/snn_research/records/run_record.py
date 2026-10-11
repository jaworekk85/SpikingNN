"""RunRecord schema, validation and the immutable result store.

Neuron, rule, encoding and protocol parameters are stored in the full resolved
``config``; the remaining fields hold provenance and outputs that a config cannot.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

MEASUREMENT_KEYS = ("source", "method", "boundary", "units")


@dataclass(frozen=True)
class RunRecord:
    experiment: str
    config: dict[str, Any]
    seed: int
    T: int
    provenance: dict[str, Any]  # git commit/dirty flag, versions, device
    rule_declaration: dict[str, Any]
    model: dict[str, Any]  # kind, neuron and weight counts
    dataset: dict[str, Any]  # version and split sizes
    protocol: str
    metrics: dict[str, float]
    traces: dict[str, list[float]]
    backend_kind: str = "simulation"
    shift: dict[str, Any] | None = None
    energy_proxy: float | None = None
    proxy_assumptions: dict[str, Any] | None = None
    energy_j: float | None = None
    measurement: dict[str, Any] | None = None
    created_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))

    def __post_init__(self) -> None:
        if not self.experiment or "/" in self.experiment or "\\" in self.experiment:
            raise ValueError("experiment must be a non-empty name without path separators")
        if self.backend_kind not in ("simulation", "hardware"):
            raise ValueError("backend_kind must be 'simulation' or 'hardware'")
        if self.energy_proxy is not None and not self.proxy_assumptions:
            raise ValueError("energy_proxy requires explicit proxy_assumptions")
        if self.energy_j is not None:
            if self.backend_kind != "hardware":
                raise ValueError("energy_j is only allowed from a hardware backend")
            missing = [k for k in MEASUREMENT_KEYS if not (self.measurement or {}).get(k)]
            if missing:
                raise ValueError(f"energy_j requires measurement metadata: missing {missing}")
        json.dumps(asdict(self))  # must be JSON-serialisable


def save_run(record: RunRecord, root: str | Path = "results") -> Path:
    """Write a record to a new ``<root>/<experiment>/<timestamp>_<hash>/`` directory.

    The directory is created with ``exist_ok=False``; an existing path is an error.
    """
    payload = json.dumps(asdict(record), sort_keys=True, indent=2)
    digest = hashlib.sha256(payload.encode() + os.urandom(16)).hexdigest()[:8]
    run_dir = Path(root) / record.experiment / f"{record.created_utc}_{digest}"
    run_dir.parent.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(exist_ok=False)
    (run_dir / "record.json").write_text(payload, encoding="utf-8")
    return run_dir
