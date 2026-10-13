"""The hardware boundary: backend interface, measurements and constraint checks.

No backend is implemented here. A future backend (e.g. for a partner's chip) maps an
export onto hardware, runs it and measures energy; its measurements are the only
source of ``RunRecord.energy_j``. Hardware capabilities enter as data
(``HardwareConstraints``, typically a YAML draft marked "to be confirmed"), never as
assumptions in framework code.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, Sequence

import torch

KNOWN_STATUSES = ("to be confirmed", "confirmed by partner")


@dataclass(frozen=True)
class Measurement:
    """A hardware energy measurement; maps onto ``RunRecord.energy_j`` / ``measurement``."""

    energy_j: float
    source: str  # e.g. on-board power monitor, external meter
    method: str  # e.g. integrated power over the run minus idle baseline
    boundary: str  # what is included: chip, board, host, I/O ...
    units: str = "J"

    def record_fields(self) -> dict[str, Any]:
        return {"energy_j": self.energy_j, "backend_kind": "hardware",
                "measurement": {"source": self.source, "method": self.method,
                                "boundary": self.boundary, "units": self.units}}


class HardwareBackend(Protocol):
    """Contract for a future hardware backend. ``export`` comes from
    ``backends.export.export_network``; ``run`` returns at least ``logits`` and
    per-layer spike counts; ``measure`` returns a ``Measurement`` for the last run."""

    name: str

    def load(self, description: dict[str, Any], weights: dict[str, Any]) -> None: ...

    def run(self, inputs: torch.Tensor) -> dict[str, Any]: ...

    def measure(self) -> Measurement: ...


@dataclass(frozen=True)
class HardwareConstraints:
    """Target-hardware limits as data. ``None`` = unknown (the check is skipped)."""

    name: str
    status: str = "to be confirmed"
    sources: tuple[str, ...] = ()
    neuron_models: tuple[str, ...] | None = None
    resets: tuple[str, ...] | None = None
    dt_ms: tuple[float, ...] | None = None
    weight_bits: int | None = None
    max_fan_in: int | None = None
    cores: int | None = None
    memory_per_core_bytes: int | None = None
    on_chip_learning: str | None = None  # e.g. "software on cores", "none"
    state_bits: int = 32  # assumed storage per rule-state value for memory estimates

    def __post_init__(self) -> None:
        if self.status not in KNOWN_STATUSES:
            raise ValueError(f"status must be one of {KNOWN_STATUSES}")


def _rule_state_bytes(description: dict[str, Any], state_bits: int) -> int:
    """Rough memory for rule state: values per synapse/neuron x ``state_bits``."""
    synapses, neurons = description["totals"]["synapses"], description["totals"]["neurons"]
    values = 0
    for rule in description["rules"]:
        state = rule["state"]
        values += len(state.get("per_synapse", [])) * synapses
        values += (len(state.get("per_presynaptic_neuron", [])) + len(state.get("per_postsynaptic_neuron", []))) * neurons
    return values * state_bits // 8


def check_constraints(description: dict[str, Any], hw: HardwareConstraints) -> list[tuple[str, str]]:
    """``(level, message)`` findings, level ``error`` | ``warning`` | ``info``."""
    findings: list[tuple[str, str]] = [("info", f"constraints for {hw.name} are '{hw.status}'")]
    neuron = description["neuron"]
    if hw.neuron_models is not None and neuron["model"] not in hw.neuron_models:
        findings.append(("error", f"neuron model {neuron['model']} not in {list(hw.neuron_models)}"))
    if hw.resets is not None and neuron["reset"] not in hw.resets:
        findings.append(("error", f"reset '{neuron['reset']}' not in {list(hw.resets)}"))
    if hw.dt_ms is not None and description["timing"]["dt_ms"] not in hw.dt_ms:
        findings.append(("warning", f"dt {description['timing']['dt_ms']} ms not in {list(hw.dt_ms)}"))
    if hw.max_fan_in is not None and description["totals"]["max_fan_in"] > hw.max_fan_in:
        findings.append(("error", f"max fan-in {description['totals']['max_fan_in']} > {hw.max_fan_in}"))

    local_rules = [r for r in description["rules"] if not r["declaration"]["uses_autograd"]]
    if hw.weight_bits is not None:
        for rule in local_rules:
            bits = rule["parameters"].get("weight_bits", 0)
            if not bits:
                findings.append(("warning", f"{rule['declaration']['name']} learns float weights; hardware "
                                            f"stores {hw.weight_bits}-bit weights (set weight_bits, w_max)"))
            elif bits > hw.weight_bits:
                findings.append(("error", f"{rule['declaration']['name']} uses {bits}-bit weights > {hw.weight_bits}"))
        if not local_rules:
            findings.append(("warning", f"weights are float32; quantize to {hw.weight_bits} bits before deployment"))
    if any(r["declaration"]["uses_autograd"] for r in description["rules"]):
        findings.append(("info", "an autograd rule (surrogate BPTT) is an offline reference; deploy inference only"))
    if local_rules and hw.on_chip_learning is not None:
        level = "error" if hw.on_chip_learning == "none" else "info"
        findings.append((level, f"on-chip learning: {hw.on_chip_learning}"))
    if description["readout"]["type"].startswith("non-spiking"):
        findings.append(("info", "readout is a non-spiking rate readout; map it to spike counters + host/core "
                                 "arithmetic"))

    if hw.cores is not None and hw.memory_per_core_bytes is not None:
        bits = hw.weight_bits or 32
        weight_bytes = description["totals"]["synapses"] * bits // 8
        state_bytes = _rule_state_bytes(description, hw.state_bits)
        capacity = hw.cores * hw.memory_per_core_bytes
        level = "error" if weight_bytes + state_bytes > capacity else "info"
        findings.append((level, f"estimated memory {weight_bytes + state_bytes:,} B (weights {weight_bytes:,} B at "
                                f"{bits} bit, rule state {state_bytes:,} B) vs on-chip {capacity:,} B "
                                f"(logical synapses, no routing/code overhead)"))
    return findings


def format_findings(findings: Sequence[tuple[str, str]]) -> str:
    return "\n".join(f"- **{level}**: {message}" for level, message in findings)
