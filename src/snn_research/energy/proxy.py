"""Energy proxy from operation counts with explicit, saved assumptions.

This is an estimate, never a measurement: it multiplies logical operation counts by
per-operation energies. The default coefficients are the widely reused 45 nm, 32-bit
floating-point arithmetic costs (accumulate 0.9 pJ, multiply-accumulate 4.6 pJ;
Horowitz, ISSCC 2014, as used by Han et al. 2015 and many SNN papers). They cover
arithmetic only: memory access, data movement, I/O, idle power and implementation
overhead are excluded, and memory access often dominates on real hardware.
Neuron-state updates and learning updates are not costed by default (0 pJ) and are
reported as separate counts.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class EnergyProxyConfig:
    pj_per_synop: float = 0.9  # one spike-driven accumulate
    pj_per_mac: float = 4.6  # one multiply-accumulate (non-spiking input)
    pj_per_neuron_update: float = 0.0  # one neuron state update per timestep
    pj_per_weight_update: float = 0.0  # one weight written by a learning rule
    source: str = "Horowitz ISSCC 2014 / Han et al. 2015: 45 nm FP32 arithmetic only, no memory access"

    def __post_init__(self) -> None:
        if min(self.pj_per_synop, self.pj_per_mac, self.pj_per_neuron_update, self.pj_per_weight_update) < 0:
            raise ValueError("energy coefficients must be non-negative")
        if not self.source:
            raise ValueError("the source of the energy assumptions must be stated")

    def assumptions(self) -> dict[str, object]:
        return {**asdict(self), "units": "pJ", "kind": "proxy (not measured)"}


def energy_proxy_pj(config: EnergyProxyConfig, synops: float = 0.0, macs: float = 0.0,
                    neuron_updates: float = 0.0, weight_updates: float = 0.0) -> dict[str, float]:
    """Per-component and total proxy energy in pJ for the given operation counts."""
    parts = {
        "synops_pj": synops * config.pj_per_synop,
        "macs_pj": macs * config.pj_per_mac,
        "neuron_updates_pj": neuron_updates * config.pj_per_neuron_update,
        "weight_updates_pj": weight_updates * config.pj_per_weight_update,
    }
    return {**parts, "total_pj": sum(parts.values())}
