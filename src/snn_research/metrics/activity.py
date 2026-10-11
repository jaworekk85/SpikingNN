"""Task-quality and spike-activity metrics (definitions in docs/metrics.md)."""

from __future__ import annotations

import torch


def accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    return (logits.argmax(-1) == labels).float().mean().item()


def spike_rate(spikes: torch.Tensor) -> float:
    """Spikes / (neurons x timesteps x batch) for a ``[T, batch, ...]`` spike tensor."""
    return spikes.sum().item() / spikes.numel()


def temporal_sparsity(spikes: torch.Tensor) -> float:
    """Fraction of inactive neuron-time pairs."""
    return (spikes == 0).float().mean().item()


def activity_summary(layer_spikes: dict[str, torch.Tensor]) -> dict[str, float]:
    """Per-layer spike rate and sparsity, plus overall values weighted by layer size."""
    summary: dict[str, float] = {}
    total_spikes, total_slots = 0.0, 0
    for name, spikes in layer_spikes.items():
        summary[f"{name}/spike_rate"] = spike_rate(spikes)
        summary[f"{name}/temporal_sparsity"] = temporal_sparsity(spikes)
        total_spikes += spikes.sum().item()
        total_slots += spikes.numel()
    summary["overall/spike_rate"] = total_spikes / total_slots
    summary["overall/temporal_sparsity"] = (
        sum((s == 0).sum().item() for s in layer_spikes.values()) / total_slots
    )
    return summary
