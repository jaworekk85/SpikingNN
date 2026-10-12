"""Logical synaptic-operation counting (definitions in docs/metrics.md).

A synaptic operation is one spike delivered over one existing connection: each
presynaptic spike counts its fan-out. Fan-out is the number of postsynaptic targets:
``out_features`` for dense layers; for convolutions, the number of (output channel,
output position) pairs whose receptive field contains the input position. Padding
positions are not neurons and emit nothing; weight sharing does not reduce the count
(shared weights still drive separate connections).

Non-spiking input (direct current) drives multiply-accumulates instead; those are
counted separately as ``input_macs`` (one per non-zero input value per target per
timestep), never mixed into synaptic operations. The readout accumulates the last
hidden layer's spikes, so each of those spikes counts ``n_classes`` operations.
Counts are logical events, not GPU multiply-adds or runtime.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def fan_out(layer: nn.Module, input_shape: tuple[int, ...]) -> torch.Tensor:
    """Fan-out of every input element of ``layer`` for one sample of ``input_shape``."""
    synapse = next(m for m in layer.modules() if isinstance(m, (nn.Linear, nn.Conv2d)))
    if isinstance(synapse, nn.Linear):
        return torch.full(input_shape, float(synapse.out_features))
    c, h, w = input_shape
    out = synapse(torch.zeros(1, *input_shape)).shape[1:]
    (sh, sw), (ph, pw), (kh, kw) = synapse.stride, synapse.padding, synapse.kernel_size
    extra = (h - ((out[1] - 1) * sh - 2 * ph + kh), w - ((out[2] - 1) * sw - 2 * pw + kw))
    ones_kernel = torch.ones(synapse.out_channels, c, kh, kw)
    counts = F.conv_transpose2d(torch.ones(1, *out), ones_kernel, stride=synapse.stride,
                                padding=synapse.padding, output_padding=extra)
    return counts[0]


def synapse_count(model: nn.Module, observations: dict) -> int:
    """Logical synapses (connections) per sample, including the readout."""
    total = sum(fan_out(layer, tuple(obs.pre.shape[2:])).sum() for layer, obs
                in zip(model.layers, observations.values()))
    last = list(observations.values())[-1]
    return int(total + last.spikes[0, 0].numel() * model.readout.out_features)


def synaptic_operations(model: nn.Module, observations: dict, input_is_spikes: bool) -> dict[str, torch.Tensor]:
    """Per-sample counts (``[batch]`` tensors) for one forward pass: totals ``synops`` and
    ``input_macs``, plus ``synops/<layer>`` for the synapses into each layer and
    ``synops/readout``."""
    batch = next(iter(observations.values())).spikes.shape[1]
    counts = {"synops": torch.zeros(batch), "input_macs": torch.zeros(batch)}
    for i, (name, layer, obs) in enumerate(zip(observations, model.layers, observations.values())):
        fan = fan_out(layer, tuple(obs.pre.shape[2:]))
        if i == 0 and not input_is_spikes:
            counts["input_macs"] += ((obs.pre != 0) * fan).flatten(2).sum((0, 2))
            counts[f"synops/{name}"] = torch.zeros(batch)
        else:
            counts[f"synops/{name}"] = (obs.pre * fan).flatten(2).sum((0, 2))
    last = list(observations.values())[-1]
    counts["synops/readout"] = last.spikes.flatten(2).sum((0, 2)) * model.readout.out_features
    counts["synops"] = sum(v for k, v in counts.items() if k.startswith("synops/"))
    return counts
