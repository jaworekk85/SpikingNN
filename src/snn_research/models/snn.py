"""Small feed-forward SNNs: synapse layers, each followed by LIF neurons.

``SpikingNet.forward`` takes ``[T, batch, ...]`` input and returns logits plus a
per-layer observation of presynaptic input, spikes and membrane potential. The readout is a linear
layer on the last hidden layer's time-averaged spike rate. Neuron state is created
fresh in every forward call, so no state or graph leaks between samples.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from ..neurons import LIF, LIFParams


@dataclass(frozen=True)
class ModelConfig:
    kind: str = "mlp"
    hidden: tuple[int, ...] = (32,)
    conv_channels: int = 8
    init_gain: float = 1.0  # multiplies PyTorch's default init of hidden synapses

    def __post_init__(self) -> None:
        if self.kind not in ("mlp", "conv"):
            raise ValueError(f"model kind must be 'mlp' or 'conv', got {self.kind!r}")
        if not self.hidden or any(h <= 0 for h in self.hidden) or self.conv_channels <= 0:
            raise ValueError("layer sizes must be positive")
        if self.init_gain <= 0:
            raise ValueError("init_gain must be positive")


@dataclass
class LayerObservation:
    pre: torch.Tensor  # [T, batch, ...] input activity fed into the layer's synapses
    spikes: torch.Tensor  # [T, batch, ...]
    v: torch.Tensor  # [T, batch, ...] potential before reset (the value compared with threshold)


class SpikingNet(nn.Module):
    def __init__(self, layers: list[nn.Module], readout: nn.Linear, lif: LIFParams) -> None:
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.readout = readout
        self.lif = LIF(lif)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, LayerObservation]]:
        states = [None] * len(self.layers)
        record: list[tuple[list, list, list]] = [([], [], []) for _ in self.layers]
        for x_t in x:
            h = x_t
            for i, layer in enumerate(self.layers):
                pre, spikes, vs = record[i]
                pre.append(h)
                current = layer(h)
                if states[i] is None:
                    states[i] = self.lif.init_state(current)
                h, states[i] = self.lif.step(current, states[i])
                spikes.append(h)
                vs.append(states[i].u)
        observations = {
            f"layer{i}": LayerObservation(*(torch.stack(r) for r in rec)) for i, rec in enumerate(record)
        }
        last_rate = observations[f"layer{len(self.layers) - 1}"].spikes.mean(0)
        return self.readout(last_rate.flatten(1)), observations


def build_model(config: ModelConfig, input_shape: tuple[int, ...], n_classes: int, lif: LIFParams) -> SpikingNet:
    """MLP: flat input -> hidden LIF layers. Conv: one stride-2 conv LIF layer -> hidden LIF layers."""
    layers: list[nn.Module] = []
    if config.kind == "mlp":
        n_in = int(torch.tensor(input_shape).prod())
        first: nn.Module = nn.Sequential(nn.Flatten(), nn.Linear(n_in, config.hidden[0]))
    else:
        if len(input_shape) != 3:
            raise ValueError("conv model expects input_shape (channels, height, width)")
        c, h, w = input_shape
        conv = nn.Conv2d(c, config.conv_channels, kernel_size=3, stride=2, padding=1)
        layers.append(conv)
        n_conv = config.conv_channels * ((h + 1) // 2) * ((w + 1) // 2)
        first = nn.Sequential(nn.Flatten(), nn.Linear(n_conv, config.hidden[0]))
    layers.append(first)
    layers += [nn.Linear(a, b) for a, b in zip(config.hidden, config.hidden[1:])]
    with torch.no_grad():
        for layer in layers:
            for m in layer.modules():
                if isinstance(m, (nn.Linear, nn.Conv2d)):
                    m.weight.mul_(config.init_gain)
    return SpikingNet(layers, nn.Linear(config.hidden[-1], n_classes), lif)


def count_neurons(observations: dict[str, LayerObservation]) -> int:
    return sum(obs.spikes[0, 0].numel() for obs in observations.values())
