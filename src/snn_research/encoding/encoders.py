"""Static-input encoders producing ``[T, batch, ...]`` tensors.

Static inputs are intensities in ``[0, 1]``. ``rate`` and ``latency`` emit binary
spikes; ``direct`` repeats the analogue value as input current and reports no event
rate. ``events`` passes through samples that already have a time axis
(``[batch, T, ...]`` binned event frames), drawing a spike with probability equal to
each value: binary frames pass unchanged, while shifted (fractional) frames become
dropped or added events.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

KINDS = ("rate", "latency", "direct", "events")


@dataclass(frozen=True)
class EncodingConfig:
    kind: str = "rate"
    T: int = 20

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ValueError(f"encoding kind must be one of {KINDS}, got {self.kind!r}")
        if self.T <= 0:
            raise ValueError("T must be positive")


@dataclass(frozen=True)
class EncodedInput:
    values: torch.Tensor  # [T, batch, ...]
    kind: str
    is_spikes: bool

    @property
    def T(self) -> int:
        return self.values.shape[0]

    @property
    def event_rate(self) -> float | None:
        """Input events per input-element per timestep; ``None`` for current input."""
        return self.values.mean().item() if self.is_spikes else None


def rate_encode(x: torch.Tensor, T: int, generator: torch.Generator) -> EncodedInput:
    """Bernoulli spikes with per-step probability equal to the clipped intensity."""
    p = x.clamp(0, 1).unsqueeze(0).expand(T, *x.shape)
    return EncodedInput(torch.bernoulli(p, generator=generator), "rate", is_spikes=True)


def latency_encode(x: torch.Tensor, T: int) -> EncodedInput:
    """One spike per element at ``t = round((1 - x) * (T - 1))``; zero intensity is silent."""
    x = x.clamp(0, 1)
    t_spike = torch.round((1 - x) * (T - 1)).long()
    steps = torch.arange(T, device=x.device).view(T, *([1] * x.dim()))
    spikes = ((steps == t_spike) & (x > 0)).to(x.dtype)
    return EncodedInput(spikes, "latency", is_spikes=True)


def direct_encode(x: torch.Tensor, T: int) -> EncodedInput:
    """The same analogue current at every timestep; not spike events."""
    return EncodedInput(x.unsqueeze(0).expand(T, *x.shape).clone(), "direct", is_spikes=False)


def events_encode(x: torch.Tensor, generator: torch.Generator) -> EncodedInput:
    """``[batch, T, ...]`` event frames -> ``[T, batch, ...]`` spikes (Bernoulli on values)."""
    p = x.clamp(0, 1).transpose(0, 1)
    return EncodedInput(torch.bernoulli(p, generator=generator), "events", is_spikes=True)


def encode(config: EncodingConfig, x: torch.Tensor, generator: torch.Generator) -> EncodedInput:
    if config.kind == "events":
        if x.shape[1] != config.T:
            raise ValueError(f"event samples have {x.shape[1]} time bins but encoding T is {config.T}")
        return events_encode(x, generator)
    if config.kind == "rate":
        return rate_encode(x, config.T, generator)
    if config.kind == "latency":
        return latency_encode(x, config.T)
    return direct_encode(x, config.T)
