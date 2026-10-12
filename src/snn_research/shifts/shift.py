"""Seeded, scheduled distribution shifts applied to samples by stream index.

A ``Schedule`` maps a stream index to a shift strength in ``[0, 1]`` and a phase:
``before`` (``i < start``), ``during`` (``start <= i < end``) and ``after`` (``i >= end``).
Sample ``start`` is the first shifted sample.

- ``abrupt``: strength 1 from ``start`` on; ``[start, end)`` is the adaptation window.
- ``gradual``: linear ramp from 0 at ``start`` to 1 at ``end``, then 1.
- ``incremental``: ``steps`` equal increments over ``[start, end)``, then 1.
- ``recurring``: alternating on/off blocks of ``period`` samples in ``[start, end)``,
  starting "on"; 0 after ``end`` (the original condition returns).

Shift kinds (``magnitude`` is the value at full strength):

- ``noise`` (covariate, sensor-noise-like): add ``strength * magnitude * N(0, 1)``.
- ``intensity`` (covariate, illumination-like; changes the input event rate under rate
  coding): multiply by ``1 + strength * (magnitude - 1)``.
- ``label_permutation`` (concept shift): with probability ``strength * magnitude``,
  relabel ``y -> (y + 1) % n_classes``.
- ``latency_jitter`` (temporal dynamics): after encoding, each input channel's whole
  train is delayed by a random integer in ``[0, round(strength * magnitude)]`` steps;
  activity shifted past ``T`` is lost and the start is zero-filled.

``shift_and_encode`` applies sample-level kinds in their saved order, encodes, then
applies temporal kinds in their saved order. New classes/tasks are covered by the
task-sequence protocol, not by a transform.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from ..encoding import EncodingConfig, encode

TIMINGS = ("abrupt", "gradual", "incremental", "recurring")
SAMPLE_KINDS = ("noise", "intensity", "label_permutation")
TEMPORAL_KINDS = ("latency_jitter",)
KINDS = SAMPLE_KINDS + TEMPORAL_KINDS


@dataclass(frozen=True)
class Schedule:
    timing: str = "abrupt"
    start: int = 256
    end: int = 512
    steps: int = 4
    period: int = 64

    def __post_init__(self) -> None:
        if self.timing not in TIMINGS:
            raise ValueError(f"timing must be one of {TIMINGS}, got {self.timing!r}")
        if not 0 <= self.start < self.end:
            raise ValueError("schedule requires 0 <= start < end")
        if self.steps < 1 or self.period < 1:
            raise ValueError("steps and period must be positive")

    def strength(self, index: torch.Tensor) -> torch.Tensor:
        i = index.float()
        progress = ((i - self.start) / (self.end - self.start)).clamp(0, 1)
        if self.timing == "abrupt":
            return (i >= self.start).float()
        if self.timing == "gradual":
            return progress
        if self.timing == "incremental":
            level = (torch.floor(progress * self.steps) + 1).clamp(max=self.steps) / self.steps
            return torch.where(i < self.start, 0.0, level)
        on = ((i - self.start) // self.period) % 2 == 0
        return ((i >= self.start) & (i < self.end) & on).float()

    def phase_at(self, index: int) -> str:
        if index < self.start:
            return "before"
        return "during" if index < self.end else "after"


@dataclass(frozen=True)
class ShiftConfig:
    kind: str = "noise"
    magnitude: float = 0.3
    seed: int = 0
    schedule: Schedule = field(default_factory=Schedule)

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ValueError(f"shift kind must be one of {KINDS}, got {self.kind!r}")
        if self.magnitude < 0:
            raise ValueError("magnitude must be non-negative")


def apply_shift(shift: ShiftConfig, x: torch.Tensor, y: torch.Tensor, strength: torch.Tensor,
                generator: torch.Generator, n_classes: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Shift a batch ``x [n, ...]``, ``y [n]`` with per-sample ``strength [n]``."""
    if shift.kind not in SAMPLE_KINDS:
        raise ValueError(f"{shift.kind!r} acts on encoded input; use shift_and_encode")
    s = strength.view(-1, *([1] * (x.dim() - 1)))
    if shift.kind == "noise":
        noise = torch.randn(x.shape, generator=generator)
        return (x + s * shift.magnitude * noise).clamp(0, 1), y
    if shift.kind == "intensity":
        return (x * (1 + s * (shift.magnitude - 1))).clamp(0, 1), y
    flip = torch.rand(y.shape, generator=generator) < strength * shift.magnitude
    return x, torch.where(flip, (y + 1) % n_classes, y)


def delay_inputs(values: torch.Tensor, max_delay: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    """Delay each channel of ``values [T, batch, ...]`` by a random integer in
    ``[0, max_delay[b]]`` steps, zero-filling the start."""
    T = values.shape[0]
    k = max_delay.round().long().view(-1, *([1] * (values.dim() - 2)))
    delay = (torch.rand(values.shape[1:], generator=generator) * (k + 1)).floor().long()
    source = torch.arange(T).view(T, *([1] * (values.dim() - 1))) - delay
    return values.gather(0, source.clamp(min=0)) * (source >= 0)


def shift_and_encode(shifts: tuple[ShiftConfig, ...], x: torch.Tensor, y: torch.Tensor,
                     strengths: list[torch.Tensor], generators: list[torch.Generator],
                     encoding: EncodingConfig, encode_generator: torch.Generator,
                     n_classes: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample-level shifts -> encoding -> temporal shifts; returns ``(inputs, y)``."""
    for shift, s, g in zip(shifts, strengths, generators):
        if shift.kind in SAMPLE_KINDS:
            x, y = apply_shift(shift, x, y, s, g, n_classes)
    inputs = encode(encoding, x, encode_generator).values
    for shift, s, g in zip(shifts, strengths, generators):
        if shift.kind == "latency_jitter":
            inputs = delay_inputs(inputs, s * shift.magnitude, g)
    return inputs, y
