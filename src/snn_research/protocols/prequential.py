"""Prequential (test-then-train) streaming protocol.

Each minibatch is predicted first; only then, if ``adapt`` is set, do rules learn
from that same forward pass and its labels. Labels never reach the model before its
prediction for that sample is recorded. Updates during the stream are adaptation,
not tuning on a held-out test split.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

from ..datasets import Stream
from ..encoding import EncodingConfig
from ..metrics import phase_quality, recovery_samples
from .offline import forward_and_learn


@dataclass(frozen=True)
class PrequentialConfig:
    length: int = 768
    batch_size: int = 8
    adapt: bool = True
    window: int = 64
    q_target: float = 0.8

    def __post_init__(self) -> None:
        if min(self.length, self.batch_size, self.window) <= 0:
            raise ValueError("length, batch_size and window must be positive")
        if not 0 <= self.q_target <= 1:
            raise ValueError("q_target must be in [0, 1]")


def run_prequential(model, rules: Sequence, stream: Stream, encoding: EncodingConfig,
                    config: PrequentialConfig, generator: torch.Generator) -> dict:
    """Per-sample traces and, for each shift ``k``, phase quality, recovery and
    adaptation cost (updates and spikes inside the ``during`` window, reported apart)."""
    correct: list[float] = []
    spikes: list[float] = []
    update_at: list[int] = []
    for index, inputs, y in stream.batches(config.batch_size, encoding, generator):
        logits, observations, _ = forward_and_learn(model, rules, inputs, y, learn=config.adapt)
        correct += (logits.argmax(-1) == y).float().tolist()
        spikes += sum(obs.spikes.flatten(2).sum((0, 2)) for obs in observations.values()).tolist()
        if config.adapt:
            update_at.append(int(index[0]))

    metrics: dict[str, float | None] = {"quality": sum(correct) / len(correct)}
    for k, shift in enumerate(stream.shifts):
        s = shift.schedule
        for phase, q in phase_quality(correct, s).items():
            metrics[f"shift{k}/quality_{phase}"] = q
        metrics[f"shift{k}/recovery_samples"] = recovery_samples(correct, s, config.q_target, config.window)
        metrics[f"shift{k}/adaptation_updates"] = sum(s.start <= i < s.end for i in update_at)
        metrics[f"shift{k}/adaptation_spikes"] = sum(spikes[s.start : s.end])
    traces = {"correct": correct, "spikes_per_sample": spikes, "update_at": [float(i) for i in update_at]}
    return {"traces": traces, "metrics": metrics}
