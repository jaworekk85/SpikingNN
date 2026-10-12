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
from ..metrics import phase_quality, recovery_samples, synaptic_operations
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
    adaptation cost inside the ``during`` window: update steps, weights written, spikes,
    synaptic operations and input MACs, each reported as its own field."""
    correct: list[float] = []
    cost: dict[str, list[float]] = {"spikes": [], "synops": [], "input_macs": []}
    updates: list[tuple[int, int]] = []  # (first stream index of the batch, weights written)
    for index, inputs, y in stream.batches(config.batch_size, encoding, generator):
        logits, observations, _, written = forward_and_learn(model, rules, inputs, y, learn=config.adapt)
        correct += (logits.argmax(-1) == y).float().tolist()
        cost["spikes"] += sum(obs.spikes.flatten(2).sum((0, 2)) for obs in observations.values()).tolist()
        for key, value in synaptic_operations(model, observations, encoding.kind != "direct").items():
            cost[key] += value.tolist()
        if config.adapt:
            updates.append((int(index[0]), written))

    metrics: dict[str, float | None] = {"quality": sum(correct) / len(correct)}
    for k, shift in enumerate(stream.shifts):
        s = shift.schedule
        for phase, q in phase_quality(correct, s).items():
            metrics[f"shift{k}/quality_{phase}"] = q
        metrics[f"shift{k}/recovery_samples"] = recovery_samples(correct, s, config.q_target, config.window)
        in_window = [w for i, w in updates if s.start <= i < s.end]
        metrics[f"shift{k}/adaptation_updates"] = len(in_window)
        metrics[f"shift{k}/adaptation_weights_written"] = sum(in_window)
        for key, values in cost.items():
            metrics[f"shift{k}/adaptation_{key}"] = sum(values[s.start : s.end])
    traces = {"correct": correct, **{f"{key}_per_sample": v for key, v in cost.items()},
              "update_at": [float(i) for i, _ in updates]}
    return {"traces": traces, "metrics": metrics}
