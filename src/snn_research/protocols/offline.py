"""Offline protocol: train on the ``train`` split, then evaluate once on ``test``."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from ..datasets import Split
from ..encoding import EncodingConfig, encode
from ..metrics import accuracy, activity_summary


@dataclass(frozen=True)
class OfflineConfig:
    epochs: int = 5
    batch_size: int = 32

    def __post_init__(self) -> None:
        if self.epochs <= 0 or self.batch_size <= 0:
            raise ValueError("epochs and batch_size must be positive")


def evaluate(model, split: Split, encoding: EncodingConfig, batch_size: int, generator: torch.Generator) -> dict[str, float]:
    """Accuracy and activity over a whole split, without updating the model."""
    logits, labels = [], []
    layer_spikes: dict[str, list[torch.Tensor]] = {}
    with torch.no_grad():
        for x, y in split.batches(batch_size):
            out, observations = model(encode(encoding, x, generator).values)
            logits.append(out)
            labels.append(y)
            for name, obs in observations.items():
                layer_spikes.setdefault(name, []).append(obs.spikes)
    spikes = {name: torch.cat(s, dim=1) for name, s in layer_spikes.items()}
    return {"accuracy": accuracy(torch.cat(logits), torch.cat(labels)), **activity_summary(spikes)}


def run_offline(model, rule, train: Split, test: Split, encoding: EncodingConfig,
                config: OfflineConfig, generator: torch.Generator) -> dict:
    """Return per-epoch training traces and final test metrics."""
    if train.role != "train" or test.role != "test":
        raise ValueError("run_offline expects a 'train' split and a 'test' split")
    rule.init_state(model)
    traces: dict[str, list[float]] = {"train_loss": [], "train_accuracy": []}
    for _ in range(config.epochs):
        losses, correct, seen = [], 0.0, 0
        for x, y in train.batches(config.batch_size, generator):
            logits, observations = model(encode(encoding, x, generator).values)
            loss = F.cross_entropy(logits, y)
            rule.apply(rule.step(model, observations, {"loss": loss}))
            losses.append(loss.item())
            correct += accuracy(logits.detach(), y) * len(y)
            seen += len(y)
        traces["train_loss"].append(sum(losses) / len(losses))
        traces["train_accuracy"].append(correct / seen)
    return {"traces": traces, "test": evaluate(model, test, encoding, config.batch_size, generator)}
