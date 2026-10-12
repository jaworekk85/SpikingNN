"""Offline protocol: train on the ``train`` split, then evaluate once on ``test``."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

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


def forward_and_learn(model, rules: Sequence, x: torch.Tensor, y: torch.Tensor, encoding: EncodingConfig,
                      generator: torch.Generator, learn: bool = True):
    """One forward pass; then, if ``learn``, every rule updates from that same pass.

    Rules receive the same observations and modulators: ``loss`` (cross-entropy, with
    a graph only if some rule uses autograd) and ``error`` (``onehot(label) -
    softmax(logits)``, detached). Returned logits come from before the update.
    """
    needs_graph = learn and any(rule.declaration.uses_autograd for rule in rules)
    with torch.set_grad_enabled(needs_graph):
        logits, observations = model(encode(encoding, x, generator).values)
        loss = F.cross_entropy(logits, y)
    if learn:
        error = (F.one_hot(y, logits.shape[-1]) - logits.softmax(-1)).detach()
        modulators = {"loss": loss, "error": error}
        for rule in rules:
            rule.apply(rule.step(model, observations, modulators))
    return logits.detach(), observations, loss.item()


def train_epochs(model, rules: Sequence, train: Split, encoding: EncodingConfig,
                 config: OfflineConfig, generator: torch.Generator) -> dict[str, list[float]]:
    """Train bound rules on a ``train`` split; return per-epoch loss and accuracy."""
    if train.role != "train":
        raise ValueError("training requires a 'train' split")
    traces: dict[str, list[float]] = {"train_loss": [], "train_accuracy": []}
    for _ in range(config.epochs):
        losses, correct, seen = [], 0.0, 0
        for x, y in train.batches(config.batch_size, generator):
            logits, _, loss = forward_and_learn(model, rules, x, y, encoding, generator)
            losses.append(loss)
            correct += accuracy(logits, y) * len(y)
            seen += len(y)
        traces["train_loss"].append(sum(losses) / len(losses))
        traces["train_accuracy"].append(correct / seen)
    return traces


def run_offline(model, rules: Sequence, train: Split, test: Split, encoding: EncodingConfig,
                config: OfflineConfig, generator: torch.Generator) -> dict:
    """Train on ``train``, then evaluate once on ``test``. Rules must already be bound
    with ``init_state(model)``; protocols order learning and evaluation only."""
    if test.role != "test":
        raise ValueError("run_offline evaluates on a 'test' split")
    traces = train_epochs(model, rules, train, encoding, config, generator)
    return {"traces": traces, "test": evaluate(model, test, encoding, config.batch_size, generator)}
