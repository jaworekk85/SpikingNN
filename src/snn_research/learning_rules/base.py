"""Learning-rule interface and the restricted view that local kernels receive.

A rule binds to model parameters in ``init_state(model)``, computes explicit updates
in ``step(model, layer_states, modulators)``, commits them in ``apply(updates)`` and
clears its own state (never learned weights) in ``reset()``. ``model`` is used only
for binding; local numerical kernels receive a ``LocalObservation`` and nothing else.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import torch
from torch import nn

from .declaration import RuleDeclaration


class LearningRule(Protocol):
    declaration: RuleDeclaration

    def init_state(self, model: nn.Module) -> None: ...

    def step(self, model: nn.Module, layer_states: dict[str, Any], modulators: dict[str, torch.Tensor]) -> Any: ...

    def apply(self, updates: Any) -> None: ...

    def reset(self) -> None: ...


@dataclass(frozen=True)
class LocalObservation:
    """Quantities available at one layer's synapses: ``[T, batch, n]`` tensors."""

    pre: torch.Tensor  # presynaptic activity, flattened per sample
    post: torch.Tensor  # postsynaptic spikes
    post_potential: torch.Tensor  # postsynaptic membrane potential before reset


def local_observation(obs) -> LocalObservation:
    """Flatten a model ``LayerObservation`` into the restricted local view."""
    return LocalObservation(
        pre=obs.pre.flatten(2).detach(),
        post=obs.spikes.flatten(2).detach(),
        post_potential=obs.v.flatten(2).detach(),
    )


def linear_synapses(model: nn.Module) -> dict[str, nn.Linear]:
    """Map hidden-layer names to their dense synapses; non-dense layers (conv) stay fixed."""
    bound = {}
    for i, layer in enumerate(model.layers):
        linear = [m for m in layer.modules() if isinstance(m, nn.Linear)]
        if linear:
            bound[f"layer{i}"] = linear[0]
    return bound
