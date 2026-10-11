"""Surrogate-gradient backpropagation through time (global gradient, not local)."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .declaration import RuleDeclaration


@dataclass(frozen=True)
class SurrogateBPTTConfig:
    lr: float = 1e-2

    def __post_init__(self) -> None:
        if self.lr <= 0:
            raise ValueError("lr must be positive")


class SurrogateBPTT:
    """Adam on gradients of a loss passed as the ``"loss"`` modulator.

    ``step`` back-propagates through all timesteps and layers and returns the
    gradients; ``apply`` commits them. ``reset`` clears optimizer state only.
    """

    declaration = RuleDeclaration(
        name="surrogate_bptt",
        uses_autograd=True,
        locality="global_gradient",
        online_capable=False,
        state=("adam_moments",),
        modulator_source="error",
    )

    def __init__(self, config: SurrogateBPTTConfig) -> None:
        self.config = config
        self.params: list[nn.Parameter] = []
        self.optimizer: torch.optim.Optimizer | None = None

    def init_state(self, model: nn.Module) -> None:
        self.params = [p for p in model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.Adam(self.params, lr=self.config.lr)

    def step(self, model, layer_states, modulators: dict[str, torch.Tensor]) -> list[torch.Tensor]:
        self.optimizer.zero_grad()
        modulators["loss"].backward()
        return [p.grad for p in self.params]

    def apply(self, updates: list[torch.Tensor]) -> None:
        self.optimizer.step()

    def reset(self) -> None:
        self.optimizer = torch.optim.Adam(self.params, lr=self.config.lr)
