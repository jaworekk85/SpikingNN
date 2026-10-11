"""Split-aware data containers and a seeded synthetic classification task."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import torch

ROLES = ("train", "tune", "test")


@dataclass(frozen=True)
class Split:
    x: torch.Tensor  # [n, ...], intensities in [0, 1]
    y: torch.Tensor  # [n]
    role: str
    version: str

    def __post_init__(self) -> None:
        if self.role not in ROLES:
            raise ValueError(f"role must be one of {ROLES}, got {self.role!r}")
        if len(self.x) != len(self.y):
            raise ValueError("x and y must have the same length")

    def batches(self, batch_size: int, generator: torch.Generator | None = None) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        """Yield minibatches; shuffled only when a generator is given."""
        order = torch.randperm(len(self.y), generator=generator) if generator is not None else torch.arange(len(self.y))
        for i in range(0, len(order), batch_size):
            idx = order[i : i + batch_size]
            yield self.x[idx], self.y[idx]


@dataclass(frozen=True)
class SyntheticConfig:
    n_train: int = 256
    n_test: int = 128
    n_classes: int = 4
    shape: tuple[int, ...] = (16,)
    noise: float = 0.15
    seed: int = 0

    def __post_init__(self) -> None:
        if min(self.n_train, self.n_test, self.n_classes, *self.shape) <= 0 or self.noise < 0:
            raise ValueError("synthetic dataset sizes must be positive and noise non-negative")


def synthetic_prototypes(config: SyntheticConfig) -> tuple[Split, Split]:
    """Noisy samples around one random intensity prototype per class (no downloads)."""
    g = torch.Generator().manual_seed(config.seed)
    prototypes = torch.rand(config.n_classes, *config.shape, generator=g)

    def make(n: int, role: str) -> Split:
        y = torch.randint(config.n_classes, (n,), generator=g)
        x = (prototypes[y] + config.noise * torch.randn(n, *config.shape, generator=g)).clamp(0, 1)
        return Split(x, y, role, version=f"synthetic_prototypes-v1-seed{config.seed}")

    return make(config.n_train, "train"), make(config.n_test, "test")
