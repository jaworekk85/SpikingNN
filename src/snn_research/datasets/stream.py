"""Ordered, seeded sample streams with scheduled shifts."""

from __future__ import annotations

from typing import Iterator

import torch

from ..shifts import ShiftConfig, apply_shift
from .synthetic import Split


class Stream:
    """``length`` samples drawn from a ``train``-role split in a seeded order.

    The order is a concatenation of seeded permutations of the split. Shifts are
    applied by stream index, in the given order, each with its own seeded generator.
    """

    def __init__(self, split: Split, length: int, n_classes: int, seed: int,
                 shifts: tuple[ShiftConfig, ...] = ()) -> None:
        if split.role != "train":
            raise ValueError("streams provide adaptation data and require a 'train' split")
        if length <= 0:
            raise ValueError("length must be positive")
        g = torch.Generator().manual_seed(seed)
        repeats = -(-length // len(split.y))
        self.order = torch.cat([torch.randperm(len(split.y), generator=g) for _ in range(repeats)])[:length]
        self.split, self.length, self.n_classes, self.shifts = split, length, n_classes, shifts

    def batches(self, batch_size: int) -> Iterator[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Yield ``(stream_index, x, y)`` minibatches in stream order."""
        generators = [torch.Generator().manual_seed(s.seed) for s in self.shifts]
        for start in range(0, self.length, batch_size):
            index = torch.arange(start, min(start + batch_size, self.length))
            x, y = self.split.x[self.order[index]], self.split.y[self.order[index]]
            for shift, g in zip(self.shifts, generators):
                x, y = apply_shift(shift, x, y, shift.schedule.strength(index), g, self.n_classes)
            yield index, x, y
