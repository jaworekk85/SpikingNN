"""Optional bounded-precision (fixed-point) weights for local rules.

With ``bits`` > 0, weights are clipped to ``[-w_max, w_max]`` and rounded to the
nearest of ``2**bits - 1`` symmetric levels (step ``w_max / (2**(bits-1) - 1)``), as a
signed fixed-point synapse would store them. Rounding is deterministic
round-to-nearest, so updates smaller than half a step are lost.
"""

from __future__ import annotations

import torch


def check_precision(bits: int, w_max: float) -> None:
    if bits < 0 or bits == 1:
        raise ValueError("weight_bits must be 0 (full precision) or at least 2")
    if bits and w_max <= 0:
        raise ValueError("bounded-precision weights need a positive w_max")


@torch.no_grad()
def constrain_(weight: torch.Tensor, bits: int, w_max: float) -> None:
    """Clip (if ``w_max`` > 0) and quantize (if ``bits`` > 0) ``weight`` in place."""
    if w_max > 0:
        weight.clamp_(-w_max, w_max)
    if bits:
        step = w_max / (2 ** (bits - 1) - 1)
        weight.div_(step).round_().mul_(step)
