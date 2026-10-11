"""Reference leaky integrate-and-fire (LIF) neuron in plain PyTorch.

Discrete dynamics per timestep (``dt_ms``), for input current ``I_t``:

    u_t = beta * u_{t-1} + (1 - beta) * I_t,    beta = exp(-dt_ms / tau_mem_ms)
    z_t = H(u_t - threshold)

This exponential-Euler scheme is exact at sample points for piecewise-constant input:
with constant ``I`` and ``u_0 = 0`` it gives ``u_t = I * (1 - beta**t)``, the analytic
solution of ``tau du/dt = -u + I``. After a spike, ``u`` is reduced by the threshold
(``reset="subtract"``) or set to zero (``reset="zero"``). During a refractory period
the neuron holds ``u = 0``, ignores input and cannot spike.

The Heaviside step uses a fast-sigmoid surrogate derivative in the backward pass.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class LIFParams:
    tau_mem_ms: float = 10.0
    dt_ms: float = 1.0
    threshold: float = 1.0
    reset: str = "subtract"
    refractory_steps: int = 0
    surrogate_slope: float = 10.0

    def __post_init__(self) -> None:
        if self.tau_mem_ms <= 0 or self.dt_ms <= 0:
            raise ValueError("tau_mem_ms and dt_ms must be positive")
        if self.threshold <= 0:
            raise ValueError("threshold must be positive")
        if self.reset not in ("subtract", "zero"):
            raise ValueError(f"reset must be 'subtract' or 'zero', got {self.reset!r}")
        if self.refractory_steps < 0:
            raise ValueError("refractory_steps must be non-negative")
        if self.surrogate_slope <= 0:
            raise ValueError("surrogate_slope must be positive")

    @property
    def beta(self) -> float:
        return math.exp(-self.dt_ms / self.tau_mem_ms)


@dataclass
class LIFState:
    v: torch.Tensor
    refractory: torch.Tensor  # remaining refractory steps, integer tensor


class _SpikeFn(torch.autograd.Function):
    """Heaviside forward; fast-sigmoid derivative ``1 / (1 + k|x|)^2`` backward."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, slope: float) -> torch.Tensor:
        ctx.save_for_backward(x)
        ctx.slope = slope
        return (x >= 0).to(x.dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (x,) = ctx.saved_tensors
        return grad_output / (1 + ctx.slope * x.abs()) ** 2, None


class LIF:
    """Stateless LIF dynamics; all state lives in an explicit ``LIFState``."""

    def __init__(self, params: LIFParams) -> None:
        self.params = params

    def init_state(self, like: torch.Tensor) -> LIFState:
        return LIFState(
            v=torch.zeros_like(like),
            refractory=torch.zeros_like(like, dtype=torch.long),
        )

    def step(self, input_current: torch.Tensor, state: LIFState) -> tuple[torch.Tensor, LIFState]:
        p = self.params
        active = state.refractory == 0
        v = torch.where(active, p.beta * state.v + (1 - p.beta) * input_current, torch.zeros_like(state.v))
        spikes = _SpikeFn.apply(v - p.threshold, p.surrogate_slope) * active
        if p.reset == "subtract":
            v = v - spikes * p.threshold
        else:
            v = v * (1 - spikes)
        refractory = torch.where(
            spikes.detach() > 0,
            torch.full_like(state.refractory, p.refractory_steps),
            (state.refractory - 1).clamp(min=0),
        )
        return spikes, LIFState(v=v, refractory=refractory)
