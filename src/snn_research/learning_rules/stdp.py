"""Pair-based STDP with exponential traces (unsupervised, synapse-local).

For presynaptic trace ``x`` and postsynaptic trace ``y`` (decay per step
``exp(-dt/tau)``, +1 per spike), at each step ``t``:

    dW += a_plus * z_post(t) x(t)  -  a_minus * exp(-dt/tau_minus) y(t-1) z_pre(t)

so a pre spike ``d >= 0`` steps before a post spike gives ``+a_plus exp(-d dt/tau_plus)``
and a post spike ``d > 0`` steps before a pre spike gives ``-a_minus exp(-d dt/tau_minus)``.
Updates are averaged over the batch; weights are clipped to ``[-w_max, w_max]``.
Traces start at zero for every sample.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from .base import LocalObservation, linear_synapses, local_observation
from .declaration import RuleDeclaration


@dataclass(frozen=True)
class STDPConfig:
    a_plus: float = 1e-3
    a_minus: float = 1.05e-3
    tau_plus_ms: float = 10.0
    tau_minus_ms: float = 10.0
    w_max: float = 1.0

    def __post_init__(self) -> None:
        if min(self.a_plus, self.a_minus) < 0 or min(self.tau_plus_ms, self.tau_minus_ms, self.w_max) <= 0:
            raise ValueError("STDP amplitudes must be non-negative and time constants/w_max positive")


def stdp_update(obs: LocalObservation, config: STDPConfig, dt_ms: float) -> torch.Tensor:
    """Batch-averaged weight change ``[n_post, n_pre]`` from local spikes only."""
    decay_plus = math.exp(-dt_ms / config.tau_plus_ms)
    decay_minus = math.exp(-dt_ms / config.tau_minus_ms)
    batch = obs.pre.shape[1]
    x = torch.zeros_like(obs.pre[0])
    y = torch.zeros_like(obs.post[0])
    dw = torch.zeros(obs.post.shape[2], obs.pre.shape[2])
    for pre, post in zip(obs.pre, obs.post):
        x = decay_plus * x + pre
        dw += config.a_plus * post.T @ x - config.a_minus * (decay_minus * y).T @ pre
        y = decay_minus * y + post
    return dw / batch


class STDP:
    declaration = RuleDeclaration(
        name="stdp",
        uses_autograd=False,
        locality="synapse_local",
        online_capable=True,
        state=("pre_trace", "post_trace"),
        modulator_source="none",
    )

    def __init__(self, config: STDPConfig) -> None:
        self.config = config
        self.synapses: dict = {}
        self.dt_ms = 1.0

    def init_state(self, model) -> None:
        self.synapses = linear_synapses(model)
        self.dt_ms = model.lif.params.dt_ms

    @torch.no_grad()
    def step(self, model, layer_states, modulators) -> dict[str, torch.Tensor]:
        return {name: stdp_update(local_observation(layer_states[name]), self.config, self.dt_ms)
                for name in self.synapses}

    @torch.no_grad()
    def apply(self, updates: dict[str, torch.Tensor]) -> None:
        for name, dw in updates.items():
            weight = self.synapses[name].weight
            weight.add_(dw).clamp_(-self.config.w_max, self.config.w_max)

    def reset(self) -> None:
        """Traces are per-sample and already start at zero; nothing persists."""
