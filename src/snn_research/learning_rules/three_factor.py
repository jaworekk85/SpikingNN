"""Error-modulated three-factor rule with eligibility traces.

Per synapse ``ij`` (pre ``i``, post ``j``) and step ``t``:

    x_i(t)  = beta_pre * x_i(t-1) + pre_i(t)                    presynaptic trace
    psi_j(t) = 1 / (1 + k |u_j(t) - threshold|)^2                postsynaptic pseudo-derivative
    e_ij(t) = lambda_e * e_ij(t-1) + psi_j(t) x_i(t)            eligibility trace
    dW_ij   = lr * M_j * mean_t e_ij(t)                          third factor M at sample end

The readout has its own ``readout_lr`` because its eligibility (an input rate) is on
a different scale from hidden-layer eligibility traces.

The modulator is the output error ``onehot(label) - softmax(logits)``. The readout
(non-spiking, on time-averaged spike rates) uses it directly with eligibility equal to
its input rate, i.e. the delta rule. Each hidden layer receives the error broadcast
through a fixed random feedback matrix (no transported weights or backpropagated
gradients). Updates are averaged over the batch; biases are not updated. Optional
``w_max`` clipping and ``weight_bits`` fixed-point precision apply to all bound weights.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterator

import torch

from .base import LocalObservation, linear_synapses, local_observation
from .declaration import RuleDeclaration
from .precision import check_precision, constrain_


@dataclass(frozen=True)
class ThreeFactorConfig:
    lr: float = 1.0  # hidden layers
    readout_lr: float = 5.0
    tau_pre_ms: float = 10.0
    tau_eligibility_ms: float = 20.0
    pseudo_slope: float = 5.0
    train_hidden: bool = True
    w_max: float = 0.0  # 0 = unbounded
    weight_bits: int = 0  # 0 = full precision; needs w_max > 0

    def __post_init__(self) -> None:
        if min(self.lr, self.readout_lr, self.tau_pre_ms, self.tau_eligibility_ms, self.pseudo_slope) <= 0:
            raise ValueError("three-factor parameters must be positive")
        if self.w_max < 0:
            raise ValueError("w_max must be non-negative")
        check_precision(self.weight_bits, self.w_max)


def eligibility_traces(obs: LocalObservation, config: ThreeFactorConfig, dt_ms: float,
                       threshold: float) -> Iterator[torch.Tensor]:
    """Yield the eligibility trace ``[batch, n_post, n_pre]`` at every timestep."""
    beta_pre = math.exp(-dt_ms / config.tau_pre_ms)
    lambda_e = math.exp(-dt_ms / config.tau_eligibility_ms)
    x = torch.zeros_like(obs.pre[0])
    e = torch.zeros(obs.pre.shape[1], obs.post.shape[2], obs.pre.shape[2])
    for pre, u in zip(obs.pre, obs.post_potential):
        x = beta_pre * x + pre
        psi = 1 / (1 + config.pseudo_slope * (u - threshold).abs()) ** 2
        e = lambda_e * e + psi.unsqueeze(2) * x.unsqueeze(1)
        yield e


def three_factor_update(obs: LocalObservation, modulator: torch.Tensor, config: ThreeFactorConfig,
                        dt_ms: float, threshold: float) -> torch.Tensor:
    """``lr * mean_batch(M_j * mean_t e_ij)`` for a ``[batch, n_post]`` modulator."""
    mean_e = sum(eligibility_traces(obs, config, dt_ms, threshold)) / obs.pre.shape[0]
    return config.lr * torch.einsum("bj,bji->ji", modulator, mean_e) / modulator.shape[0]


class ThreeFactor:
    declaration = RuleDeclaration(
        name="three_factor",
        uses_autograd=False,
        locality="global_modulated",
        online_capable=True,
        state=("pre_trace", "eligibility_trace", "feedback_weights"),
        modulator_source="error",
    )

    def __init__(self, config: ThreeFactorConfig) -> None:
        self.config = config
        self.synapses: dict = {}
        self.feedback: dict[str, torch.Tensor] = {}

    def init_state(self, model) -> None:
        self.synapses = linear_synapses(model) if self.config.train_hidden else {}
        self.readout = model.readout
        self.dt_ms = model.lif.params.dt_ms
        self.threshold = model.lif.params.threshold
        n_classes = model.readout.out_features
        self.feedback = {  # fixed random broadcast weights, drawn from the seeded global RNG
            name: torch.randn(syn.out_features, n_classes) / math.sqrt(n_classes)
            for name, syn in self.synapses.items()
        }
        for synapse in [self.readout, *self.synapses.values()]:
            constrain_(synapse.weight, self.config.weight_bits, self.config.w_max)

    @torch.no_grad()
    def step(self, model, layer_states, modulators: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        error = modulators["error"]  # [batch, n_classes]
        last = layer_states[f"layer{len(layer_states) - 1}"]
        rate = last.spikes.flatten(2).mean(0).detach()  # readout input
        updates = {"readout": self.config.readout_lr * error.T @ rate / error.shape[0]}
        for name in self.synapses:
            modulator = error @ self.feedback[name].T
            updates[name] = three_factor_update(local_observation(layer_states[name]), modulator,
                                                self.config, self.dt_ms, self.threshold)
        return updates

    @torch.no_grad()
    def apply(self, updates: dict[str, torch.Tensor]) -> None:
        for name, dw in updates.items():
            synapse = self.readout if name == "readout" else self.synapses[name]
            synapse.weight.add_(dw)
            constrain_(synapse.weight, self.config.weight_bits, self.config.w_max)

    def reset(self) -> None:
        """Traces are per-sample; feedback weights are fixed. Nothing to clear."""
