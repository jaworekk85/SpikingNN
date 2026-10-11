import dataclasses
import inspect
import math

import pytest
import torch

from snn_research.learning_rules import (
    STDP,
    LocalObservation,
    RuleDeclaration,
    STDPConfig,
    ThreeFactor,
    ThreeFactorConfig,
    eligibility_traces,
    stdp_update,
    three_factor_update,
)
from snn_research.models import ModelConfig, build_model
from snn_research.neurons import LIFParams

STDP_CFG = STDPConfig(a_plus=1.0, a_minus=0.5, tau_plus_ms=4.0, tau_minus_ms=8.0)
TF_CFG = ThreeFactorConfig(lr=1.0, tau_pre_ms=5.0, tau_eligibility_ms=10.0)


def spike_pair(t_pre: int, t_post: int, T: int = 12) -> LocalObservation:
    pre, post = torch.zeros(T, 1, 1), torch.zeros(T, 1, 1)
    pre[t_pre], post[t_post] = 1, 1
    return LocalObservation(pre=pre, post=post, post_potential=torch.zeros(T, 1, 1))


@pytest.mark.parametrize("d", [-3, -1, 0, 1, 3])
def test_stdp_window_sign_and_shape(d):
    dw = stdp_update(spike_pair(t_pre=5, t_post=5 + d), STDP_CFG, dt_ms=1.0).item()
    if d >= 0:  # pre before (or with) post: potentiation
        assert dw == pytest.approx(STDP_CFG.a_plus * math.exp(-d / STDP_CFG.tau_plus_ms))
    else:  # post before pre: depression
        assert dw == pytest.approx(-STDP_CFG.a_minus * math.exp(d / STDP_CFG.tau_minus_ms))


def test_eligibility_trace_decays_exponentially():
    T = 6
    pre, u = torch.zeros(T, 1, 1), torch.full((T, 1, 1), 1e9)  # psi ~ 0 after t=0
    pre[0], u[0] = 1, 1.0  # one coincidence at t=0 with psi = 1 (u at threshold)
    obs = LocalObservation(pre=pre, post=torch.zeros(T, 1, 1), post_potential=u)
    traces = [e.item() for e in eligibility_traces(obs, TF_CFG, dt_ms=1.0, threshold=1.0)]
    lam = math.exp(-1.0 / TF_CFG.tau_eligibility_ms)
    assert traces == pytest.approx([lam**t for t in range(T)], rel=1e-5)


def random_obs(batch: int, seed: int = 0, T: int = 8, n_pre: int = 5, n_post: int = 3) -> LocalObservation:
    g = torch.Generator().manual_seed(seed)
    return LocalObservation(
        pre=(torch.rand(T, batch, n_pre, generator=g) < 0.3).float(),
        post=(torch.rand(T, batch, n_post, generator=g) < 0.3).float(),
        post_potential=torch.rand(T, batch, n_post, generator=g) * 2,
    )


def test_three_factor_update_is_zero_without_modulator():
    obs = random_obs(batch=4)
    dw = three_factor_update(obs, torch.zeros(4, 3), TF_CFG, dt_ms=1.0, threshold=1.0)
    assert torch.count_nonzero(dw) == 0


def sample(obs: LocalObservation, b: int) -> LocalObservation:
    return LocalObservation(*(getattr(obs, f.name)[:, b : b + 1] for f in dataclasses.fields(obs)))


def test_minibatch_update_equals_mean_of_per_sample_updates():
    obs, modulator = random_obs(batch=3), torch.randn(3, 3)
    batch_stdp = stdp_update(obs, STDP_CFG, 1.0)
    batch_tf = three_factor_update(obs, modulator, TF_CFG, 1.0, 1.0)
    per_stdp = torch.stack([stdp_update(sample(obs, b), STDP_CFG, 1.0) for b in range(3)]).mean(0)
    per_tf = torch.stack([
        three_factor_update(sample(obs, b), modulator[b : b + 1], TF_CFG, 1.0, 1.0) for b in range(3)
    ]).mean(0)
    torch.testing.assert_close(batch_stdp, per_stdp)
    torch.testing.assert_close(batch_tf, per_tf)


@pytest.mark.parametrize("rule", [STDP(STDPConfig(a_plus=0.05, a_minus=0.05)), ThreeFactor(ThreeFactorConfig())])
def test_local_rules_build_no_graph_and_populate_no_grad(rule):
    torch.manual_seed(0)
    model = build_model(ModelConfig(hidden=(6,), init_gain=5.0), (4,), 3, LIFParams())
    before = [p.detach().clone() for p in model.parameters()]
    rule.init_state(model)
    with torch.enable_grad():  # the forward builds a graph; the rule must not use it
        logits, observations = model(torch.ones(10, 2, 4))
        error = torch.eye(3)[:2] - logits.softmax(-1).detach()
        updates = rule.step(model, observations, {"error": error})
        rule.apply(updates)
    assert all(not u.requires_grad and u.grad_fn is None for u in updates.values())
    assert all(p.grad is None for p in model.parameters())
    assert any(not torch.equal(b, p) for b, p in zip(before, model.parameters()))


def test_local_kernels_receive_only_the_restricted_observation():
    assert {f.name for f in dataclasses.fields(LocalObservation)} == {"pre", "post", "post_potential"}
    for kernel in (stdp_update, three_factor_update, eligibility_traces):
        first = next(iter(inspect.signature(kernel).parameters.values()))
        assert first.annotation in (LocalObservation, "LocalObservation")
        assert "model" not in inspect.signature(kernel).parameters


def test_declarations_of_local_rules():
    assert STDP.declaration.locality == "synapse_local" and not STDP.declaration.uses_autograd
    assert STDP.declaration.modulator_source == "none"
    assert ThreeFactor.declaration.locality == "global_modulated" and not ThreeFactor.declaration.uses_autograd
    assert ThreeFactor.declaration.modulator_source == "error"
    with pytest.raises(ValueError):
        RuleDeclaration("bad", uses_autograd=True, locality="synapse_local", online_capable=True,
                        state=(), modulator_source="none")
