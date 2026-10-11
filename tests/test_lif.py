import pytest
import torch

from snn_research.neurons import LIF, LIFParams


def run(lif: LIF, current: torch.Tensor, steps: int):
    state = lif.init_state(current)
    spikes, vs = [], []
    for _ in range(steps):
        z, state = lif.step(current, state)
        spikes.append(z)
        vs.append(state.v)
    return torch.stack(spikes), torch.stack(vs)


def test_constant_input_matches_analytic_solution():
    params = LIFParams(tau_mem_ms=5.0, dt_ms=0.5, threshold=100.0)
    current = torch.tensor([0.3, 1.0, 2.5])
    spikes, vs = run(LIF(params), current, 40)
    t_ms = torch.arange(1, 41).unsqueeze(1) * params.dt_ms
    analytic = current * (1 - torch.exp(-t_ms / params.tau_mem_ms))
    assert spikes.sum() == 0
    torch.testing.assert_close(vs, analytic)


def test_threshold_crossing():
    lif = LIF(LIFParams(threshold=1.0))
    below, _ = run(lif, torch.tensor([0.99]), 200)  # u -> 0.99 asymptotically
    above, _ = run(lif, torch.tensor([1.2]), 200)
    assert below.sum() == 0
    assert above.sum() > 0


@pytest.mark.parametrize("reset, expected", [("subtract", 0.5), ("zero", 0.0)])
def test_reset_rules(reset, expected):
    lif = LIF(LIFParams(threshold=1.0, reset=reset))
    state = lif.init_state(torch.zeros(1))
    state.v = torch.tensor([1.5])
    z, state = lif.step(torch.tensor([1.5]), state)  # u = 1.5 -> spike
    assert z.item() == 1.0
    assert state.v.item() == pytest.approx(expected)


def test_refractory_period_blocks_integration_and_spikes():
    lif = LIF(LIFParams(tau_mem_ms=1.0, threshold=0.5, reset="zero", refractory_steps=3))
    spikes, vs = run(lif, torch.tensor([10.0]), 9)
    assert spikes.squeeze().tolist() == [1, 0, 0, 0, 1, 0, 0, 0, 1]
    assert vs[1:4].abs().sum() == 0


def test_surrogate_gradient_flows_through_spikes():
    lif = LIF(LIFParams())
    current = torch.full((4,), 0.8, requires_grad=True)
    spikes, _ = run(lif, current, 10)
    spikes.sum().backward()
    assert current.grad is not None and current.grad.abs().sum() > 0


@pytest.mark.parametrize("bad", [{"tau_mem_ms": 0}, {"reset": "hard"}, {"refractory_steps": -1}])
def test_invalid_params_rejected(bad):
    with pytest.raises(ValueError):
        LIFParams(**bad)
