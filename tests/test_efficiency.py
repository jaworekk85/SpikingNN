import pytest
import torch
from torch import nn

from snn_research.datasets import Stream, SyntheticConfig, synthetic_prototypes
from snn_research.encoding import EncodingConfig
from snn_research.energy import EnergyProxyConfig, energy_proxy_pj
from snn_research.learning_rules import ThreeFactor, ThreeFactorConfig, constrain_
from snn_research.metrics import fan_out, synapse_count, synaptic_operations
from snn_research.models import ModelConfig, build_model
from snn_research.neurons import LIFParams
from snn_research.protocols import PrequentialConfig, run_prequential
from snn_research.shifts import Schedule, ShiftConfig


def brute_force_fan_out(conv: nn.Conv2d, shape: tuple[int, int, int]) -> torch.Tensor:
    """Count, for each input element, the outputs its one-hot input reaches."""
    probe = nn.Conv2d(conv.in_channels, conv.out_channels, conv.kernel_size, conv.stride, conv.padding, bias=False)
    nn.init.ones_(probe.weight)
    counts = torch.zeros(shape)
    for idx in range(counts.numel()):
        x = torch.zeros(counts.numel())
        x[idx] = 1
        counts.view(-1)[idx] = (probe(x.view(1, *shape)) != 0).sum()
    return counts


@pytest.mark.parametrize("hw, stride, padding", [(7, 2, 1), (8, 2, 1), (6, 1, 0), (5, 2, 0)])
def test_conv_fan_out_matches_brute_force(hw, stride, padding):
    conv = nn.Conv2d(2, 3, kernel_size=3, stride=stride, padding=padding)
    shape = (2, hw, hw)
    torch.testing.assert_close(fan_out(conv, shape), brute_force_fan_out(conv, shape))


def test_dense_counts_and_synapse_count():
    torch.manual_seed(0)
    model = build_model(ModelConfig(hidden=(5, 4)), (6,), 3, LIFParams())
    with torch.no_grad():
        _, obs = model(torch.zeros(2, 1, 6))
    assert torch.equal(fan_out(model.layers[0], (6,)), torch.full((6,), 5.0))
    assert synapse_count(model, obs) == 6 * 5 + 5 * 4 + 4 * 3  # dense: one synapse per weight


def test_synaptic_operations_count_spikes_times_fan_out_and_keep_macs_apart():
    torch.manual_seed(0)
    model = build_model(ModelConfig(hidden=(5,), init_gain=10.0), (6,), 3, LIFParams(threshold=0.1))
    inputs = torch.zeros(4, 2, 6)
    inputs[0, 0, :3] = 1  # sample 0: 3 input spikes
    with torch.no_grad():
        _, obs = model(inputs)
    hidden_spikes = obs["layer0"].spikes.sum((0, 2))
    ops = synaptic_operations(model, obs, input_is_spikes=True)
    torch.testing.assert_close(ops["synops"], torch.tensor([3 * 5.0, 0.0]) + hidden_spikes * 3)
    assert ops["input_macs"].sum() == 0
    as_current = synaptic_operations(model, obs, input_is_spikes=False)
    torch.testing.assert_close(as_current["input_macs"], torch.tensor([3 * 5.0, 0.0]))
    torch.testing.assert_close(as_current["synops"], hidden_spikes * 3)


def test_energy_proxy_uses_explicit_coefficients():
    config = EnergyProxyConfig(pj_per_synop=1.0, pj_per_mac=5.0, pj_per_neuron_update=0.5, pj_per_weight_update=2.0)
    parts = energy_proxy_pj(config, synops=10, macs=2, neuron_updates=4, weight_updates=3)
    assert parts == {"synops_pj": 10.0, "macs_pj": 10.0, "neuron_updates_pj": 2.0,
                     "weight_updates_pj": 6.0, "total_pj": 28.0}
    assumptions = EnergyProxyConfig().assumptions()
    assert assumptions["units"] == "pJ" and "proxy" in assumptions["kind"] and assumptions["source"]
    with pytest.raises(ValueError):
        EnergyProxyConfig(pj_per_synop=-1)
    with pytest.raises(ValueError):
        EnergyProxyConfig(source="")


def test_bounded_precision_rounds_to_fixed_levels():
    w = torch.linspace(-3, 3, 101)
    constrain_(w, bits=3, w_max=1.5)  # 3 bits: levels -1.5, -1, -0.5, 0, 0.5, 1, 1.5
    assert set(w.tolist()) <= {-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5}
    with pytest.raises(ValueError):
        ThreeFactorConfig(weight_bits=4)  # quantization needs a bound
    with pytest.raises(ValueError):
        ThreeFactorConfig(weight_bits=1, w_max=1.0)


def test_rule_keeps_weights_on_the_fixed_point_grid_and_prequential_reports_costs():
    train, _ = synthetic_prototypes(SyntheticConfig(n_train=32, n_test=8, n_classes=3, shape=(6,)))
    torch.manual_seed(0)
    model = build_model(ModelConfig(hidden=(8,), init_gain=5.0), (6,), 3, LIFParams())
    rule = ThreeFactor(ThreeFactorConfig(w_max=2.0, weight_bits=4))
    rule.init_state(model)
    shift = ShiftConfig(schedule=Schedule(start=8, end=24))
    stream = Stream(train, length=32, n_classes=3, seed=0, shifts=(shift,))
    result = run_prequential(model, [rule], stream, EncodingConfig(T=5), PrequentialConfig(length=32, window=8),
                             torch.Generator().manual_seed(0))
    step = 2.0 / (2**3 - 1)
    for w in (model.readout.weight, model.layers[0][1].weight):
        assert torch.allclose(w / step, (w / step).round(), atol=1e-4) and w.abs().max() <= 2.0
    m = result["metrics"]
    assert m["shift0/adaptation_updates"] == 2  # batches starting at 8 and 16
    assert m["shift0/adaptation_weights_written"] == 2 * (6 * 8 + 8 * 3)
    assert m["shift0/adaptation_synops"] > 0 and m["shift0/adaptation_input_macs"] == 0
