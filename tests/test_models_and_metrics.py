import pytest
import torch

from snn_research.metrics import activity_summary, spike_rate, temporal_sparsity
from snn_research.models import ModelConfig, build_model, count_neurons
from snn_research.neurons import LIFParams


@pytest.mark.parametrize(
    "config, input_shape",
    [(ModelConfig(kind="mlp", hidden=(12, 6)), (10,)), (ModelConfig(kind="conv", hidden=(6,), conv_channels=4), (1, 7, 7))],
)
def test_forward_shapes_and_observations(config, input_shape):
    torch.manual_seed(0)
    model = build_model(config, input_shape, n_classes=3, lif=LIFParams())
    x = torch.rand(5, 2, *input_shape)
    logits, obs = model(x)
    assert logits.shape == (2, 3)
    assert all(o.spikes.shape[:2] == (5, 2) and o.spikes.shape == o.v.shape for o in obs.values())
    expected = 4 * 4 * 4 + 6 if config.kind == "conv" else 12 + 6
    assert count_neurons(obs) == expected


def test_forward_has_no_state_carried_between_calls():
    torch.manual_seed(0)
    model = build_model(ModelConfig(), (8,), 2, LIFParams())
    x = torch.rand(10, 3, 8)
    first, _ = model(x)
    second, _ = model(x)
    torch.testing.assert_close(first, second)


def test_activity_metrics_on_hand_computed_tensor():
    a = torch.zeros(2, 1, 4)  # T=2, batch=1, 4 neurons -> 8 slots
    a[0, 0, :2] = 1
    b = torch.zeros(2, 1, 2)  # 4 slots
    b[1, 0, 0] = 1
    assert spike_rate(a) == 0.25 and temporal_sparsity(a) == 0.75
    summary = activity_summary({"a": a, "b": b})
    assert summary["overall/spike_rate"] == pytest.approx(3 / 12)
    assert summary["overall/temporal_sparsity"] == pytest.approx(9 / 12)
