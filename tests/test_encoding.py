import torch

from snn_research.encoding import EncodingConfig, direct_encode, encode, latency_encode, rate_encode


def test_rate_encoding_is_seeded_and_matches_intensity():
    x = torch.tensor([[0.0, 0.25, 1.0]])
    a = rate_encode(x, 4000, torch.Generator().manual_seed(1))
    b = rate_encode(x, 4000, torch.Generator().manual_seed(1))
    assert a.values.shape == (4000, 1, 3)
    assert torch.equal(a.values, b.values)
    torch.testing.assert_close(a.values.mean(0), x, atol=0.03, rtol=0)
    assert a.is_spikes and abs(a.event_rate - x.mean().item()) < 0.03


def test_latency_encoding_single_spike_at_expected_time():
    x = torch.tensor([[1.0, 0.5, 0.0]])
    enc = latency_encode(x, T=5)
    assert enc.values[:, 0, 0].tolist() == [1, 0, 0, 0, 0]
    assert enc.values[:, 0, 1].tolist() == [0, 0, 1, 0, 0]
    assert enc.values[:, 0, 2].sum() == 0


def test_direct_encoding_is_current_not_events():
    x = torch.rand(2, 3)
    enc = direct_encode(x, T=7)
    assert enc.values.shape == (7, 2, 3)
    assert torch.equal(enc.values[3], x)
    assert not enc.is_spikes and enc.event_rate is None


def test_encode_dispatches_on_config():
    x = torch.rand(2, 3)
    enc = encode(EncodingConfig(kind="latency", T=6), x, torch.Generator())
    assert enc.kind == "latency" and enc.T == 6
