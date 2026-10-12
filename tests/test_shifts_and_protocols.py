import pytest
import torch

from snn_research.datasets import Split, Stream, SyntheticConfig, synthetic_prototypes
from snn_research.encoding import EncodingConfig
from snn_research.learning_rules import ThreeFactor, ThreeFactorConfig
from snn_research.metrics import accuracy, forgetting, phase_quality, recovery_samples
from snn_research.models import ModelConfig, build_model
from snn_research.neurons import LIFParams
from snn_research.protocols import OfflineConfig, PrequentialConfig, run_prequential, run_task_sequence
from snn_research.shifts import Schedule, ShiftConfig, apply_shift, delay_inputs

INDEX = torch.tensor([0, 9, 10, 12, 15, 17, 20, 30])


@pytest.mark.parametrize(
    "schedule, expected",
    [
        (Schedule("abrupt", start=10, end=20), [0, 0, 1, 1, 1, 1, 1, 1]),
        (Schedule("gradual", start=10, end=20), [0, 0, 0, 0.2, 0.5, 0.7, 1, 1]),
        (Schedule("incremental", start=10, end=20, steps=2), [0, 0, 0.5, 0.5, 1, 1, 1, 1]),
        (Schedule("recurring", start=10, end=20, period=3), [0, 0, 1, 1, 0, 1, 0, 0]),
    ],
)
def test_schedule_strengths(schedule, expected):
    torch.testing.assert_close(schedule.strength(INDEX), torch.tensor(expected, dtype=torch.float))


def test_schedule_phases_and_validation():
    s = Schedule(start=10, end=20)
    assert [s.phase_at(i) for i in (9, 10, 19, 20)] == ["before", "during", "during", "after"]
    with pytest.raises(ValueError):
        Schedule(start=5, end=5)


def test_shift_kinds_and_zero_strength():
    x, y = torch.full((6, 4), 0.5), torch.tensor([0, 1, 2, 0, 1, 2])
    g = torch.Generator().manual_seed(0)
    for kind in ("noise", "intensity", "label_permutation"):
        xs, ys = apply_shift(ShiftConfig(kind, magnitude=0.5), x, y, torch.zeros(6), g, n_classes=3)
        assert torch.equal(xs, x) and torch.equal(ys, y)
    xs, _ = apply_shift(ShiftConfig("intensity", magnitude=0.5), x, y, torch.ones(6), g, 3)
    assert torch.allclose(xs, torch.full_like(x, 0.25))
    _, ys = apply_shift(ShiftConfig("label_permutation", magnitude=1.0), x, y, torch.ones(6), g, 3)
    assert ys.tolist() == [1, 2, 0, 1, 2, 0]
    a = apply_shift(ShiftConfig("noise"), x, y, torch.ones(6), torch.Generator().manual_seed(3), 3)[0]
    b = apply_shift(ShiftConfig("noise"), x, y, torch.ones(6), torch.Generator().manual_seed(3), 3)[0]
    assert torch.equal(a, b) and not torch.equal(a, x)


def test_latency_jitter_delays_each_channel_within_bound():
    T, k = 12, 3
    spikes = torch.zeros(T, 2, 50)
    spikes[0] = 1  # every channel spikes once, at t = 0
    delayed = delay_inputs(spikes, torch.tensor([0.0, k]), torch.Generator().manual_seed(0))
    assert torch.equal(delayed[:, 0], spikes[:, 0])  # zero delay leaves the sample unchanged
    t_spike = delayed[:, 1].argmax(0)
    assert delayed[:, 1].sum(0).eq(1).all() and t_spike.max() == k and t_spike.min() == 0
    again = delay_inputs(spikes, torch.tensor([0.0, k]), torch.Generator().manual_seed(0))
    assert torch.equal(delayed, again)
    with pytest.raises(ValueError):  # temporal kinds act on encoded input only
        apply_shift(ShiftConfig("latency_jitter"), spikes[0], torch.zeros(2), torch.ones(2), torch.Generator(), 3)


def tiny_data():
    return synthetic_prototypes(SyntheticConfig(n_train=40, n_test=20, n_classes=3, shape=(6,)))


DIRECT = EncodingConfig(kind="direct", T=4)  # deterministic encoding for exact comparisons


def test_stream_order_shifts_and_role_guard():
    train, test = tiny_data()
    shift = ShiftConfig("intensity", magnitude=0.0, schedule=Schedule("abrupt", start=50, end=60))
    stream = Stream(train, length=100, n_classes=3, seed=1, shifts=(shift,))
    batches = list(stream.batches(25, DIRECT, torch.Generator()))
    index = torch.cat([b[0] for b in batches])
    assert torch.equal(index, torch.arange(100))
    assert torch.equal(stream.order, Stream(train, 100, 3, seed=1).order)
    inputs = torch.cat([b[1] for b in batches], dim=1)
    assert inputs[:, :50].sum() > 0 and inputs[:, 50:].sum() == 0  # intensity 0 from sample 50 on
    with pytest.raises(ValueError):
        Stream(test, length=10, n_classes=3, seed=0)


def test_temporal_shift_is_applied_after_encoding_from_its_start():
    train, _ = tiny_data()
    shift = ShiftConfig("latency_jitter", magnitude=3, schedule=Schedule("abrupt", start=20, end=30))
    clean = torch.cat([b[1] for b in Stream(train, 40, 3, seed=0).batches(10, DIRECT, torch.Generator())], 1)
    jittered = torch.cat([b[1] for b in Stream(train, 40, 3, seed=0, shifts=(shift,))
                          .batches(10, DIRECT, torch.Generator())], 1)
    assert torch.equal(clean[:, :20], jittered[:, :20])
    assert not torch.equal(clean[:, 20:], jittered[:, 20:])
    assert jittered[0, 20:].sum() < clean[0, 20:].sum()  # delayed channels are zero at t = 0


def test_adaptation_metrics():
    s = Schedule(start=4, end=8)
    correct = [1, 1, 1, 1, 0, 0, 1, 1, 1, 1]
    assert phase_quality(correct, s) == {"before": 1.0, "during": 0.5, "after": 1.0}
    assert recovery_samples(correct, s, q_target=1.0, window=2) == 4  # window [6, 7]
    assert recovery_samples(correct[:6], s, q_target=1.0, window=2) is None
    assert forgetting([0.5, 0.9, 0.6]) == pytest.approx(0.3)


class SpyModel(torch.nn.Module):
    """Wraps a model and logs every forward pass with the logits it returned."""

    def __init__(self, model, log):
        super().__init__()
        self.inner, self.log = model, log
        self.layers, self.readout, self.lif = model.layers, model.readout, model.lif

    def forward(self, x):
        logits, obs = self.inner(x)
        self.log.append(("forward", logits.detach().clone()))
        return logits, obs


class SpyRule(ThreeFactor):
    def __init__(self, log):
        super().__init__(ThreeFactorConfig())
        self.log = log

    def step(self, model, layer_states, modulators):
        self.log.append(("step", None))
        return super().step(model, layer_states, modulators)


def make_model():
    torch.manual_seed(0)
    return build_model(ModelConfig(hidden=(8,), init_gain=5.0), (6,), 3, LIFParams())


def test_prequential_predicts_before_training_on_each_batch():
    train, _ = tiny_data()
    log: list = []
    model, rule = SpyModel(make_model(), log), SpyRule(log)
    rule.init_state(model)
    stream = Stream(train, length=24, n_classes=3, seed=0)
    config = PrequentialConfig(length=24, batch_size=8, window=8)
    result = run_prequential(model, [rule], stream, EncodingConfig(T=5), config, torch.Generator())
    assert [event for event, _ in log] == ["forward", "step"] * 3
    labels = [y for _, _, y in stream.batches(8, EncodingConfig(T=5), torch.Generator())]
    for k, (_, logits) in enumerate(log[::2]):  # recorded quality uses pre-update logits
        assert sum(result["traces"]["correct"][8 * k : 8 * k + 8]) / 8 == accuracy(logits, labels[k])


def test_prequential_without_adaptation_never_updates():
    train, _ = tiny_data()
    model, rule = make_model(), ThreeFactor(ThreeFactorConfig())
    rule.init_state(model)
    before = [p.clone() for p in model.parameters()]
    shift = ShiftConfig(schedule=Schedule(start=8, end=16))
    stream = Stream(train, length=24, n_classes=3, seed=0, shifts=(shift,))
    config = PrequentialConfig(length=24, batch_size=8, adapt=False, window=8)
    result = run_prequential(model, [rule], stream, EncodingConfig(T=5), config, torch.Generator())
    assert all(torch.equal(b, p) for b, p in zip(before, model.parameters()))
    assert result["metrics"]["shift0/adaptation_updates"] == 0
    assert set(result["metrics"]) >= {f"shift0/quality_{p}" for p in ("before", "during", "after")}


def test_task_sequence_reports_matrix_and_forgetting():
    train, test = tiny_data()
    tasks = [(train.subset((0, 1)), test.subset((0, 1))), (train.subset((2,)), test.subset((2,)))]
    model, rule = make_model(), ThreeFactor(ThreeFactorConfig())
    rule.init_state(model)
    result = run_task_sequence(model, [rule], tasks, EncodingConfig(T=5), OfflineConfig(epochs=2, batch_size=8),
                               torch.Generator().manual_seed(0))
    matrix = result["accuracy_matrix"]
    assert len(matrix) == 2 and all(len(row) == 2 for row in matrix)
    assert result["metrics"]["task0/forgetting"] == pytest.approx(max(matrix[0][0], matrix[1][0]) - matrix[1][0])
    assert result["metrics"]["task1/forgetting"] == 0.0
    with pytest.raises(ValueError):
        run_task_sequence(model, [rule], [(train, train)], EncodingConfig(T=5), OfflineConfig(), torch.Generator())


def test_subset_keeps_role_and_classes():
    train, _ = tiny_data()
    sub = train.subset((1,))
    assert sub.role == "train" and set(sub.y.tolist()) == {1}
    with pytest.raises(ValueError):
        Split(sub.x, sub.y, role="validation", version="v")
