"""Reference experiment wiring: configuration, data, rules and one seed's run.

Per seed: offline pretraining on clean data -> test accuracy (clean and with the shift
at full strength) -> prequential stream with one scheduled shift, adapting online ->
clean/shifted accuracy again (forgetting) -> operation counts and an energy *proxy*
for inference and, separately, for adaptation in the shift window -> one RunRecord.

Rules: ``surrogate_bptt`` (global gradient), ``three_factor`` (error-modulated, all
layers) or ``stdp`` (STDP on hidden layers + three-factor delta rule on the readout,
since unsupervised STDP alone cannot learn labels). Pipeline validation only.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from typing import Any

import torch

from snn_research.datasets import (
    MNISTConfig,
    NMNISTConfig,
    Split,
    Stream,
    SyntheticConfig,
    load_mnist,
    load_nmnist,
    synthetic_prototypes,
)
from snn_research.encoding import EncodingConfig
from snn_research.energy import EnergyProxyConfig, energy_proxy_pj
from snn_research.learning_rules import STDP, STDPConfig, SurrogateBPTT, SurrogateBPTTConfig, ThreeFactor, ThreeFactorConfig
from snn_research.metrics import forgetting, synapse_count
from snn_research.models import ModelConfig, SpikingNet, build_model, count_neurons
from snn_research.neurons import LIFParams
from snn_research.protocols import OfflineConfig, PrequentialConfig, evaluate, run_offline, run_prequential
from snn_research.records import RunRecord
from snn_research.shifts import ShiftConfig
from snn_research.utils import provenance, resolved, seed_everything

RULES = ("surrogate_bptt", "three_factor", "stdp")


@dataclass(frozen=True)
class TuningConfig:
    """Hyperparameter search on a ``tune`` split carved out of the training data."""

    tune_fraction: float = 0.15
    split_seed: int = 0
    seeds: tuple[int, ...] = (100, 101)  # distinct from evaluation seeds
    grids: dict[str, dict[str, list[Any]]] = field(default_factory=dict)  # rule -> {dotted.param: values}

    def __post_init__(self) -> None:
        if not 0 < self.tune_fraction < 1 or not self.seeds:
            raise ValueError("tune_fraction must be in (0, 1) and at least one tuning seed is needed")
        unknown = set(self.grids) - set(RULES)
        if unknown:
            raise ValueError(f"tuning grids for unknown rules: {sorted(unknown)}")


@dataclass(frozen=True)
class ReferenceConfig:
    experiment: str = "reference"
    seeds: tuple[int, ...] = (0,)
    device: str = "cpu"
    dataset: str = "synthetic"  # synthetic | mnist (variant may be fashion_mnist) | nmnist
    data: SyntheticConfig = field(default_factory=SyntheticConfig)
    mnist: MNISTConfig = field(default_factory=MNISTConfig)
    nmnist: NMNISTConfig = field(default_factory=NMNISTConfig)
    encoding: EncodingConfig = field(default_factory=EncodingConfig)
    neuron: LIFParams = field(default_factory=LIFParams)
    model: ModelConfig = field(default_factory=ModelConfig)
    rule: str = "surrogate_bptt"
    surrogate_bptt: SurrogateBPTTConfig = field(default_factory=SurrogateBPTTConfig)
    three_factor: ThreeFactorConfig = field(default_factory=ThreeFactorConfig)
    stdp: STDPConfig = field(default_factory=STDPConfig)
    pretrain: OfflineConfig = field(default_factory=OfflineConfig)
    stream: PrequentialConfig = field(default_factory=PrequentialConfig)
    shift: ShiftConfig = field(default_factory=ShiftConfig)
    energy: EnergyProxyConfig = field(default_factory=EnergyProxyConfig)
    tuning: TuningConfig = field(default_factory=TuningConfig)

    def __post_init__(self) -> None:
        if self.dataset not in ("synthetic", "mnist", "nmnist"):
            raise ValueError("dataset must be 'synthetic', 'mnist' or 'nmnist'")
        if self.dataset == "nmnist":
            if self.encoding.kind != "events" or self.encoding.T != self.nmnist.T:
                raise ValueError("N-MNIST needs encoding kind 'events' with T equal to nmnist.T")
            if abs(self.neuron.dt_ms - self.nmnist.bin_ms) > 1e-9:
                raise ValueError(f"neuron dt_ms must equal the event bin width ({self.nmnist.bin_ms} ms)")
        elif self.encoding.kind == "events":
            raise ValueError("encoding 'events' requires dataset 'nmnist'")
        if self.rule not in RULES:
            raise ValueError(f"rule must be one of {RULES}, got {self.rule!r}")
        if not self.seeds:
            raise ValueError("at least one seed is required")
        if self.device != "cpu":
            raise ValueError("only device 'cpu' is supported")
        if self.shift.schedule.end > self.stream.length:
            raise ValueError("the shift schedule must end within the stream")


def build_rules(config: ReferenceConfig) -> list:
    if config.rule == "surrogate_bptt":
        return [SurrogateBPTT(config.surrogate_bptt)]
    if config.rule == "three_factor":
        return [ThreeFactor(config.three_factor)]
    return [STDP(config.stdp), ThreeFactor(replace(config.three_factor, train_hidden=False))]


def load_data(config: ReferenceConfig) -> tuple[Split, Split, int]:
    """Train/test splits and the number of classes."""
    if config.dataset == "mnist":
        return (*load_mnist(config.mnist), config.mnist.n_classes)
    if config.dataset == "nmnist":
        return (*load_nmnist(config.nmnist), config.nmnist.n_classes)
    return (*synthetic_prototypes(config.data), config.data.n_classes)


def input_shape(config: ReferenceConfig) -> tuple[int, ...]:
    if config.dataset == "nmnist":
        return config.nmnist.shape
    return config.mnist.shape if config.dataset == "mnist" else tuple(config.data.shape)


def run_seed(config: ReferenceConfig, seed: int) -> RunRecord:
    return train_and_record(config, seed)[0]


def train_and_record(config: ReferenceConfig, seed: int,
                     data: tuple[Split, Split, int] | None = None) -> tuple[RunRecord, SpikingNet, list]:
    """Run one seed; return its record plus the trained model and bound rules.

    ``data`` = (train, evaluation split, n_classes) overrides the configured data; the
    evaluation split is ``test`` normally and ``tune`` during tuning, and metric names
    carry its role (``pretrain/test/...`` or ``pretrain/tune/...``).
    """
    generator = seed_everything(seed)
    train, test, n_classes = data or load_data(config)
    role = test.role
    model = build_model(config.model, input_shape(config), n_classes, config.neuron)
    rules = build_rules(config)
    for rule in rules:
        rule.init_state(model)

    def test_accuracy(shifts: tuple[ShiftConfig, ...] = ()) -> float:
        """Clean test accuracy, or under ``shifts`` at full strength."""
        return evaluate(model, test, config.encoding, config.pretrain.batch_size, generator,
                        shifts, n_classes)["accuracy"]

    pretrain = run_offline(model, rules, train, test, config.encoding, config.pretrain, generator)
    clean_before, shifted_before = pretrain["test"]["accuracy"], test_accuracy((config.shift,))
    stream = Stream(train, config.stream.length, n_classes, seed=seed, shifts=(config.shift,))
    online = run_prequential(model, rules, stream, config.encoding, config.stream, generator)
    clean_after, shifted_after = test_accuracy(), test_accuracy((config.shift,))

    metrics = {f"pretrain/{role}/{k}": v for k, v in pretrain["test"].items()}
    metrics.update({f"stream/{k}": v for k, v in online["metrics"].items()})
    metrics.update({
        f"{role}_clean/before_stream": clean_before,
        f"{role}_clean/after_stream": clean_after,
        f"{role}_clean/forgetting": forgetting([clean_before, clean_after]),
        f"{role}_shifted/before_stream": shifted_before,
        f"{role}_shifted/after_stream": shifted_after,
    })

    with torch.no_grad():
        _, observations = model(torch.zeros(1, 1, *input_shape(config)))
    n_neurons = count_neurons(observations)
    ops = pretrain["test"]
    inference = energy_proxy_pj(config.energy, synops=ops["synops_per_sample"], macs=ops["input_macs_per_sample"],
                                neuron_updates=ops["neuron_updates_per_sample"])
    metrics.update({f"pretrain/{role}/energy_proxy_per_sample/{k}": v for k, v in inference.items()})
    window = config.shift.schedule.end - config.shift.schedule.start
    adaptation = energy_proxy_pj(
        config.energy,
        synops=metrics["stream/shift0/adaptation_synops"],
        macs=metrics["stream/shift0/adaptation_input_macs"],
        neuron_updates=n_neurons * config.encoding.T * window,
        weight_updates=metrics["stream/shift0/adaptation_weights_written"],
    )
    metrics.update({f"stream/shift0/adaptation_energy_proxy/{k}": v for k, v in adaptation.items()})
    record = RunRecord(
        experiment=config.experiment,
        config=resolved(config),
        seed=seed,
        T=config.encoding.T,
        provenance=provenance(config.device),
        rule_declarations=[asdict(rule.declaration) for rule in rules],
        model={
            "kind": config.model.kind,
            "n_neurons": n_neurons,
            "n_synapses": synapse_count(model, observations),
            "n_weights": sum(p.numel() for n, p in model.named_parameters() if n.endswith("weight")),
        },
        dataset={"version": train.version, "evaluation_version": test.version, "evaluation_role": role,
                 "n_train": len(train.y), f"n_{role}": len(test.y),
                 f"shifted_{role}": f"{role} split under `shift` at full strength"},
        protocol="offline_pretrain -> prequential_stream",
        metrics=metrics,
        traces={**{f"pretrain/{k}": v for k, v in pretrain["traces"].items()},
                **{f"stream/{k}": v for k, v in online["traces"].items()}},
        shift=resolved(config.shift),
        energy_proxy=inference["total_pj"],
        proxy_assumptions={**config.energy.assumptions(),
                           "quantity": f"inference energy per clean {role} sample after pretraining"},
    )
    return record, model, rules
