"""Reference pipeline: data (synthetic, MNIST or N-MNIST) -> encoder -> LIF SNN -> learning rule.

1. Offline pretraining on clean data; test accuracy on clean and fully shifted test data.
2. Prequential stream with one scheduled shift (pre-shift -> shift -> post-shift),
   adapting online with the same rule(s) unless ``stream.adapt`` is false.
3. Clean and shifted test accuracy again; forgetting of the clean condition.
4. Operation counts and an energy *proxy* (explicit assumptions in ``energy``) for
   test inference and, separately, for adaptation during the shift window.
One RunRecord per seed. ``--rule all`` runs every rule; ``--report`` then writes a
multi-seed comparison report (tables + figure) for the runs just made.

Rules: ``surrogate_bptt`` (global gradient), ``three_factor`` (error-modulated,
all layers) or ``stdp`` (STDP on hidden layers + three-factor delta rule on the
readout, since unsupervised STDP alone cannot learn labels).
Pipeline validation only; the numbers are not research results.

    python experiments/reference/run.py --config configs/experiments/reference_tiny.yaml --rule all --report
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
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
from snn_research.learning_rules import (
    STDP,
    STDPConfig,
    SurrogateBPTT,
    SurrogateBPTTConfig,
    ThreeFactor,
    ThreeFactorConfig,
)
from snn_research.backends import HardwareConstraints, check_constraints, export_network, save_export
from snn_research.models import ModelConfig, SpikingNet, build_model, count_neurons
from snn_research.neurons import LIFParams
from snn_research.metrics import forgetting, synapse_count
from snn_research.protocols import OfflineConfig, PrequentialConfig, evaluate, run_offline, run_prequential
from snn_research.records import RunRecord, save_run
from snn_research.reporting import load_records, write_report
from snn_research.shifts import ShiftConfig
from snn_research.utils import load_config, provenance, resolved, seed_everything


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

REPORT_TABLES = {
    "Quality and adaptation": [
        ("pretrain test acc", "pretrain/test/accuracy", 3),
        ("stream before", "stream/shift0/quality_before", 3),
        ("stream during", "stream/shift0/quality_during", 3),
        ("stream after", "stream/shift0/quality_after", 3),
        ("recovery (samples)", "stream/shift0/recovery_samples", 0),
        ("clean test after stream", "test_clean/after_stream", 3),
        ("forgetting (clean)", "test_clean/forgetting", 3),
        ("shifted test after stream", "test_shifted/after_stream", 3),
    ],
    "Activity and inference cost (per test sample, after pretraining)": [
        ("spike rate", "pretrain/test/overall/spike_rate", 3),
        ("spikes", "pretrain/test/spikes_per_sample", 0),
        ("synops", "pretrain/test/synops_per_sample", 0),
        ("synops into layer0 (input)", "pretrain/test/synops_per_sample/layer0", 0),
        ("synops into readout", "pretrain/test/synops_per_sample/readout", 0),
        ("energy proxy (pJ)", "pretrain/test/energy_proxy_per_sample/total_pj", 0),
    ],
    "Adaptation cost (shift window, separate fields)": [
        ("update steps", "stream/shift0/adaptation_updates", 0),
        ("weights written", "stream/shift0/adaptation_weights_written", 0),
        ("spikes", "stream/shift0/adaptation_spikes", 0),
        ("synops", "stream/shift0/adaptation_synops", 0),
        ("energy proxy (pJ)", "stream/shift0/adaptation_energy_proxy/total_pj", 0),
    ],
}


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


def main(argv: list[str] | None = None) -> list[Path]:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--rule", choices=(*RULES, "all"), help="override the configured rule; 'all' runs each")
    parser.add_argument("--report", action="store_true", help="write a comparison report for these runs")
    parser.add_argument("--export", action="store_true",
                        help="write export.json + weights.npz (model + rule description) into each run directory")
    parser.add_argument("--hardware", type=Path,
                        help="hardware-constraints YAML to check exports against (implies --export)")
    args = parser.parse_args(argv)
    hardware = load_config(HardwareConstraints, args.hardware) if args.hardware else None

    base = load_config(ReferenceConfig, args.config)
    rules = RULES if args.rule == "all" else [args.rule or base.rule]
    run_dirs = []
    for rule in rules:
        config = replace(base, rule=rule)  # saved in the resolved config
        for seed in config.seeds:
            record, model, rules_used = train_and_record(config, seed)
            run_dir = save_run(record, args.results)
            run_dirs.append(run_dir)
            report_line(config.rule, seed, record, run_dir)
            if args.export or hardware:
                export_run(config, model, rules_used, run_dir, hardware)
    if args.report:
        report_dir = write_report(load_records(run_dirs), args.results, REPORT_TABLES)
        print(f"report -> {report_dir / 'report.md'}")
    return run_dirs


def export_run(config: ReferenceConfig, model: SpikingNet, rules: list, run_dir: Path,
               hardware: HardwareConstraints | None) -> None:
    """Export the trained model + rules; optionally check them against hardware limits."""
    description, weights = export_network(model, rules, config.encoding, input_shape(config))
    if hardware:
        findings = check_constraints(description, hardware)
        description["constraint_check"] = {"hardware": hardware.name, "status": hardware.status,
                                           "findings": [{"level": lv, "message": m} for lv, m in findings]}
        counts = {lv: sum(f[0] == lv for f in findings) for lv in ("error", "warning")}
        print(f"  export + {hardware.name} check: {counts['error']} errors, {counts['warning']} warnings")
    save_export(description, weights, run_dir)


def report_line(rule: str, seed: int, record: RunRecord, run_dir: Path) -> None:
    m = record.metrics
    print(f"{rule} seed {seed}: pretrain test {m['pretrain/test/accuracy']:.3f} | stream "
          f"before/during/after {m['stream/shift0/quality_before']:.3f}/"
          f"{m['stream/shift0/quality_during']:.3f}/{m['stream/shift0/quality_after']:.3f}, "
          f"recovery {m['stream/shift0/recovery_samples']} | clean test after "
          f"{m['test_clean/after_stream']:.3f} | synops/sample {m['pretrain/test/synops_per_sample']:.0f}, "
          f"proxy {record.energy_proxy:.0f} pJ/sample -> {run_dir}")


if __name__ == "__main__":
    main()
