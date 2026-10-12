"""Reference pipeline: synthetic data -> encoder -> LIF SNN -> learning rule.

1. Offline pretraining on clean data; test accuracy on clean and fully shifted test data.
2. Prequential stream with one scheduled shift (pre-shift -> shift -> post-shift),
   adapting online with the same rule(s) unless ``stream.adapt`` is false.
3. Clean and shifted test accuracy again; forgetting of the clean condition.
4. Operation counts and an energy *proxy* (explicit assumptions in ``energy``) for
   test inference and, separately, for adaptation during the shift window.
One RunRecord per seed.

Rules: ``surrogate_bptt`` (global gradient), ``three_factor`` (error-modulated,
all layers) or ``stdp`` (STDP on hidden layers + three-factor delta rule on the
readout, since unsupervised STDP alone cannot learn labels).
Pipeline validation only; the numbers are not research results.

    python experiments/reference/run.py --config configs/experiments/reference_tiny.yaml [--rule stdp]
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import torch

from snn_research.datasets import Stream, SyntheticConfig, synthetic_prototypes
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
from snn_research.models import ModelConfig, build_model, count_neurons
from snn_research.neurons import LIFParams
from snn_research.metrics import forgetting, synapse_count
from snn_research.protocols import OfflineConfig, PrequentialConfig, evaluate, run_offline, run_prequential
from snn_research.records import RunRecord, save_run
from snn_research.shifts import ShiftConfig
from snn_research.utils import load_config, provenance, resolved, seed_everything


RULES = ("surrogate_bptt", "three_factor", "stdp")


@dataclass(frozen=True)
class ReferenceConfig:
    experiment: str = "reference"
    seeds: tuple[int, ...] = (0,)
    device: str = "cpu"
    data: SyntheticConfig = field(default_factory=SyntheticConfig)
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

    def __post_init__(self) -> None:
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


def run_seed(config: ReferenceConfig, seed: int) -> RunRecord:
    generator = seed_everything(seed)
    n_classes = config.data.n_classes
    train, test = synthetic_prototypes(config.data)
    model = build_model(config.model, tuple(train.x.shape[1:]), n_classes, config.neuron)
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

    metrics = {f"pretrain/test/{k}": v for k, v in pretrain["test"].items()}
    metrics.update({f"stream/{k}": v for k, v in online["metrics"].items()})
    metrics.update({
        "test_clean/before_stream": clean_before,
        "test_clean/after_stream": clean_after,
        "test_clean/forgetting": forgetting([clean_before, clean_after]),
        "test_shifted/before_stream": shifted_before,
        "test_shifted/after_stream": shifted_after,
    })

    with torch.no_grad():
        _, observations = model(train.x[:1].unsqueeze(0))
    n_neurons = count_neurons(observations)
    ops = pretrain["test"]
    inference = energy_proxy_pj(config.energy, synops=ops["synops_per_sample"], macs=ops["input_macs_per_sample"],
                                neuron_updates=ops["neuron_updates_per_sample"])
    metrics.update({f"pretrain/test/energy_proxy_per_sample/{k}": v for k, v in inference.items()})
    window = config.shift.schedule.end - config.shift.schedule.start
    adaptation = energy_proxy_pj(
        config.energy,
        synops=metrics["stream/shift0/adaptation_synops"],
        macs=metrics["stream/shift0/adaptation_input_macs"],
        neuron_updates=n_neurons * config.encoding.T * window,
        weight_updates=metrics["stream/shift0/adaptation_weights_written"],
    )
    metrics.update({f"stream/shift0/adaptation_energy_proxy/{k}": v for k, v in adaptation.items()})
    return RunRecord(
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
        dataset={"version": train.version, "n_train": len(train.y), "n_test": len(test.y),
                 "shifted_test": "test split under `shift` at full strength"},
        protocol="offline_pretrain -> prequential_stream",
        metrics=metrics,
        traces={**{f"pretrain/{k}": v for k, v in pretrain["traces"].items()},
                **{f"stream/{k}": v for k, v in online["traces"].items()}},
        shift=resolved(config.shift),
        energy_proxy=inference["total_pj"],
        proxy_assumptions={**config.energy.assumptions(),
                           "quantity": "inference energy per clean test sample after pretraining"},
    )


def main(argv: list[str] | None = None) -> list[Path]:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--rule", choices=RULES, help="override the configured rule")
    args = parser.parse_args(argv)

    config = load_config(ReferenceConfig, args.config)
    if args.rule:
        config = replace(config, rule=args.rule)  # saved in the resolved config
    run_dirs = []
    for seed in config.seeds:
        record = run_seed(config, seed)
        run_dir = save_run(record, args.results)
        run_dirs.append(run_dir)
        m = record.metrics
        print(f"{config.rule} seed {seed}: pretrain test {m['pretrain/test/accuracy']:.3f} | stream "
              f"before/during/after {m['stream/shift0/quality_before']:.3f}/"
              f"{m['stream/shift0/quality_during']:.3f}/{m['stream/shift0/quality_after']:.3f}, "
              f"recovery {m['stream/shift0/recovery_samples']} | clean test after "
              f"{m['test_clean/after_stream']:.3f} | synops/sample {m['pretrain/test/synops_per_sample']:.0f}, "
              f"proxy {record.energy_proxy:.0f} pJ/sample -> {run_dir}")
    return run_dirs


if __name__ == "__main__":
    main()
