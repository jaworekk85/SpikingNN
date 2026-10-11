"""Reference pipeline: synthetic data -> encoder -> LIF SNN -> learning rule ->
offline evaluation -> one RunRecord per seed.

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

from snn_research.datasets import SyntheticConfig, synthetic_prototypes
from snn_research.encoding import EncodingConfig
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
from snn_research.protocols import OfflineConfig, run_offline
from snn_research.records import RunRecord, save_run
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
    protocol: OfflineConfig = field(default_factory=OfflineConfig)

    def __post_init__(self) -> None:
        if self.rule not in RULES:
            raise ValueError(f"rule must be one of {RULES}, got {self.rule!r}")
        if not self.seeds:
            raise ValueError("at least one seed is required")
        if self.device != "cpu":
            raise ValueError("only device 'cpu' is supported in Phase 1")


def build_rules(config: ReferenceConfig) -> list:
    if config.rule == "surrogate_bptt":
        return [SurrogateBPTT(config.surrogate_bptt)]
    if config.rule == "three_factor":
        return [ThreeFactor(config.three_factor)]
    return [STDP(config.stdp), ThreeFactor(replace(config.three_factor, train_hidden=False))]


def run_seed(config: ReferenceConfig, seed: int) -> RunRecord:
    generator = seed_everything(seed)
    train, test = synthetic_prototypes(config.data)
    model = build_model(config.model, tuple(train.x.shape[1:]), config.data.n_classes, config.neuron)
    rules = build_rules(config)
    result = run_offline(model, rules, train, test, config.encoding, config.protocol, generator)

    with torch.no_grad():
        _, observations = model(train.x[:1].unsqueeze(0))
    return RunRecord(
        experiment=config.experiment,
        config=resolved(config),
        seed=seed,
        T=config.encoding.T,
        provenance=provenance(config.device),
        rule_declarations=[asdict(rule.declaration) for rule in rules],
        model={
            "kind": config.model.kind,
            "n_neurons": count_neurons(observations),
            "n_weights": sum(p.numel() for n, p in model.named_parameters() if n.endswith("weight")),
        },
        dataset={"version": train.version, "n_train": len(train.y), "n_test": len(test.y)},
        protocol="offline",
        metrics={f"test/{k}": v for k, v in result["test"].items()},
        traces=result["traces"],
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
        print(f"{config.rule} seed {seed}: test accuracy {record.metrics['test/accuracy']:.3f}, "
              f"spike rate {record.metrics['test/overall/spike_rate']:.3f} -> {run_dir}")
    return run_dirs


if __name__ == "__main__":
    main()
