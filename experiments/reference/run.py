"""Reference pipeline (Phase 1 subset): synthetic data -> encoder -> LIF SNN ->
surrogate BPTT -> offline evaluation -> one RunRecord per seed.

Pipeline validation only; the numbers are not research results.

    python experiments/reference/run.py --config configs/experiments/reference_tiny.yaml
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch

from snn_research.datasets import SyntheticConfig, synthetic_prototypes
from snn_research.encoding import EncodingConfig
from snn_research.learning_rules import SurrogateBPTT, SurrogateBPTTConfig
from snn_research.models import ModelConfig, build_model, count_neurons
from snn_research.neurons import LIFParams
from snn_research.protocols import OfflineConfig, run_offline
from snn_research.records import RunRecord, save_run
from snn_research.utils import load_config, provenance, resolved, seed_everything


@dataclass(frozen=True)
class ReferenceConfig:
    experiment: str = "reference"
    seeds: tuple[int, ...] = (0,)
    device: str = "cpu"
    data: SyntheticConfig = field(default_factory=SyntheticConfig)
    encoding: EncodingConfig = field(default_factory=EncodingConfig)
    neuron: LIFParams = field(default_factory=LIFParams)
    model: ModelConfig = field(default_factory=ModelConfig)
    rule: SurrogateBPTTConfig = field(default_factory=SurrogateBPTTConfig)
    protocol: OfflineConfig = field(default_factory=OfflineConfig)

    def __post_init__(self) -> None:
        if not self.seeds:
            raise ValueError("at least one seed is required")
        if self.device != "cpu":
            raise ValueError("only device 'cpu' is supported in Phase 1")


def run_seed(config: ReferenceConfig, seed: int) -> RunRecord:
    generator = seed_everything(seed)
    train, test = synthetic_prototypes(config.data)
    model = build_model(config.model, tuple(train.x.shape[1:]), config.data.n_classes, config.neuron)
    rule = SurrogateBPTT(config.rule)
    result = run_offline(model, rule, train, test, config.encoding, config.protocol, generator)

    with torch.no_grad():
        _, observations = model(train.x[:1].unsqueeze(0))
    return RunRecord(
        experiment=config.experiment,
        config=resolved(config),
        seed=seed,
        T=config.encoding.T,
        provenance=provenance(config.device),
        rule_declaration=asdict(rule.declaration),
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
    args = parser.parse_args(argv)

    config = load_config(ReferenceConfig, args.config)
    run_dirs = []
    for seed in config.seeds:
        record = run_seed(config, seed)
        run_dir = save_run(record, args.results)
        run_dirs.append(run_dir)
        print(f"seed {seed}: test accuracy {record.metrics['test/accuracy']:.3f}, "
              f"spike rate {record.metrics['test/overall/spike_rate']:.3f} -> {run_dir}")
    return run_dirs


if __name__ == "__main__":
    main()
