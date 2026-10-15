"""Tune learning-rule hyperparameters on a tune split; the test split is never used.

    python experiments/reference/tune.py --config configs/experiments/reference_mnist.yaml [--rules three_factor stdp]

The training data is split (seeded) into fit and ``tune`` parts. Every candidate in
the config's ``tuning.grids[rule]`` runs the full reference protocol (pretraining,
shifted stream) on the fit part with each tuning seed, and is scored on the tune
part by

    objective = (tune accuracy after pretraining + shifted-tune accuracy after the stream) / 2

which rewards both learning and adaptation. Results go to a new
``results/<experiment>/tuning_<UTC>_<hash>/`` directory (``tuning.json`` with every
score, ``best_overrides.yaml`` to copy into the experiment config).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import yaml
from pipeline import RULES, ReferenceConfig, load_data, train_and_record

from snn_research.protocols import grid_search
from snn_research.utils import load_config, override, provenance, resolved

OBJECTIVE = "(pretrain/tune/accuracy + tune_shifted/after_stream) / 2"


def objective(metrics: dict) -> float:
    return (metrics["pretrain/tune/accuracy"] + metrics["tune_shifted/after_stream"]) / 2


def tune_rule(config: ReferenceConfig, rule: str, fit, tune, n_classes: int) -> dict:
    base = replace(config, rule=rule)

    def score(params: dict, seed: int, train, tune_split) -> float:
        candidate = base
        for key, value in params.items():
            candidate = override(candidate, key, value)
        value = objective(train_and_record(candidate, seed, data=(train, tune_split, n_classes))[0].metrics)
        print(f"  {rule} seed {seed} {params}: {value:.3f}", flush=True)
        return value

    return grid_search(config.tuning.grids[rule], score, fit, tune, config.tuning.seeds)


def main(argv: list[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--rules", nargs="+", choices=RULES, help="default: every rule with a grid")
    parser.add_argument("--results", type=Path, default=Path("results"))
    args = parser.parse_args(argv)

    config = load_config(ReferenceConfig, args.config)
    rules = args.rules or [r for r in RULES if r in config.tuning.grids]
    missing = [r for r in rules if r not in config.tuning.grids]
    if missing:
        raise SystemExit(f"no tuning grid for {missing} in {args.config}")
    train, _test_never_used, n_classes = load_data(config)
    del _test_never_used
    fit, tune = train.split_off_tune(config.tuning.tune_fraction, config.tuning.split_seed)

    results = {}
    for rule in rules:
        print(f"{rule}: {len(config.tuning.grids[rule])} parameters, seeds {list(config.tuning.seeds)}")
        results[rule] = tune_rule(config, rule, fit, tune, n_classes)
        print(f"{rule}: best {results[rule]['best']['params']} (objective {results[rule]['best']['mean']:.3f})")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    digest = hashlib.sha256(json.dumps(results, sort_keys=True).encode() + os.urandom(8)).hexdigest()[:8]
    out = args.results / config.experiment / f"tuning_{stamp}_{digest}"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.mkdir(exist_ok=False)
    summary = {
        "objective": OBJECTIVE,
        "config": resolved(config),
        "fit_version": fit.version, "tune_version": tune.version,
        "n_fit": len(fit.y), "n_tune": len(tune.y),
        "test_split_used": False,
        "provenance": provenance(config.device),
        "results": results,
    }
    (out / "tuning.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    best = {rule: result["best"]["params"] for rule, result in results.items()}
    (out / "best_overrides.yaml").write_text(yaml.safe_dump(best, sort_keys=False), encoding="utf-8")
    print(f"tuning -> {out}")
    return out


if __name__ == "__main__":
    main()
