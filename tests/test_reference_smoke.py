"""End-to-end smoke runs of the reference pipeline on synthetic data, for every rule."""

import importlib.util
import json
import sys
from dataclasses import replace
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "experiments" / "reference_tiny.yaml"


def load_reference():
    spec = importlib.util.spec_from_file_location("reference_run", ROOT / "experiments" / "reference" / "run.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # needed by dataclasses during exec
    spec.loader.exec_module(module)
    return module


reference = load_reference()


def test_reference_cli_writes_one_record_per_seed(tmp_path):
    run_dirs = reference.main(["--config", str(CONFIG), "--results", str(tmp_path)])
    assert len(run_dirs) == 2
    record = json.loads((run_dirs[0] / "record.json").read_text())
    assert record["traces"]["train_loss"][-1] < record["traces"]["train_loss"][0]
    assert record["energy_proxy"] is None and record["energy_j"] is None


@pytest.mark.parametrize(
    "rule, localities",
    [
        ("surrogate_bptt", ["global_gradient"]),
        ("three_factor", ["global_modulated"]),
        ("stdp", ["synapse_local", "global_modulated"]),
    ],
)
def test_same_model_and_data_train_with_each_rule(rule, localities):
    config = replace(reference.load_config(reference.ReferenceConfig, CONFIG), rule=rule)
    a, b = reference.run_seed(config, 0), reference.run_seed(config, 0)
    assert a.metrics["test/accuracy"] > 0.4  # 4 classes: chance is 0.25
    assert [d["locality"] for d in a.rule_declarations] == localities
    assert a.metrics == b.metrics and a.traces == b.traces  # same seed -> same result
