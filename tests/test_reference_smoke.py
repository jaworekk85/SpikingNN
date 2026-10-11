"""End-to-end smoke run of the reference pipeline on synthetic data."""

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "experiments" / "reference_tiny.yaml"


def load_reference():
    spec = importlib.util.spec_from_file_location("reference_run", ROOT / "experiments" / "reference" / "run.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # needed by dataclasses during exec
    spec.loader.exec_module(module)
    return module


def test_reference_runs_learns_and_is_deterministic(tmp_path):
    reference = load_reference()
    run_dirs = reference.main(["--config", str(CONFIG), "--results", str(tmp_path)])
    assert len(run_dirs) == 2  # one record per configured seed

    record = json.loads((run_dirs[0] / "record.json").read_text())
    assert record["metrics"]["test/accuracy"] > 0.4  # 4 classes: chance is 0.25
    assert record["traces"]["train_loss"][-1] < record["traces"]["train_loss"][0]
    assert record["rule_declaration"]["locality"] == "global_gradient"
    assert record["energy_proxy"] is None and record["energy_j"] is None

    config = reference.load_config(reference.ReferenceConfig, CONFIG)
    a, b = reference.run_seed(config, 0), reference.run_seed(config, 0)
    assert a.metrics == b.metrics and a.traces == b.traces
