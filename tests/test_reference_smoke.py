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


def test_reference_cli_writes_one_record_per_seed_and_a_report(tmp_path):
    run_dirs = reference.main(["--config", str(CONFIG), "--results", str(tmp_path), "--report"])
    assert len(run_dirs) == 2
    (report_dir,) = (tmp_path / "reference_tiny").glob("report_*")
    assert {p.name for p in report_dir.iterdir()} == {"report.md", "figure.png", "summary.json"}
    record = json.loads((run_dirs[0] / "record.json").read_text())
    loss = record["traces"]["pretrain/train_loss"]
    assert loss[-1] < loss[0]
    assert record["energy_j"] is None  # simulation: proxy only, never measured energy
    assert record["energy_proxy"] > 0 and record["proxy_assumptions"]["kind"].startswith("proxy")
    assert record["model"]["n_synapses"] > 0
    assert record["metrics"]["stream/shift0/adaptation_energy_proxy/total_pj"] > 0
    assert record["shift"]["kind"] == "noise" and len(record["traces"]["stream/correct"]) == 768
    for phase in ("before", "during", "after"):  # pre/during/post-shift quality always reported
        assert f"stream/shift0/quality_{phase}" in record["metrics"]
    for key in ("test_clean/forgetting", "stream/shift0/recovery_samples", "stream/shift0/adaptation_updates"):
        assert key in record["metrics"]


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
    assert a.metrics["pretrain/test/accuracy"] > 0.4  # 4 classes: chance is 0.25
    assert [d["locality"] for d in a.rule_declarations] == localities
    assert a.metrics == b.metrics and a.traces == b.traces  # same seed -> same result
