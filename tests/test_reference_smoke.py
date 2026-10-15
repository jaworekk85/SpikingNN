"""End-to-end smoke runs of the reference pipeline on synthetic data, for every rule."""

import json
from dataclasses import replace
from pathlib import Path

import pipeline
import pytest
import run

from snn_research.utils import load_config

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "experiments" / "reference_tiny.yaml"


def test_reference_cli_writes_one_record_per_seed_and_a_report(tmp_path):
    hardware = ROOT / "configs" / "hardware" / "spinnaker2_draft.yaml"
    run_dirs = run.main(["--config", str(CONFIG), "--results", str(tmp_path), "--report",
                               "--hardware", str(hardware)])
    assert len(run_dirs) == 2
    export = json.loads((run_dirs[0] / "export.json").read_text())
    assert export["constraint_check"]["status"] == "to be confirmed"
    assert (run_dirs[0] / "weights.npz").exists()
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
    config = replace(load_config(pipeline.ReferenceConfig, CONFIG), rule=rule)
    a, b = pipeline.run_seed(config, 0), pipeline.run_seed(config, 0)
    assert a.metrics["pretrain/test/accuracy"] > 0.4  # 4 classes: chance is 0.25
    assert [d["locality"] for d in a.rule_declarations] == localities
    assert a.metrics == b.metrics and a.traces == b.traces  # same seed -> same result
