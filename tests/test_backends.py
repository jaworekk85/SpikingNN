import json
from pathlib import Path

import numpy as np
import pytest
import torch

from snn_research.backends import (
    HardwareConstraints,
    Measurement,
    check_constraints,
    export_network,
    save_export,
)
from snn_research.encoding import EncodingConfig
from snn_research.learning_rules import STDP, STDPConfig, SurrogateBPTT, SurrogateBPTTConfig, ThreeFactor, ThreeFactorConfig
from snn_research.metrics import synapse_count
from snn_research.models import ModelConfig, build_model
from snn_research.neurons import LIFParams
from snn_research.records import RunRecord
from snn_research.utils import load_config

ROOT = Path(__file__).resolve().parents[1]


def bound(model_config: ModelConfig, input_shape, rules):
    torch.manual_seed(0)
    model = build_model(model_config, input_shape, 3, LIFParams(tau_mem_ms=8.0, reset="zero", refractory_steps=1))
    for rule in rules:
        rule.init_state(model)
    return model


def test_export_describes_dense_network_and_rules(tmp_path):
    rules = [STDP(STDPConfig(weight_bits=8)), ThreeFactor(ThreeFactorConfig(train_hidden=False))]
    model = bound(ModelConfig(hidden=(6, 5)), (4,), rules)
    description, weights = export_network(model, rules, EncodingConfig("rate", T=12), (4,))
    json.dumps(description)  # fully serialisable
    assert description["timing"] == {"dt_ms": 1.0, "T": 12}
    neuron = description["neuron"]
    assert neuron["reset"] == "zero" and neuron["refractory_steps"] == 1
    assert neuron["beta"] == pytest.approx(np.exp(-1 / 8)) and neuron["input_scale"] == pytest.approx(1 - neuron["beta"])
    assert [(l["type"], l["neurons"], l["synapses"], l["max_fan_in"]) for l in description["layers"]] == [
        ("dense", 6, 24, 4), ("dense", 5, 30, 6)]
    with torch.no_grad():
        _, obs = model(torch.zeros(2, 1, 4))
    assert description["totals"]["synapses"] == synapse_count(model, obs) == 24 + 30 + 15
    names = [r["declaration"]["name"] for r in description["rules"]]
    assert names == ["stdp", "three_factor"]
    assert description["rules"][0]["parameters"]["weight_bits"] == 8
    assert description["rules"][1]["targets"] == ["readout"]
    assert description["rules"][1]["state"]["per_synapse"] == ["eligibility_trace"]

    path = save_export(description, weights, tmp_path)
    loaded = np.load(tmp_path / "weights.npz")
    np.testing.assert_allclose(loaded["layer0.weight"], model.layers[0][1].weight.detach().numpy())
    np.testing.assert_allclose(loaded["readout.weight"], model.readout.weight.detach().numpy())
    assert json.loads(path.read_text())["format"].startswith("snn_research.export/")
    with pytest.raises(FileExistsError):
        save_export(description, weights, tmp_path)


def test_export_of_conv_network_counts_logical_synapses():
    model = bound(ModelConfig(kind="conv", hidden=(5,), conv_channels=2), (1, 6, 6), [])
    description, _ = export_network(model, [SurrogateBPTT(SurrogateBPTTConfig())], EncodingConfig(T=4), (1, 6, 6))
    conv = description["layers"][0]
    assert conv["type"] == "conv2d" and conv["output_shape"] == [2, 3, 3] and conv["max_fan_in"] == 9
    with torch.no_grad():
        _, obs = model(torch.zeros(2, 1, 1, 6, 6))
    assert description["totals"]["synapses"] == synapse_count(model, obs)
    assert description["rules"][0]["targets"] == ["all parameters"]


def test_constraint_check_flags_mismatches_and_skips_unknowns():
    rules = [ThreeFactor(ThreeFactorConfig())]
    model = bound(ModelConfig(hidden=(6,)), (4,), rules)
    description, _ = export_network(model, rules, EncodingConfig(T=10), (4,))
    unknown = check_constraints(description, HardwareConstraints(name="unknown chip"))
    assert [lv for lv, _ in unknown if lv != "info"] == []
    strict = HardwareConstraints(name="strict", neuron_models=("IF",), resets=("subtract",), dt_ms=(0.1,),
                                 weight_bits=4, max_fan_in=2, cores=1, memory_per_core_bytes=8,
                                 on_chip_learning="none")
    messages = "\n".join(f"{lv}: {m}" for lv, m in check_constraints(description, strict))
    for expected in ("error: neuron model LIF", "error: reset 'zero'", "warning: dt 1.0 ms",
                     "error: max fan-in 4 > 2", "warning: three_factor learns float weights",
                     "error: on-chip learning: none", "error: estimated memory"):
        assert expected in messages
    with pytest.raises(ValueError):
        HardwareConstraints(name="x", status="assumed")


def test_spinnaker2_draft_config_loads_and_is_marked_unconfirmed():
    hw = load_config(HardwareConstraints, ROOT / "configs" / "hardware" / "spinnaker2_draft.yaml")
    assert hw.status == "to be confirmed" and hw.weight_bits == 8 and hw.sources


def test_hardware_measurement_is_the_only_path_to_measured_energy():
    m = Measurement(energy_j=0.5, source="board power monitor", method="integrated power", boundary="chip")
    base = dict(experiment="hw", config={}, seed=0, T=5, provenance={}, rule_declarations=[], model={},
                dataset={}, protocol="p", metrics={}, traces={})
    record = RunRecord(**base, **m.record_fields())
    assert record.energy_j == 0.5 and record.backend_kind == "hardware"
    with pytest.raises(ValueError):
        RunRecord(**base, energy_j=0.5)  # simulation records cannot carry measured energy
