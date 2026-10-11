import json
from dataclasses import dataclass, field

import pytest
import torch

from snn_research.datasets import Split
from snn_research.records import RunRecord, save_run
from snn_research.utils import from_dict


def make_record(**overrides) -> RunRecord:
    fields = dict(
        experiment="unit", config={"a": 1}, seed=0, T=5, provenance={}, rule_declaration={},
        model={}, dataset={}, protocol="offline", metrics={"test/accuracy": 0.5}, traces={"loss": [1.0]},
    )
    return RunRecord(**{**fields, **overrides})


def test_save_creates_new_directories_and_never_overwrites(tmp_path, monkeypatch):
    record = make_record()
    first, second = save_run(record, tmp_path), save_run(record, tmp_path)
    assert first != second  # same record and timestamp, different run nonce
    assert json.loads((first / "record.json").read_text())["metrics"] == {"test/accuracy": 0.5}

    monkeypatch.setattr("os.urandom", lambda n: bytes(n))
    save_run(record, tmp_path)
    with pytest.raises(FileExistsError):
        save_run(record, tmp_path)  # identical path must not be reused


def test_energy_proxy_requires_assumptions():
    with pytest.raises(ValueError):
        make_record(energy_proxy=1.0)
    make_record(energy_proxy=1.0, proxy_assumptions={"pj_per_synop": 1.0})


def test_measured_energy_only_from_hardware_with_metadata():
    measurement = {"source": "x", "method": "y", "boundary": "chip", "units": "J"}
    with pytest.raises(ValueError):
        make_record(energy_j=1.0, measurement=measurement)  # simulation backend
    with pytest.raises(ValueError):
        make_record(energy_j=1.0, backend_kind="hardware", measurement={"source": "x"})
    make_record(energy_j=1.0, backend_kind="hardware", measurement=measurement)


def test_split_role_is_validated():
    with pytest.raises(ValueError):
        Split(torch.zeros(2, 3), torch.zeros(2), role="validation", version="v")


@dataclass(frozen=True)
class Inner:
    size: int = 1


@dataclass(frozen=True)
class Outer:
    inner: Inner = field(default_factory=Inner)
    sizes: tuple[int, ...] = (1,)


def test_config_loading_builds_nested_dataclasses_and_rejects_unknown_keys():
    config = from_dict(Outer, {"inner": {"size": 3}, "sizes": [2, 4]})
    assert config == Outer(Inner(3), (2, 4))
    with pytest.raises(ValueError, match="unknown keys"):
        from_dict(Outer, {"inner": {"size": 3, "typo": 1}})
