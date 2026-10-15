import json
from dataclasses import dataclass, field
from pathlib import Path

import pytest
import torch
import tune
import yaml

from snn_research.datasets import Split
from snn_research.protocols import grid, grid_search
from snn_research.utils import override

ROOT = Path(__file__).resolve().parents[1]


def split(role: str, n: int = 20) -> Split:
    return Split(torch.arange(n, dtype=torch.float).view(n, 1), torch.arange(n) % 2, role, "v")


def test_tune_split_is_disjoint_seeded_and_role_tagged():
    train = split("train")
    fit, tune = train.split_off_tune(0.25, seed=3)
    assert (fit.role, tune.role) == ("train", "tune") and (len(fit.y), len(tune.y)) == (15, 5)
    assert not set(fit.x.view(-1).tolist()) & set(tune.x.view(-1).tolist())
    assert torch.equal(tune.x, train.split_off_tune(0.25, seed=3)[1].x)
    with pytest.raises(ValueError):
        split("test").split_off_tune(0.25, seed=0)


def test_grid_is_a_cartesian_product_in_saved_order():
    assert grid({"a": [1, 2], "b": ["x", "y"]}) == [
        {"a": 1, "b": "x"}, {"a": 1, "b": "y"}, {"a": 2, "b": "x"}, {"a": 2, "b": "y"}]


def test_grid_search_never_sees_test_data_and_picks_best_mean():
    seen_roles = set()

    def score(params, seed, train, tune):
        seen_roles.update({train.role, tune.role})
        return params["lr"] * (1 if seed == 0 else 3)

    result = grid_search({"lr": [0.1, 0.3, 0.2]}, score, split("train"), split("tune"), seeds=[0, 1])
    assert seen_roles == {"train", "tune"}
    assert result["best"]["params"] == {"lr": 0.3} and result["best"]["scores"] == pytest.approx([0.3, 0.9])
    for train, tune in [(split("train"), split("test")), (split("test"), split("tune")), (split("train"), split("train"))]:
        with pytest.raises(ValueError, match="test data must never be used"):
            grid_search({"lr": [0.1]}, score, train, tune, seeds=[0])


@dataclass(frozen=True)
class Inner:
    lr: float = 0.1

    def __post_init__(self):
        if self.lr <= 0:
            raise ValueError("lr must be positive")


@dataclass(frozen=True)
class Outer:
    inner: Inner = field(default_factory=Inner)
    seeds: tuple[int, ...] = (0,)


def test_override_replaces_nested_fields_and_revalidates():
    assert override(Outer(), "inner.lr", 0.5) == Outer(Inner(0.5))
    assert override(Outer(), "seeds", [1, 2]).seeds == (1, 2)
    with pytest.raises(ValueError, match="unknown config field"):
        override(Outer(), "inner.typo", 1)
    with pytest.raises(ValueError, match="positive"):
        override(Outer(), "inner.lr", -1.0)


def test_tune_script_scores_on_tune_split_only(tmp_path, monkeypatch):
    real = tune.train_and_record
    roles = []

    def spy(config, seed, data=None):
        roles.append((data[0].role, data[1].role))
        return real(config, seed, data=data)

    monkeypatch.setattr(tune, "train_and_record", spy)
    out = tune.main(["--config", str(ROOT / "configs" / "experiments" / "reference_tiny.yaml"),
                     "--rules", "three_factor", "--results", str(tmp_path)])
    assert roles and set(roles) == {("train", "tune")}
    summary = json.loads((out / "tuning.json").read_text())
    assert summary["test_split_used"] is False and summary["n_tune"] > 0
    assert len(summary["results"]["three_factor"]["candidates"]) == 2
    best = yaml.safe_load((out / "best_overrides.yaml").read_text())
    assert set(best) == {"three_factor"} and "three_factor.lr" in best["three_factor"]
