import json

import pytest

from snn_research.reporting import (
    aggregate,
    check_comparable,
    load_records,
    metrics_table,
    summarize,
    trailing_mean,
    write_report,
)


def record(rule: str, seed: int, acc: float, recovery: int | None = 10, T: int = 20, n_synapses: int = 640) -> dict:
    return {
        "experiment": "unit",
        "created_utc": f"2026{seed}",
        "seed": seed,
        "T": T,
        "protocol": "p",
        "config": {"rule": rule, "encoding": {"kind": "rate"}, "stream": {"window": 2, "length": 6}},
        "model": {"n_neurons": 32, "n_synapses": n_synapses},
        "rule_declarations": [{"name": rule}],
        "provenance": {"git_commit": "abcdef123", "git_dirty": False},
        "shift": {"kind": "noise", "magnitude": 0.3, "schedule": {"start": 2, "end": 4}},
        "metrics": {"acc": acc, "recovery": recovery, "pretrain/test/synops_per_sample": 100.0 + seed,
                    "pretrain/test/accuracy": acc},
        "traces": {"stream/correct": [1, 1, 0, 0, 1, 1]},
        "proxy_assumptions": {"kind": "proxy (not measured)", "units": "pJ"},
    }


def test_comparability_guard_refuses_mismatches_unless_allowed():
    same = [record("a", 0, 0.9), record("b", 0, 0.8)]
    assert check_comparable(same) == []
    different = [record("a", 0, 0.9), record("b", 0, 0.8, T=10, n_synapses=100)]
    with pytest.raises(ValueError, match="not comparable"):
        check_comparable(different)
    mismatches = check_comparable(different, allow_mismatch=True)
    assert any(m.startswith("T differs") for m in mismatches)
    assert any(m.startswith("model size") for m in mismatches)


def test_aggregate_and_summarize_keep_unavailable_values_explicit():
    groups = aggregate([record("a", 0, 0.8), record("a", 1, 1.0, recovery=None), record("b", 0, 0.5)])
    assert groups["a"]["seed"] == [0, 1] and groups["a"]["recovery"] == [10, None]
    s = summarize(groups["a"]["acc"])
    assert s["mean"] == pytest.approx(0.9) and s["std"] == pytest.approx(0.1414, abs=1e-4) and s["n"] == 2
    assert summarize([None])["mean"] is None
    table = metrics_table(groups, [("accuracy", "acc", 2), ("recovery", "recovery", 0)])
    assert "| a | 2 | 0.90 ± 0.14 | 10 (1 n/a) |" in table
    assert "| b | 1 | 0.50 | 10 |" in table


def test_trailing_mean():
    assert trailing_mean([1, 0, 1, 1], window=2) == [1.0, 0.5, 0.5, 1.0]


def test_write_report_creates_a_new_directory_with_all_outputs(tmp_path):
    runs = tmp_path / "runs"
    for i, rec in enumerate([record("a", 0, 0.8), record("a", 1, 0.9), record("b", 0, 0.7)]):
        (runs / f"run{i}").mkdir(parents=True)
        (runs / f"run{i}" / "record.json").write_text(json.dumps(rec))
    records = load_records([runs])
    first = write_report(records, tmp_path / "out", {"Quality": [("accuracy", "acc", 2)]})
    second = write_report(records, tmp_path / "out", {"Quality": [("accuracy", "acc", 2)]})
    assert first != second  # reports are never overwritten
    text = (first / "report.md").read_text(encoding="utf-8")
    assert "Identical across runs" in text and "| a | 2 | 0.85 ± 0.07 |" in text and "not measurements" in text
    assert (first / "figure.png").stat().st_size > 10_000
    summary = json.loads((first / "summary.json").read_text())
    assert summary["groups"]["a"]["acc"]["n"] == 2 and summary["mismatches"] == []
    with pytest.raises(FileNotFoundError):
        load_records([tmp_path / "empty"])
