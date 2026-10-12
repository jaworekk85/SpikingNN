"""Load saved RunRecords, guard comparability, aggregate over seeds, render tables.

Runs are compared only if ``T``, encoding, model size (neurons and synapses) and
protocol match. Otherwise ``check_comparable`` raises, unless ``allow_mismatch=True``,
in which case the mismatches are returned so the report can show them.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Sequence


def load_records(paths: Sequence[str | Path]) -> list[dict[str, Any]]:
    """Read ``record.json`` from run directories, or from every run under a directory."""
    files: list[Path] = []
    for path in map(Path, paths):
        files += [path / "record.json"] if (path / "record.json").exists() else sorted(path.rglob("record.json"))
    if not files:
        raise FileNotFoundError(f"no record.json found under {list(map(str, paths))}")
    return [json.loads(f.read_text(encoding="utf-8")) for f in files]


def group_label(record: dict[str, Any]) -> str:
    """The compared condition: the configured rule, else the declared rule names."""
    return record["config"].get("rule") or "+".join(d["name"] for d in record["rule_declarations"])


def comparison_signature(record: dict[str, Any]) -> dict[str, Any]:
    model = record["model"]
    return {
        "T": record["T"],
        "encoding": record["config"].get("encoding", {}).get("kind"),
        "model size (neurons, synapses)": (model.get("n_neurons"), model.get("n_synapses")),
        "protocol": record["protocol"],
    }


def check_comparable(records: Sequence[dict[str, Any]], allow_mismatch: bool = False) -> list[str]:
    """Return mismatch descriptions; raise if there are any and ``allow_mismatch`` is false."""
    mismatches = []
    for field in comparison_signature(records[0]):
        values = sorted({str(comparison_signature(r)[field]) for r in records})
        if len(values) > 1:
            mismatches.append(f"{field} differs: {', '.join(values)}")
    if mismatches and not allow_mismatch:
        raise ValueError("runs are not comparable: " + "; ".join(mismatches)
                         + " (pass allow_mismatch=True to compare anyway)")
    return mismatches


def aggregate(records: Sequence[dict[str, Any]]) -> dict[str, dict[str, list[float | None]]]:
    """``{group: {metric: [value per run]}}``; ``None`` marks unavailable values."""
    runs_by_group: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        runs_by_group.setdefault(group_label(record), []).append(record)
    groups = {}
    for label, runs in runs_by_group.items():
        keys = sorted(set().union(*(run["metrics"] for run in runs)))
        groups[label] = {key: [run["metrics"].get(key) for run in runs] for key in keys}
        groups[label]["seed"] = [run["seed"] for run in runs]
    return groups


def summarize(values: Sequence[float | None]) -> dict[str, float | int | None]:
    """Mean and sample standard deviation over available values, with counts."""
    present = [v for v in values if v is not None]
    n = len(present)
    mean = sum(present) / n if n else None
    std = math.sqrt(sum((v - mean) ** 2 for v in present) / (n - 1)) if n > 1 else None
    return {"mean": mean, "std": std, "n": n, "missing": len(values) - n}


def format_cell(values: Sequence[float | None], digits: int) -> str:
    s = summarize(values)
    if s["mean"] is None:
        return "n/a"
    text = f"{s['mean']:,.{digits}f}"
    if s["std"] is not None:
        text += f" ± {s['std']:,.{digits}f}"
    if s["missing"]:
        text += f" ({s['missing']} n/a)"
    return text


def metrics_table(groups: dict[str, dict[str, list[float | None]]],
                  columns: Sequence[tuple[str, str, int]]) -> str:
    """Markdown table: one row per group, ``(header, metric key, digits)`` per column."""
    header = "| condition | seeds | " + " | ".join(h for h, _, _ in columns) + " |"
    lines = [header, "|" + " --- |" * (len(columns) + 2)]
    for label, metrics in sorted(groups.items()):
        cells = [format_cell(metrics.get(key, []), digits) for _, key, digits in columns]
        lines.append(f"| {label} | {len(metrics['seed'])} | " + " | ".join(cells) + " |")
    return "\n".join(lines)
