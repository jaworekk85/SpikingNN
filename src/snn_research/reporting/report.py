"""Write a comparison report into a new, immutable directory.

``results/<experiment>/report_<UTC timestamp>_<hash>/`` gets ``report.md`` (tables of
mean ± sample std over seeds), ``figure.png`` and ``summary.json``. An existing
directory is never reused.
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

from .compare import aggregate, check_comparable, comparison_signature, metrics_table, summarize
from .figures import reference_figure

Column = tuple[str, str, int]  # (header, metric key, digits)


def write_report(records: Sequence[dict[str, Any]], results_root: str | Path,
                 tables: dict[str, Sequence[Column]], allow_mismatch: bool = False,
                 figure: Callable[[Sequence[dict[str, Any]], Path], Path] = reference_figure) -> Path:
    mismatches = check_comparable(records, allow_mismatch)
    groups = aggregate(records)
    experiment = records[0]["experiment"]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    digest = hashlib.sha256(json.dumps([r["created_utc"] for r in records]).encode() + os.urandom(8)).hexdigest()[:8]
    out = Path(results_root) / experiment / f"report_{stamp}_{digest}"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.mkdir(exist_ok=False)

    figure(records, out / "figure.png")
    signature = comparison_signature(records[0])
    commits = sorted({str(r["provenance"].get("git_commit"))[:7] + ("+dirty" if r["provenance"].get("git_dirty") else "")
                      for r in records})
    run_counts = ", ".join(f"{label}: {len(metrics['seed'])}" for label, metrics in sorted(groups.items()))
    datasets = sorted({str(r.get("dataset", {}).get("version")) for r in records})
    lines = [
        f"# Report: {experiment}",
        "",
        f"Generated {stamp} from {len(records)} runs ({run_counts}).",
        f"Code versions: {', '.join(commits)}. Data: {', '.join(datasets)}.",
        "",
        "**Pipeline validation, not a research result.** Energy values are",
        "proxies from logical operation counts with the saved assumptions below, not measurements.",
        "",
        "## Comparability",
        "",
    ]
    if mismatches:
        lines += ["**Mismatched settings (compared only because `allow_mismatch` was set):**", ""]
        lines += [f"- {m}" for m in mismatches]
    else:
        lines += ["Identical across runs: " + "; ".join(f"{k} = {v}" for k, v in signature.items()) + "."]
    shift = records[0].get("shift")
    if shift:
        lines += ["", f"Shift: `{shift['kind']}`, magnitude {shift['magnitude']}, schedule {shift['schedule']}."]
    for title, columns in tables.items():
        lines += ["", f"## {title}", "", "Mean ± sample standard deviation over seeds; n/a = unavailable "
                  "(e.g. never recovered).", "", metrics_table(groups, columns)]
    assumptions = records[0].get("proxy_assumptions")
    if assumptions:
        lines += ["", "## Energy-proxy assumptions", ""]
        lines += [f"- {k}: {v}" for k, v in assumptions.items()]
    lines += ["", "## Figure", "", "![reference figure](figure.png)", ""]
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")

    summary = {label: {key: summarize(values) for key, values in metrics.items() if key != "seed"}
               for label, metrics in groups.items()}
    (out / "summary.json").write_text(json.dumps({"mismatches": mismatches, "groups": summary}, indent=2),
                                      encoding="utf-8")
    return out
