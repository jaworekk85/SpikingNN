"""Write a multi-seed comparison report for saved reference runs.

    python experiments/reference/report.py results/reference_tiny [more run dirs ...] [--allow-mismatch]

Arguments are run directories or directories containing runs. Runs whose ``T``,
encoding, model size or protocol differ are refused unless ``--allow-mismatch`` is
given, in which case the mismatches are listed in the report.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from run import REPORT_TABLES

from snn_research.reporting import load_records, write_report


def main(argv: list[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("runs", type=Path, nargs="+")
    parser.add_argument("--results", type=Path, default=Path("results"), help="where the report directory goes")
    parser.add_argument("--allow-mismatch", action="store_true")
    args = parser.parse_args(argv)
    records = [r for r in load_records(args.runs) if r.get("experiment")]
    report_dir = write_report(records, args.results, REPORT_TABLES, allow_mismatch=args.allow_mismatch)
    print(f"report -> {report_dir / 'report.md'}")
    return report_dir


if __name__ == "__main__":
    main()
