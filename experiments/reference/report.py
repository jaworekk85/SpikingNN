"""Write a multi-seed comparison report for saved reference runs.

    python experiments/reference/report.py results/reference_tiny [more run dirs ...] [--allow-mismatch]

Arguments are run directories or directories containing runs. Runs whose ``T``,
encoding, model size or protocol differ are refused unless ``--allow-mismatch`` is
given, in which case the mismatches are listed in the report.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from snn_research.reporting import load_records, write_report


REPORT_TABLES = {
    "Quality and adaptation": [
        ("pretrain test acc", "pretrain/test/accuracy", 3),
        ("stream before", "stream/shift0/quality_before", 3),
        ("stream during", "stream/shift0/quality_during", 3),
        ("stream after", "stream/shift0/quality_after", 3),
        ("recovery (samples)", "stream/shift0/recovery_samples", 0),
        ("clean test after stream", "test_clean/after_stream", 3),
        ("forgetting (clean)", "test_clean/forgetting", 3),
        ("shifted test after stream", "test_shifted/after_stream", 3),
    ],
    "Activity and inference cost (per test sample, after pretraining)": [
        ("spike rate", "pretrain/test/overall/spike_rate", 3),
        ("spikes", "pretrain/test/spikes_per_sample", 0),
        ("synops", "pretrain/test/synops_per_sample", 0),
        ("synops into layer0 (input)", "pretrain/test/synops_per_sample/layer0", 0),
        ("synops into readout", "pretrain/test/synops_per_sample/readout", 0),
        ("energy proxy (pJ)", "pretrain/test/energy_proxy_per_sample/total_pj", 0),
    ],
    "Adaptation cost (shift window, separate fields)": [
        ("update steps", "stream/shift0/adaptation_updates", 0),
        ("weights written", "stream/shift0/adaptation_weights_written", 0),
        ("spikes", "stream/shift0/adaptation_spikes", 0),
        ("synops", "stream/shift0/adaptation_synops", 0),
        ("energy proxy (pJ)", "stream/shift0/adaptation_energy_proxy/total_pj", 0),
    ],
}


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
