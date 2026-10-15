"""Run the reference experiment for one rule or all rules; optionally report and export.

    python experiments/reference/run.py --config configs/experiments/reference_tiny.yaml --rule all --report

See ``pipeline.py`` for what one run does. ``--report`` writes a multi-seed comparison
(tables + figure) for the runs just made; ``--export`` / ``--hardware <yaml>`` write the
model + rule description (and a hardware-constraint check) into each run directory.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from pipeline import RULES, ReferenceConfig, input_shape, train_and_record
from report import REPORT_TABLES

from snn_research.backends import HardwareConstraints, check_constraints, export_network, save_export
from snn_research.records import RunRecord, save_run
from snn_research.reporting import load_records, write_report
from snn_research.utils import load_config


def main(argv: list[str] | None = None) -> list[Path]:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--rule", choices=(*RULES, "all"), help="override the configured rule; 'all' runs each")
    parser.add_argument("--report", action="store_true", help="write a comparison report for these runs")
    parser.add_argument("--export", action="store_true",
                        help="write export.json + weights.npz (model + rule description) into each run directory")
    parser.add_argument("--hardware", type=Path,
                        help="hardware-constraints YAML to check exports against (implies --export)")
    args = parser.parse_args(argv)
    hardware = load_config(HardwareConstraints, args.hardware) if args.hardware else None

    base = load_config(ReferenceConfig, args.config)
    rules = RULES if args.rule == "all" else [args.rule or base.rule]
    run_dirs = []
    for rule in rules:
        config = replace(base, rule=rule)  # saved in the resolved config
        for seed in config.seeds:
            record, model, rules_used = train_and_record(config, seed)
            run_dir = save_run(record, args.results)
            run_dirs.append(run_dir)
            report_line(config.rule, seed, record, run_dir)
            if args.export or hardware:
                export_run(config, model, rules_used, run_dir, hardware)
    if args.report:
        report_dir = write_report(load_records(run_dirs), args.results, REPORT_TABLES)
        print(f"report -> {report_dir / 'report.md'}")
    return run_dirs


def export_run(config: ReferenceConfig, model: SpikingNet, rules: list, run_dir: Path,
               hardware: HardwareConstraints | None) -> None:
    """Export the trained model + rules; optionally check them against hardware limits."""
    description, weights = export_network(model, rules, config.encoding, input_shape(config))
    if hardware:
        findings = check_constraints(description, hardware)
        description["constraint_check"] = {"hardware": hardware.name, "status": hardware.status,
                                           "findings": [{"level": lv, "message": m} for lv, m in findings]}
        counts = {lv: sum(f[0] == lv for f in findings) for lv in ("error", "warning")}
        print(f"  export + {hardware.name} check: {counts['error']} errors, {counts['warning']} warnings")
    save_export(description, weights, run_dir)


def report_line(rule: str, seed: int, record: RunRecord, run_dir: Path) -> None:
    m = record.metrics
    stream = "/".join(f"{m[f'stream/shift0/quality_{p}']:.3f}" for p in ("before", "during", "after"))
    print(f"{rule} seed {seed}: test {m['pretrain/test/accuracy']:.3f} | stream {stream} | "
          f"recovery {m['stream/shift0/recovery_samples']} | {run_dir}")


if __name__ == "__main__":
    main()
