"""Comparable-run validation, multi-seed aggregation, tables and figures."""

from .compare import (
    aggregate,
    check_comparable,
    comparison_signature,
    group_label,
    load_records,
    metrics_table,
    summarize,
)
from .figures import reference_figure, trailing_mean
from .report import write_report

__all__ = [
    "aggregate",
    "check_comparable",
    "comparison_signature",
    "group_label",
    "load_records",
    "metrics_table",
    "reference_figure",
    "summarize",
    "trailing_mean",
    "write_report",
]
