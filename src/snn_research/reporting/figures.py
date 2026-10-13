"""Reference figure: stream quality around the shift, and quality vs synaptic operations.

Uses matplotlib's object-oriented API (no pyplot, no GUI backend). Colors follow a
colorblind-validated categorical order (first three slots validated for all pairs);
every series is also named in a legend and directly labelled, so identity never
relies on color alone.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator, StrMethodFormatter

from .compare import group_label

SERIES = ("#2a78d6", "#eb6834", "#1baf7a")  # blue, orange, aqua: validated all-pairs
SURFACE, TEXT, TEXT_2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SHIFT_FILL = "#e1e0d9"


def trailing_mean(values: Sequence[float], window: int) -> list[float]:
    out, total = [], 0.0
    for i, v in enumerate(values):
        total += v - (values[i - window] if i >= window else 0.0)
        out.append(total / min(i + 1, window))
    return out


def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=TEXT_2, labelsize=9)
    ax.xaxis.label.set_color(TEXT_2)
    ax.yaxis.label.set_color(TEXT_2)
    ax.title.set_color(TEXT)


def reference_figure(records: Sequence[dict[str, Any]], path: str | Path) -> Path:
    """Two panels. Left: prequential accuracy over the stream (trailing window, mean
    over seeds) with the shift window shaded. Right: test accuracy vs synaptic
    operations per test sample after pretraining (one dot per seed, ringed mean)."""
    labels = sorted({group_label(r) for r in records})
    if len(labels) > len(SERIES):
        raise ValueError(f"at most {len(SERIES)} conditions per figure; facet or fold the rest")
    colors = dict(zip(labels, SERIES))
    stream_cfg, schedule = records[0]["config"]["stream"], records[0]["shift"]["schedule"]
    window = stream_cfg["window"]

    fig = Figure(figsize=(11, 4.6), dpi=150, facecolor=SURFACE)
    left, right = fig.subplots(1, 2, gridspec_kw={"width_ratios": [1.5, 1]})
    for ax in (left, right):
        _style(ax)

    left.axvspan(schedule["start"], schedule["end"], color=SHIFT_FILL, alpha=0.6, linewidth=0)
    left.text((schedule["start"] + schedule["end"]) / 2, 1.02, f"shift window ({records[0]['shift']['kind']})",
              ha="center", va="bottom", fontsize=8, color=TEXT_2)
    for label in labels:
        runs = [r for r in records if group_label(r) == label]
        curves = [trailing_mean(r["traces"]["stream/correct"], window) for r in runs]
        mean = [sum(c[i] for c in curves) / len(curves) for i in range(min(map(len, curves)))]
        xs = range(window - 1, len(mean))  # full windows only
        left.plot(xs, mean[window - 1 :], color=colors[label], linewidth=2, label=f"{label} (n={len(runs)})")
        left.annotate(label, (len(mean) - 1, mean[-1]), xytext=(6, 0), textcoords="offset points",
                      va="center", fontsize=8, color=TEXT)
    length = stream_cfg["length"]
    left.set_ylim(0, 1.08)
    left.set_xlim(0, length * 1.18)  # room for direct labels
    left.set_xticks(sorted({0, schedule["start"], schedule["end"], length}))
    left.set_xlabel("stream sample")
    left.set_ylabel(f"prequential accuracy (trailing {window})")
    left.set_title("Quality before, during and after the shift", fontsize=10, loc="left")

    for label in labels:
        runs = [r for r in records if group_label(r) == label]
        x = [r["metrics"]["pretrain/test/synops_per_sample"] for r in runs]
        y = [r["metrics"]["pretrain/test/accuracy"] for r in runs]
        right.scatter(x, y, s=28, color=colors[label], alpha=0.55, linewidths=0)
        mx, my = sum(x) / len(x), sum(y) / len(y)
        right.scatter([mx], [my], s=90, color=colors[label], edgecolors=SURFACE, linewidths=2, zorder=3)
        right.annotate(label, (mx, my), xytext=(7, 5), textcoords="offset points", fontsize=8, color=TEXT)
    right.xaxis.set_major_locator(MaxNLocator(nbins=4))
    right.xaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    right.set_xlabel("synaptic operations per test sample")
    right.set_ylabel("test accuracy after pretraining")
    right.set_title("Quality vs activity (dots: seeds)", fontsize=10, loc="left")

    legend = left.legend(loc="upper left", bbox_to_anchor=(0, -0.16), ncol=len(labels), fontsize=8, frameon=False)
    for text in legend.get_texts():
        text.set_color(TEXT)
    fig.text(0.01, 0.01, "Simulation only. Synaptic operations are logical counts, not measured energy.",
             fontsize=7, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    path = Path(path)
    fig.savefig(path, facecolor=SURFACE)
    return path
