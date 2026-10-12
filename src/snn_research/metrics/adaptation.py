"""Shift-phase quality, recovery and forgetting (definitions in docs/metrics.md).

``correct`` is a per-sample 0/1 trace indexed by stream position. Unavailable values
(an empty phase, no recovery) are ``None``, never a silent number.
"""

from __future__ import annotations

from typing import Sequence

from ..shifts import Schedule


def phase_quality(correct: Sequence[float], schedule: Schedule) -> dict[str, float | None]:
    """Mean quality over the ``before``, ``during`` and ``after`` phases of a shift."""
    groups: dict[str, list[float]] = {"before": [], "during": [], "after": []}
    for i, c in enumerate(correct):
        groups[schedule.phase_at(i)].append(c)
    return {phase: (sum(v) / len(v) if v else None) for phase, v in groups.items()}


def recovery_samples(correct: Sequence[float], schedule: Schedule, q_target: float, window: int) -> int | None:
    """Samples after shift onset until the trailing ``window`` quality first reaches ``q_target``.

    Only windows fully after onset count, so pre-shift samples cannot mask the drop.
    """
    for i in range(schedule.start + window - 1, len(correct)):
        if sum(correct[i - window + 1 : i + 1]) / window >= q_target:
            return i + 1 - schedule.start
    return None


def forgetting(history: Sequence[float]) -> float:
    """``Q_max - Q_final`` over a task's quality history, from when it was first learned."""
    return max(history) - history[-1]
