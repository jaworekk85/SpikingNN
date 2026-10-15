"""Hyperparameter search that can only see ``train`` and ``tune`` data.

``grid_search`` refuses any split whose role is not exactly ``train`` (for fitting) and
``tune`` (for scoring), so test data cannot influence hyperparameters. The scoring
function receives only those two splits.
"""

from __future__ import annotations

import itertools
from typing import Any, Callable, Sequence

from ..datasets import Split

Score = Callable[[dict[str, Any], int, Split, Split], float]


def grid(space: dict[str, Sequence[Any]]) -> list[dict[str, Any]]:
    """Cartesian product of ``{name: values}`` in a fixed, saved order."""
    names = list(space)
    return [dict(zip(names, values)) for values in itertools.product(*(space[n] for n in names))]


def grid_search(space: dict[str, Sequence[Any]], score: Score, train: Split, tune: Split,
                seeds: Sequence[int]) -> dict[str, Any]:
    """Score every candidate on every seed; the best has the highest mean score."""
    if train.role != "train" or tune.role != "tune":
        raise ValueError(f"tuning needs a 'train' and a 'tune' split, got '{train.role}' and '{tune.role}' "
                         "(test data must never be used for tuning)")
    if not seeds:
        raise ValueError("at least one tuning seed is required")
    candidates = []
    for params in grid(space):
        scores = [float(score(params, seed, train, tune)) for seed in seeds]
        candidates.append({"params": params, "scores": scores, "mean": sum(scores) / len(scores)})
    best = max(range(len(candidates)), key=lambda i: candidates[i]["mean"])  # first wins ties
    return {"candidates": candidates, "best": candidates[best], "seeds": list(seeds)}
