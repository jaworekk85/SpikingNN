"""Task/domain sequence: train tasks in order, evaluating every task after each one."""

from __future__ import annotations

from typing import Sequence

import torch

from ..datasets import Split
from ..encoding import EncodingConfig
from ..metrics import forgetting
from .offline import OfflineConfig, evaluate, train_epochs


def run_task_sequence(model, rules: Sequence, tasks: Sequence[tuple[Split, Split]], encoding: EncodingConfig,
                      config: OfflineConfig, generator: torch.Generator) -> dict:
    """``accuracy[i][a]``: test accuracy on task ``a`` after training task ``i``.

    Every task is evaluated at every checkpoint; forgetting of task ``a`` uses only
    rows from ``a`` on (after it was learned).
    """
    if any(test.role != "test" for _, test in tasks):
        raise ValueError("task evaluation requires 'test' splits")
    matrix: list[list[float]] = []
    for train, _ in tasks:
        train_epochs(model, rules, train, encoding, config, generator)
        matrix.append([evaluate(model, test, encoding, config.batch_size, generator)["accuracy"]
                       for _, test in tasks])
    metrics = {f"task{a}/forgetting": forgetting([row[a] for row in matrix[a:]]) for a in range(len(tasks))}
    metrics.update({f"task{a}/final_accuracy": matrix[-1][a] for a in range(len(tasks))})
    return {"accuracy_matrix": matrix, "metrics": metrics}
