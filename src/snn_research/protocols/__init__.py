"""Training and evaluation protocols: offline, prequential and task sequence."""

from .offline import OfflineConfig, evaluate, forward_and_learn, run_offline, train_epochs
from .prequential import PrequentialConfig, run_prequential
from .task_sequence import run_task_sequence

__all__ = [
    "OfflineConfig",
    "PrequentialConfig",
    "evaluate",
    "forward_and_learn",
    "run_offline",
    "run_prequential",
    "run_task_sequence",
    "train_epochs",
]
