"""Declared, swappable learning rules."""

from .declaration import LOCALITIES, MODULATOR_SOURCES, RuleDeclaration
from .surrogate_bptt import SurrogateBPTT, SurrogateBPTTConfig

__all__ = [
    "LOCALITIES",
    "MODULATOR_SOURCES",
    "RuleDeclaration",
    "SurrogateBPTT",
    "SurrogateBPTTConfig",
]
