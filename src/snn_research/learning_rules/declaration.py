"""Declared learning-rule properties, saved with every run."""

from __future__ import annotations

from dataclasses import dataclass

LOCALITIES = ("synapse_local", "layer_local", "global_modulated", "global_gradient")
MODULATOR_SOURCES = ("none", "reward", "error", "prediction_error", "learned")


@dataclass(frozen=True)
class RuleDeclaration:
    name: str
    uses_autograd: bool
    locality: str
    online_capable: bool
    state: tuple[str, ...]
    modulator_source: str

    def __post_init__(self) -> None:
        if self.locality not in LOCALITIES:
            raise ValueError(f"locality must be one of {LOCALITIES}, got {self.locality!r}")
        if self.modulator_source not in MODULATOR_SOURCES:
            raise ValueError(f"modulator_source must be one of {MODULATOR_SOURCES}")
        if self.uses_autograd and self.locality != "global_gradient":
            raise ValueError("a rule using autograd must declare locality 'global_gradient'")
