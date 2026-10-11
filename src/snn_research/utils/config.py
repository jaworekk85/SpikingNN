"""Build nested frozen dataclasses from YAML, rejecting unknown keys."""

from __future__ import annotations

import dataclasses
import typing
from pathlib import Path
from typing import Any, TypeVar

import yaml

T = TypeVar("T")


def from_dict(cls: type[T], data: dict[str, Any] | None) -> T:
    data = data or {}
    hints = typing.get_type_hints(cls)
    names = {f.name for f in dataclasses.fields(cls)}
    unknown = set(data) - names
    if unknown:
        raise ValueError(f"unknown keys for {cls.__name__}: {sorted(unknown)}")
    kwargs = {}
    for name, value in data.items():
        hint = hints[name]
        if dataclasses.is_dataclass(hint):
            value = from_dict(hint, value)
        elif typing.get_origin(hint) is tuple:
            value = tuple(value)
        kwargs[name] = value
    return cls(**kwargs)


def load_config(cls: type[T], path: str | Path) -> T:
    return from_dict(cls, yaml.safe_load(Path(path).read_text(encoding="utf-8")))


def resolved(config: Any) -> dict[str, Any]:
    """Full config including defaults, as plain JSON/YAML-compatible data."""
    return dataclasses.asdict(config)
