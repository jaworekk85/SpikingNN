"""Split-aware datasets and streams."""

from .stream import Stream
from .synthetic import ROLES, Split, SyntheticConfig, synthetic_prototypes

__all__ = ["ROLES", "Split", "Stream", "SyntheticConfig", "synthetic_prototypes"]
