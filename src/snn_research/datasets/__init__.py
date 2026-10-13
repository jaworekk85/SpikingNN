"""Split-aware datasets and streams."""

from .mnist import MNISTConfig, load_mnist
from .nmnist import NMNISTConfig, load_nmnist
from .stream import Stream
from .synthetic import ROLES, Split, SyntheticConfig, synthetic_prototypes

__all__ = [
    "MNISTConfig",
    "NMNISTConfig",
    "ROLES",
    "Split",
    "Stream",
    "SyntheticConfig",
    "load_mnist",
    "load_nmnist",
    "synthetic_prototypes",
]
