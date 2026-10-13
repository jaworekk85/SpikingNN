"""MNIST / Fashion-MNIST from the original IDX files (no torchvision dependency).

Files are downloaded once into ``root`` (default ``data/``, ignored by Git), checked
against published MD5 sums, and parsed with numpy. Optional seeded subsampling keeps
CPU runs small; the version string records variant, sizes and subsampling seed.
Tests never call this module's download path.
"""

from __future__ import annotations

import gzip
import hashlib
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .synthetic import Split

SOURCES = {
    "mnist": ("https://ossci-datasets.s3.amazonaws.com/mnist/", {
        "train-images-idx3-ubyte.gz": "f68b3c2dcbeaaa9fbdd348bbdeb94873",
        "train-labels-idx1-ubyte.gz": "d53e105ee54ea40749a09fcbcd1e9432",
        "t10k-images-idx3-ubyte.gz": "9fb629c4189551a2d022fa330f9573f3",
        "t10k-labels-idx1-ubyte.gz": "ec29112dd5afa0611ce80d1b7f02629c",
    }),
    "fashion_mnist": ("http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/", {
        "train-images-idx3-ubyte.gz": "8d4fb7e6c68d591d4c3dfef9ec88bf0d",
        "train-labels-idx1-ubyte.gz": "25c81989df183df01b3e8a0aad5dffbe",
        "t10k-images-idx3-ubyte.gz": "bef4ecab320f06d8554ea6380940ec79",
        "t10k-labels-idx1-ubyte.gz": "bb300cfdad3c16e7a12a480ee83cd310",
    }),
}


@dataclass(frozen=True)
class MNISTConfig:
    variant: str = "mnist"  # mnist | fashion_mnist
    root: str = "data"
    n_train: int = 0  # 0 = all 60,000
    n_test: int = 0  # 0 = all 10,000
    seed: int = 0  # subsampling seed
    flatten: bool = True  # [784] instead of [1, 28, 28]

    def __post_init__(self) -> None:
        if self.variant not in SOURCES:
            raise ValueError(f"variant must be one of {tuple(SOURCES)}")
        if self.n_train < 0 or self.n_test < 0:
            raise ValueError("n_train and n_test must be non-negative")

    @property
    def n_classes(self) -> int:
        return 10

    @property
    def shape(self) -> tuple[int, ...]:
        return (784,) if self.flatten else (1, 28, 28)


def _md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def fetch(variant: str, root: str | Path) -> Path:
    """Download (if needed) and verify the four IDX files; return their directory."""
    base_url, files = SOURCES[variant]
    directory = Path(root) / variant
    directory.mkdir(parents=True, exist_ok=True)
    for name, md5 in files.items():
        path = directory / name
        if not path.exists():
            tmp = path.with_suffix(".part")
            urllib.request.urlretrieve(base_url + name, tmp)
            tmp.replace(path)
        if _md5(path) != md5:
            raise ValueError(f"MD5 mismatch for {path}; delete it and retry")
    return directory


def read_idx(path: Path) -> np.ndarray:
    """Parse a gzipped IDX file (unsigned-byte data)."""
    with gzip.open(path, "rb") as f:
        data = f.read()
    if data[2] != 0x08:
        raise ValueError(f"{path}: expected unsigned-byte IDX data")
    ndim = data[3]
    shape = tuple(int.from_bytes(data[4 + 4 * i : 8 + 4 * i], "big") for i in range(ndim))
    return np.frombuffer(data, dtype=np.uint8, offset=4 + 4 * ndim).reshape(shape)


def _subsample(n_available: int, n: int, generator: torch.Generator) -> torch.Tensor:
    return torch.randperm(n_available, generator=generator)[:n] if n else torch.arange(n_available)


def load_mnist(config: MNISTConfig) -> tuple[Split, Split]:
    """Train and test splits with intensities in ``[0, 1]``."""
    directory = fetch(config.variant, config.root)
    g = torch.Generator().manual_seed(config.seed)

    def split(prefix: str, n: int, role: str) -> Split:
        images = torch.from_numpy(read_idx(directory / f"{prefix}-images-idx3-ubyte.gz").copy())
        labels = torch.from_numpy(read_idx(directory / f"{prefix}-labels-idx1-ubyte.gz").astype(np.int64))
        idx = _subsample(len(labels), n, g)
        x = images[idx].float().div_(255).reshape(len(idx), *config.shape)
        version = f"{config.variant}-idx-{role}{len(idx)}-seed{config.seed}"
        return Split(x, labels[idx], role, version)

    return split("train", config.n_train, "train"), split("t10k", config.n_test, "test")
