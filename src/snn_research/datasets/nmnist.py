"""N-MNIST event data (Orchard et al. 2015), binned into spike frames.

The archives (``train.zip``, ``test.zip``; Mendeley dataset 468j46mzdv, MD5-checked)
are downloaded once into ``<root>/nmnist`` (ignored by Git) and read directly from the
zip files; nothing is extracted. Each ``.bin`` file holds 5-byte events: x, y,
polarity (bit 7 of byte 2) and a 23-bit microsecond timestamp; events with
``y == 240`` are overflow markers adding 2**13 to later timestamps (as in Tonic's
reader).

Events in ``[0, window_ms)`` are binned into ``T`` frames of ``window_ms / T`` ms each,
with two polarity channels on the 34x34 sensor; a frame element is 1 if at least one
event fell into it. Samples are stored as ``uint8`` ``[n, T, 2, 34, 34]`` (or flattened
to ``[n, T, 2312]``) and cached as a ``.pt`` file keyed by the configuration.
"""

from __future__ import annotations

import hashlib
import json
import urllib.request
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from .synthetic import Split

BASE_URL = "https://data.mendeley.com/public-files/datasets/468j46mzdv/files/"
ARCHIVES = {
    "train": ("39c25547-014b-4137-a934-9d29fa53c7a0/file_downloaded", "train.zip", "20959b8e626244a1b502305a9e6e2031", "Train"),
    "test": ("05a4d654-7e03-4c15-bdfa-9bb2bcbea494/file_downloaded", "test.zip", "69ca8762b2fe404d9b9bad1103e97832", "Test"),
}
SENSOR = (2, 34, 34)  # polarity, y, x


@dataclass(frozen=True)
class NMNISTConfig:
    root: str = "data"
    n_train: int = 6000  # 0 = all 60,000
    n_test: int = 1000  # 0 = all 10,000
    seed: int = 0
    T: int = 30
    window_ms: float = 300.0  # all three saccades; 100 ms covers the first one
    flatten: bool = True

    def __post_init__(self) -> None:
        if self.n_train < 0 or self.n_test < 0 or self.T <= 0 or self.window_ms <= 0:
            raise ValueError("invalid N-MNIST configuration")

    @property
    def n_classes(self) -> int:
        return 10

    @property
    def bin_ms(self) -> float:
        return self.window_ms / self.T

    @property
    def shape(self) -> tuple[int, ...]:
        """Per-timestep sample shape."""
        return (int(np.prod(SENSOR)),) if self.flatten else SENSOR


def decode_events(raw: bytes) -> dict[str, np.ndarray]:
    """Decode N-MNIST 5-byte events into ``x, y, p`` (0/1) and ``t`` (microseconds)."""
    data = np.frombuffer(raw, dtype=np.uint8).astype(np.uint32)
    x, y, b2, b3, b4 = (data[i::5] for i in range(5))
    p = (b2 & 128) >> 7
    t = ((b2 & 127) << 16) | (b3 << 8) | b4
    overflow = np.flatnonzero(y == 240)
    for i in overflow:
        t[i:] += 1 << 13
    keep = y != 240
    return {"x": x[keep], "y": y[keep], "p": p[keep], "t": t[keep]}


def bin_events(events: dict[str, np.ndarray], T: int, window_ms: float) -> np.ndarray:
    """Binary frames ``uint8 [T, 2, 34, 34]`` for events with ``t < window_ms``."""
    frames = np.zeros((T, *SENSOR), dtype=np.uint8)
    t_bin = (events["t"] / (window_ms * 1000.0) * T).astype(np.int64)
    keep = (t_bin < T) & (events["x"] < SENSOR[2]) & (events["y"] < SENSOR[1])
    frames[t_bin[keep], events["p"][keep], events["y"][keep], events["x"][keep]] = 1
    return frames


def fetch(root: str | Path) -> Path:
    directory = Path(root) / "nmnist"
    directory.mkdir(parents=True, exist_ok=True)
    for url, name, md5, _ in ARCHIVES.values():
        path = directory / name
        if not path.exists():
            tmp = path.with_suffix(".part")
            urllib.request.urlretrieve(BASE_URL + url, tmp)
            tmp.replace(path)
        digest = hashlib.md5()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1 << 24), b""):
                digest.update(chunk)
        if digest.hexdigest() != md5:
            raise ValueError(f"MD5 mismatch for {path}; delete it and retry")
    return directory


def read_split(archive: Path, folder: str, n: int, seed: int, T: int, window_ms: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Seeded subsample of ``n`` samples (0 = all) from a zip archive, binned."""
    with zipfile.ZipFile(archive) as zf:
        names = sorted(name for name in zf.namelist() if name.endswith(".bin") and name.startswith(folder + "/"))
        order = torch.randperm(len(names), generator=torch.Generator().manual_seed(seed))
        chosen = [names[i] for i in (order[:n] if n else order)]
        frames = np.stack([bin_events(decode_events(zf.read(name)), T, window_ms) for name in chosen])
    labels = torch.tensor([int(name.split("/")[1]) for name in chosen])
    return torch.from_numpy(frames), labels


def load_nmnist(config: NMNISTConfig) -> tuple[Split, Split]:
    """Train and test splits of binned ``uint8`` spike frames ``[n, T, ...]``."""
    key = hashlib.sha256(json.dumps(asdict(config), sort_keys=True).encode()).hexdigest()[:12]
    cache = Path(config.root) / "nmnist" / f"binned_{key}.pt"
    if cache.exists():  # built from MD5-verified archives
        data = torch.load(cache)
    else:
        directory = fetch(config.root)
        data = {}
        for role, n in (("train", config.n_train), ("test", config.n_test)):
            _, name, _, folder = ARCHIVES[role]
            x, y = read_split(directory / name, folder, n, config.seed, config.T, config.window_ms)
            data[role] = (x, y)
        torch.save(data, cache)
    splits = []
    for role in ("train", "test"):
        x, y = data[role]
        x = x.reshape(len(y), config.T, *config.shape)
        version = (f"nmnist-{role}{len(y)}-seed{config.seed}-T{config.T}-window{config.window_ms:g}ms")
        splits.append(Split(x, y, role, version))
    return splits[0], splits[1]
