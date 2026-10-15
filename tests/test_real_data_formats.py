"""File-format, encoding and configuration checks for MNIST and N-MNIST, with tiny
hand-made files: nothing is downloaded."""

import gzip
import zipfile
from dataclasses import replace
from pathlib import Path

import numpy as np
import pipeline
import pytest
import torch

from snn_research.datasets import MNISTConfig, Split
from snn_research.datasets.mnist import fetch as fetch_mnist
from snn_research.datasets.mnist import read_idx
from snn_research.datasets.nmnist import NMNISTConfig, bin_events, decode_events, read_split
from snn_research.encoding import EncodingConfig, encode, events_encode
from snn_research.utils import load_config

ROOT = Path(__file__).resolve().parents[1]


def write_idx(path: Path, array: np.ndarray) -> None:
    header = bytes([0, 0, 0x08, array.ndim]) + b"".join(int(d).to_bytes(4, "big") for d in array.shape)
    with gzip.open(path, "wb") as f:
        f.write(header + array.astype(np.uint8).tobytes())


def test_idx_reader_round_trip(tmp_path):
    images = np.arange(2 * 3 * 4).reshape(2, 3, 4)
    write_idx(tmp_path / "x.gz", images)
    np.testing.assert_array_equal(read_idx(tmp_path / "x.gz"), images)


def test_mnist_rejects_files_with_wrong_checksum(tmp_path):
    directory = tmp_path / "mnist"
    directory.mkdir()
    for name in ("train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz",
                 "t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz"):
        write_idx(directory / name, np.zeros((1, 1)))
    with pytest.raises(ValueError, match="MD5 mismatch"):
        fetch_mnist("mnist", tmp_path)  # files exist, so nothing is downloaded
    assert MNISTConfig(flatten=False).shape == (1, 28, 28)
    with pytest.raises(ValueError):
        MNISTConfig(variant="kmnist")


def event_bytes(events: list[tuple[int, int, int, int]]) -> bytes:
    """Encode (x, y, p, t_us) as N-MNIST 5-byte events; y=240 rows are overflow markers."""
    out = bytearray()
    for x, y, p, t in events:
        out += bytes([x, y, (p << 7) | ((t >> 16) & 127), (t >> 8) & 255, t & 255])
    return bytes(out)


def test_decode_events_handles_polarity_timestamps_and_overflow():
    raw = event_bytes([(1, 2, 1, 5), (33, 0, 0, 8191), (0, 240, 0, 0), (4, 5, 1, 10)])
    ev = decode_events(raw)
    assert ev["x"].tolist() == [1, 33, 4] and ev["y"].tolist() == [2, 0, 5] and ev["p"].tolist() == [1, 0, 1]
    assert ev["t"].tolist() == [5, 8191, 10 + 8192]  # the marker adds 2**13 to later events


def test_bin_events_makes_binary_frames_within_the_window():
    ev = decode_events(event_bytes([(1, 2, 1, 0), (1, 2, 1, 50), (3, 4, 0, 15_000), (3, 4, 0, 40_000)]))
    frames = bin_events(ev, T=3, window_ms=30.0)  # 10 ms bins; the 40 ms event is outside
    assert frames.shape == (3, 2, 34, 34) and frames.dtype == np.uint8
    assert frames[0, 1, 2, 1] == 1 and frames[1, 0, 4, 3] == 1 and frames.sum() == 2


def test_read_split_from_zip_without_extracting(tmp_path):
    archive = tmp_path / "train.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("Train/3/00001.bin", event_bytes([(0, 0, 0, 0)]))
        zf.writestr("Train/7/00002.bin", event_bytes([(5, 5, 1, 2_000)]))
    x, y = read_split(archive, "Train", n=0, seed=0, T=2, window_ms=4.0)
    assert x.shape == (2, 2, 2, 34, 34) and sorted(y.tolist()) == [3, 7]
    assert not (tmp_path / "Train").exists()
    assert NMNISTConfig(T=30, window_ms=300).bin_ms == 10.0 and NMNISTConfig().shape == (2312,)


def test_events_encoding_passes_binary_frames_through_time_first():
    frames = (torch.rand(4, 6, 10) < 0.2).float()  # [batch, T, features]
    enc = events_encode(frames, torch.Generator().manual_seed(0))
    assert enc.is_spikes and enc.values.shape == (6, 4, 10)
    assert torch.equal(enc.values, frames.transpose(0, 1))
    with pytest.raises(ValueError, match="time bins"):
        encode(EncodingConfig("events", T=5), frames, torch.Generator())


def test_compact_uint8_splits_yield_float_batches():
    split = Split(torch.ones(3, 2, 4, dtype=torch.uint8), torch.tensor([0, 1, 2]), "train", "v")
    x, _ = next(split.batches(2))
    assert x.dtype == torch.float32


def test_reference_config_ties_event_data_to_encoding_and_timestep():
    config = load_config(pipeline.ReferenceConfig, ROOT / "configs" / "experiments" / "reference_nmnist.yaml")
    assert config.neuron.dt_ms == config.nmnist.bin_ms and config.encoding.T == config.nmnist.T
    with pytest.raises(ValueError, match="dt_ms"):
        replace(config, neuron=replace(config.neuron, dt_ms=1.0))
    with pytest.raises(ValueError, match="events"):
        replace(config, encoding=EncodingConfig("rate", T=30))
