# Reference pipeline validation

Current subset (`run.py`, config `configs/experiments/reference_tiny.yaml`):
seeded synthetic prototypes -> rate encoding -> LIF MLP -> learning rule ->
offline pretraining -> prequential stream with one seeded, scheduled shift (pre-shift
-> shift -> post-shift, online adaptation) -> clean and shifted test accuracy ->
synaptic operations and an energy proxy with explicit assumptions -> one `RunRecord`
per seed. The rule is chosen by `rule:` in the config or
`--rule surrogate_bptt | three_factor | stdp | all`. `--report` (or `report.py` on saved
runs) writes a multi-seed comparison into a new `report_<UTC>_<hash>/` directory:
`report.md` (quality/adaptation, activity/inference cost and adaptation cost tables,
mean ± sample std over seeds, energy-proxy assumptions), `figure.png` (stream quality
around the shift; test accuracy vs synaptic operations) and `summary.json`. Runs with
different `T`, encoding, model size or protocol are refused unless `--allow-mismatch`.
Datasets: `dataset: synthetic | mnist | nmnist` (`reference_mnist.yaml`,
`reference_nmnist.yaml`; downloads are MD5-checked and cached in `data/`). For N-MNIST
the encoding is `events` with `T` equal to the number of bins and `neuron.dt_ms` equal
to the bin width; the config refuses anything else.

The complete planned reference uses a small LIF MLP, rate-encoded MNIST
or a tiny event-data subset, surrogate BPTT and one local/three-factor rule, a
seeded shift and pre/during/post evaluation. It reports accuracy, forgetting,
recovery, spike rate and synaptic operations in a figure and metrics table.

Implement in steps as Phases 1–4 supply the required components. Tests use synthetic
data without downloads. Its smallest complete configuration must run on laptop CPU
in a few minutes. Pipeline validation numbers are not research results.
