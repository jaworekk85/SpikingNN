# Reference pipeline validation

Phase 1 subset (`run.py`, config `configs/experiments/reference_tiny.yaml`):
seeded synthetic prototypes -> rate encoding -> LIF MLP -> surrogate BPTT ->
offline test evaluation -> one `RunRecord` per seed. MNIST, the local rule, the shift
and the figure/table arrive with later phases.

The complete planned reference uses a small LIF MLP, rate-encoded MNIST
or a tiny event-data subset, surrogate BPTT and one local/three-factor rule, a
seeded shift and pre/during/post evaluation. It reports accuracy, forgetting,
recovery, spike rate and synaptic operations in a figure and metrics table.

Implement in steps as Phases 1–4 supply the required components. Tests use synthetic
data without downloads. Its smallest complete configuration must run on laptop CPU
in a few minutes. Pipeline validation numbers are not research results.
