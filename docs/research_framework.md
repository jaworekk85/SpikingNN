# Research framework architecture

Current architecture of `snn_research` (Phases 0–5 implemented). The authoritative
requirements are in `SNN_AGENT_RESEARCH_SPEC.md`; details live in
[`learning_rules.md`](learning_rules.md), [`metrics.md`](metrics.md) and
[`neuromorphic_hardware.md`](neuromorphic_hardware.md).

## Dependencies

Python 3.11+, PyTorch (CPU by default), numpy, PyYAML, matplotlib (object-oriented
API, no GUI backend); pytest for tests. Datasets are read from their original files
with numpy (no torchvision). Not used: snnTorch (the reference LIF is our own; snnTorch
stays an optional adapter if ever needed), Hydra, registries, Brian2/GeNN, hardware SDKs.

## Modules

| Module | Main entry points |
| --- | --- |
| `neurons/` | `LIF`, `LIFParams`, `LIFState`: exponential-Euler LIF, subtract/zero reset, refractory steps, surrogate spike |
| `models/` | `build_model` (MLP or small conv) -> `SpikingNet`; per-layer `LayerObservation(pre, spikes, v)` |
| `encoding/` | `encode`: rate, latency, direct current, binned `events` |
| `datasets/` | `Split` (roles `train`/`tune`/`test`, `split_off_tune`, `subset`), `Stream`, synthetic, MNIST/Fashion-MNIST, N-MNIST |
| `shifts/` | `Schedule`, `ShiftConfig`, `shift_and_encode` (sample shifts -> encoding -> temporal shifts) |
| `learning_rules/` | `LearningRule` protocol, `RuleDeclaration`, `SurrogateBPTT`, `STDP`, `ThreeFactor`, fixed-point `constrain_` |
| `protocols/` | `run_offline`, `run_prequential`, `run_task_sequence`, `grid_search` (train/tune only) |
| `metrics/` | accuracy, spike rate, sparsity, phase quality, recovery, forgetting, synaptic operations |
| `energy/` | `EnergyProxyConfig`, `energy_proxy_pj` (proxy with saved assumptions) |
| `records/` | `RunRecord` (validated), `save_run` (immutable run directory) |
| `reporting/` | `load_records`, `check_comparable`, `aggregate`, `write_report` (tables + figure) |
| `backends/` | `export_network`, `HardwareBackend` protocol, `Measurement`, `HardwareConstraints` checks |
| `utils/` | YAML -> frozen dataclasses (`load_config`, `override`), seeding, provenance |

`experiments/reference/` wires these together (`pipeline.py`, `run.py`, `report.py`,
`tune.py`); it owns no scientific primitives. `configs/experiments/` holds experiment
configs listing only non-default values; `configs/hardware/` holds hardware drafts.

## Data flow

```text
config + seed -> data split / Stream -> shifts -> encoder -> SpikingNet (LIF)
                                                         |  observations
                     protocol (order) <- rules (+ loss / error modulators)
                                                         |
              metrics, synaptic ops, energy proxy -> RunRecord -> run dir -> report
                                         model + rules -> export -> constraint check
```

Protocols own ordering; models expose observations; rules decide updates and are
bound once by the caller (`init_state`), so rule state persists across protocol stages.
Rules never import datasets or protocols; shifts never train; metrics never modify.

## Conventions

- Tensors `[T, batch, ...]`; `dt` in milliseconds; neuron state is fresh per sample.
- Configs are YAML -> nested frozen dataclasses: unknown keys and invalid values are
  rejected, and the full resolved config is saved with every run.
- Local rule kernels receive only `LocalObservation(pre, post, post_potential)`.
- Counts are logical events (spikes, synaptic operations, MACs), not GPU work.

## Spec invariants and where they are enforced

| Invariant (spec §10) | Enforcement | Tests |
| --- | --- | --- |
| Immutable results | `save_run`, `write_report`, `save_export`, `tune.py` create new dirs/files with `exist_ok=False` | `test_records`, `test_reporting`, `test_backends` |
| Locality verified | declarations validated; local rules run under `no_grad`, kernels see only `LocalObservation` | `test_learning_rules` (no graph, no `.grad`, kernel signatures) |
| Proxy vs measured energy | `RunRecord`: `energy_proxy` needs assumptions; `energy_j` only with a hardware `Measurement` | `test_records`, `test_backends` |
| No tuning on test | `grid_search` accepts only `train`/`tune` splits; `Stream` needs `train` | `test_tuning`, `test_shifts_and_protocols` |
| Comparable settings | `check_comparable` refuses differing `T`, encoding, model size, protocol unless `allow_mismatch` | `test_reporting` |
| Pre/during/post shift reported | `run_prequential` always reports all three phases per shift; forgetting/adaptation separate | `test_shifts_and_protocols`, smoke test |
| Plausibility wording | docs describe rules by declared, tested properties | review |
