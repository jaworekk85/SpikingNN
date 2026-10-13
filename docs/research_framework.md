# Research framework architecture — Phase 0 proposal

This proposal follows `SNN_AGENT_RESEARCH_SPEC.md`. Phase 0 creates documentation
and an installable package skeleton only. Interfaces below are proposed contracts,
not implemented functionality. Research directions are design context; they do not
authorize experiments. Stop for review before Phase 1.

## Dependency plan

| Dependency | Role | Introduction |
| --- | --- | --- |
| Python 3.11+ | Standard-library dataclasses, typing, pathlib, JSON | Phase 0 |
| setuptools + wheel | Package build with a `src/` layout | Phase 0 |
| PyTorch | Reference tensors, LIF dynamics, models and training | Phase 1 |
| PyYAML | Load YAML into explicitly validated dataclasses | Phase 1 |
| pytest | Tiny CPU dynamics, invariants and smoke tests | Phase 1 |
| (no torchvision) | MNIST/Fashion-MNIST read from original IDX files, N-MNIST from the original zips with numpy; MD5-checked cached downloads | Real-data step (2026-10-13) |
| snnTorch | Single optional surrogate-gradient adapter | When first needed |
| matplotlib | Reference figure generation (object-oriented API, no GUI backend) | Reporting (after Phase 4) |

The Phase 0 runtime has no third-party dependencies. Add dependencies when code
uses them; record their resolved versions in each run. No Hydra, simulator registry,
Brian2, GeNN or hardware SDK is introduced. CPU is the default; GPU is optional.

## Layout and ownership

| Module | Responsibility |
| --- | --- |
| `neurons/` | Reference LIF state, threshold, reset and refractory dynamics |
| `models/` | Small MLP/conv SNN and exposed per-layer state |
| `learning_rules/` | Rule declarations, rule state and explicit parameter updates |
| `encoding/` | Rate, latency, direct-current and binned-event (`events`) encoding |
| `datasets/` | Split roles, synthetic data, MNIST/Fashion-MNIST, N-MNIST (binned events), ordered streams |
| `shifts/` | Seeded transforms and explicit shift schedules |
| `protocols/` | Learning/evaluation ordering and shift-phase bookkeeping |
| `metrics/` | Quality, activity, forgetting, recovery and separate adaptation costs |
| `energy/` | Proxy assumptions and proxy calculation |
| `backends/` | Backend adapters and future hardware boundary |
| `records/` | RunRecord schema, validation and immutable result store |
| `reporting/` | Comparable-run validation, tables and figures |
| `utils/` | Seed control, configuration loading and environment provenance |

`configs/` mirrors configurable components. `experiments/reference/` will wire
framework components together; it owns no scientific primitives. The Jülich
experiment and partner-facing document are introduced only in Phase 6.

## Data flow and dependency boundaries

```text
resolved config + seed
        |
dataset / stream -> scheduled shift -> encoder -> model on simulation backend
        |                                         |
        +--------------- protocol ----------------+
                            |
                  learning rule + modulators
                            |
                  metrics + energy proxy
                            |
                   validated RunRecord
                            |
                immutable store -> reporting
```

Protocols control order. Models expose observations; rules decide updates. Rules
do not import datasets or protocols. Shifts transform samples without training
models. Metrics consume observations without modifying them. Records are independent
of backend implementations. Backend-specific imports remain in `backends/`; using
PyTorch tensors for the reference implementation does not require an adapter factory.

## Shared conventions

- Encoded tensor shape (accepted 2026-10-11): `[T, batch, ...]`; a timestep slice is `[batch, ...]`; `dt` is in milliseconds.
- Config declares `T`, `dt_ms`, seed list, device and reset policy explicitly.
- States reset at explicit protocol boundaries; online traces persist only when configured.
- Dataset splits carry `train`, `tune` or `test` roles. Test data cannot enter tuning.
- Learning observations contain only documented quantities; full models are not
  passed to a synapse-local update function.
- Configuration uses YAML and nested typed dataclasses, validated for unknown keys,
  positive sizes, supported choices and compatible components. No implicit defaults
  may be omitted from the saved resolved configuration.

## Proposed interfaces

### Neuron and model (Phase 1)

`LIF.step(input_current, state) -> (spikes, next_state)` exposes membrane potential,
threshold, time constant, reset (`subtract` or `zero`) and refractory state. Specify
the discrete integration scheme and compare it with the analytic constant-input
solution. Model forward execution exposes a named mapping of per-layer observations,
including spikes, membrane state and available pre/post activity. Document reset and
detach boundaries so state cannot accidentally retain a previous sample's graph.

### LearningRule (contract now; implementations in Phases 1–2)

```text
properties: uses_autograd, locality, online_capable, state, modulator_source
init_state(model) -> None
step(model, layer_states, modulators) -> updates
apply(updates) -> None
reset() -> None
```

`init_state` binds target parameters and initializes traces. `step` calculates
explicit updates; `apply` commits them to the bound parameters. `reset` resets rule
state without resetting learned weights. Define reset scope explicitly for each rule.

The outer orchestration signature above matches the spec. A local rule's numerical
kernel receives a restricted `LocalObservation` with documented pre/post spikes,
traces, local weight and any declared modulator. The model argument is used for
binding/dispatch only, never passed into the local kernel. No arbitrary layer losses,
other-layer states or gradients are exposed to a synapse-local kernel. Tests must
verify this boundary and run non-autograd rules with graph/gradient guards.

Declaration values are explicit:

```text
locality: synapse_local | layer_local | global_modulated | global_gradient
modulator_source: none | reward | error | prediction_error | learned
```

Surrogate BPTT declares autograd and global gradients. STDP declares its local
quantities; three-factor rules declare eligibility state and modulator provenance.
Predictive/local and learned-plasticity rules remain future extension points.

### Encoder and stream (Phases 1–3)

`Encoder.encode(sample, generator) -> encoded_input` returns `T`, encoded values,
encoding identity and input event rate. A seeded generator controls stochastic
encoding. Direct current must be identified as current, not misreported as events.

`Stream` yields samples with sample/task identifiers, split role, dataset version
and stream index. Ordering and shift schedules are saved, not inferred from shuffling.
Tests use synthetic samples without downloads. Tuning helpers reject test-role inputs.

### Shift (Phase 3)

```text
schedule: timing, start, end, recurrence and seed
definition: shift kind + parameters
apply(sample, stream_index, generator) -> shifted_sample
phase_at(stream_index) -> before | during | after
```

Composition applies shifts in a saved order. Timing covers abrupt, gradual,
recurring and incremental changes. Types cover tasks/classes, covariates, event
rate, temporal dynamics, concept/labels and unseen conditions. Begin with a small
seeded input-noise example; frequency drift alone does not satisfy the design.
For an abrupt shift, document which boundary sample represents the transition.

Implemented (Phase 3): `Schedule(timing, start, end, steps, period)` with
`strength(index)` and `phase_at(index)`; sample `start` is the first shifted sample.
Kinds: `noise` (covariate), `intensity` (illumination-like, changes input event rate
under rate coding), `label_permutation` (concept) and `latency_jitter` (temporal
dynamics: random per-channel delays applied after encoding). New classes/tasks use the
task sequence protocol. `Stream` encodes its samples: sample-level shifts in saved
order -> encoding -> temporal shifts in saved order, each shift with its own seeded
generator. Held-out unseen conditions beyond a fully shifted test split are not
implemented yet.

### Protocol (Phase 3; offline subset in Phase 1)

`Protocol.run(model, rule, stream, encoder, metrics) -> RunRecord` owns state reset,
evaluation and update order. Use direct explicit composition; no registry.

Implemented as functions (`run_offline`, `run_prequential`, `run_task_sequence`)
returning traces and metrics; the experiment assembles the `RunRecord`. Rules are
bound once by the caller (`init_state`), so rule state (e.g. random feedback weights,
optimizer moments) persists across protocol stages.

- Offline: train on `train`, optionally select using `tune`, then evaluate `test`.
- Task sequence: train/adapt in sequence and evaluate all seen tasks.
- Prequential: predict and record quality for sample `i` before using its permitted
  adaptation information. Evaluation labels must not reach the model before prediction.
  Any subsequent adaptation is recorded explicitly and does not count as tuning on
  a held-out test split. Adaptation permissions and label availability are protocol fields.

Shift protocols record quality before, during and after each scheduled shift.
Forgetting, recovery and adaptation costs remain separate fields. Future comparisons
use multiple configured seeds by default. Cheap pipeline checks may use one seed.

### RunRecord and result store (Phase 1 onward)

One record per run includes experiment identity, Git commit and dirty flag, full
resolved config, seeds, framework versions, device, `T`, neuron and rule parameters,
rule declarations, model neuron/synapse counts, dataset version/split, encoding,
shift definition/schedule, protocol, proxy assumptions and raw metric traces.

Simulation records may contain `energy_proxy` only with mandatory
`proxy_assumptions`. `energy_j` requires a hardware backend and measurement metadata
(source, method, measurement boundary and units). Unsupported fields fail validation.

Use atomic directory creation with `exist_ok=False` at
`results/<experiment>/<UTC timestamp>_<short-hash>/`. The hash includes resolved
config, provenance and a run nonce to allow repeated runs without overwriting.
Reject an existing path; do not silently append or replace files. Save resolved
config, `record.json` and raw traces under that directory. Generated results are
excluded from Git.

Comparison rejects mismatched `T`, model size, encoding or protocol unless
`allow_mismatch=True`; all accepted mismatches then appear in the output.

### HardwareBackend (placeholder; Phase 5 elaboration)

Future interface: `export(model, rule)`, `run(...)`, `measure(...)`. This phase
documents the boundary only; it does not implement an abstract class, exporter,
hardware runner or simulated measurement. See `neuromorphic_hardware.md`.

## Validation and phase boundaries

Phase 0 checks package building/imports and source syntax only. Phase 1 adds
constant-input LIF dynamics, threshold/reset/refractory, deterministic encoding,
activity metrics, result invariants and a synthetic smoke test. Phase 2 adds STDP
window, eligibility decay, zero-modulator and locality tests. Phase 3 verifies shift
schedules and test-before-train ordering. Later phases verify efficiency accounting
and export assumptions. CPU tests must run well under one minute.

The full reference experiment compares surrogate and local learning under a shift;
it cannot be complete until Phases 2–3. Early synthetic smoke runs validate only
the implemented pipeline. No research result or hardware measurement is claimed.
