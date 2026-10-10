# SNN Research Framework — Agent Development Spec

## 0. How to use this document

This document defines **how** the repository is built and **what kind of research it must enable**.
It does **not** define specific papers or experiments to run.

- Read this file before any substantial change.
- Section 2 (research directions) is **context for design decisions only**. Do not implement experiments for these directions unless explicitly asked.
- Section 15 describes one concrete collaboration pilot (Forschungszentrum Jülich). It is implemented only in Phase 6 or when explicitly asked, on top of the general framework — never by special-casing the framework for it.
- Work in phases (section 13). **Stop after each phase**, update `STATUS.md`, and wait for review.
- When this spec is ambiguous, prefer the simpler design and record the open question in `STATUS.md`.

---

## 1. Purpose and identity

A general research framework for studying **spiking neural networks, local and biologically inspired learning, online/continual adaptation, and neuromorphic efficiency**.

Guiding chain:

```
learning rule → internal organization → adaptation → activity sparsity → (measured) energy
```

The repository is built around **learning rules and non-stationary data streams as first-class objects**, not around one architecture, one dataset, or one paper.

Development starts on ordinary CPU/GPU. Neuromorphic hardware (SpiNNaker2, Loihi-class, others via collaborators) is a later target and must be anticipated by the design, not implemented now.

---

## 2. Research directions (context only)

The abstractions must make the following lines of inquiry easy to set up. They are open questions, not tasks.

1. **Learning rule → representation.** How does the learning rule (surrogate-gradient BPTT, STDP-like, three-factor, predictive/local, learned plasticity) shape sparsity, modularity, redundancy and robustness of learned representations?
2. **Learning rule → adaptation.** Can local plasticity adapt to changing conditions with less forgetting and lower adaptation cost than global training?
3. **Learning rule → activity and energy.** Do local or spike-native rules produce activity patterns that translate into lower energy on neuromorphic hardware?
4. **Structural plasticity.** Growth/pruning of synapses and neurons as part of learning.
5. **Event-based sensing.** Event-camera style inputs relevant to robotics.

---

## 3. Key distinctions the framework must preserve

```
spike count ≠ synaptic operations ≠ GPU runtime ≠ neuromorphic energy
biological inspiration ≠ biological plausibility
surrogate-gradient training ≠ local learning
```

These are enforced in code where possible (section 10).

---

## 4. Design principles

1. **Learning rules are first-class, swappable objects**, independent of model and dataset.
2. **Non-stationarity is first-class.** Streams, shifts and evaluation protocols are explicit objects, not ad hoc dataset tweaks.
3. **Backend isolation.** Scientific logic does not depend on a specific simulator; backend-specific code lives in `backends/`.
4. **Hardware-friendly by construction.** Local rules should be expressible using only quantities available at the synapse/neuron (pre/post activity, traces, weight, modulatory signal), with optional fixed-point weights and bounded precision, so they can later be ported to neuromorphic hardware.
5. **Cheap by default.** Everything has a tiny configuration that runs on a laptop CPU in seconds.
6. **Proxy ≠ measurement.** Energy proxies and measured energy are separate fields (section 10).
7. **Small and boring beats large and clever.** No registries/factories/class hierarchies until at least two concrete implementations need them.

---

## 5. Software stack

- Python, PyTorch as the primary backend.
- Own minimal, transparent LIF implementation in PyTorch as the reference.
- One of snnTorch / SpikingJelly as an **optional** adapter for surrogate-gradient models (pick one; do not support both initially).
- Local rules implemented in plain PyTorch without autograd (`torch.no_grad`).
- Brian2 / GeNN: not now; only if explicit dynamics or scale require them later.
- Configs: YAML + typed dataclasses (or pydantic). Do not introduce Hydra or similar frameworks unless asked.

---

## 6. Core abstractions

### 6.1 `Neuron`
LIF reference; optional IF and adaptive LIF.
For membrane potential `u_t`: `z_t = H(u_t − θ)`.
Exposes: membrane state, threshold, reset rule (subtract / zero), refractory option, spike tensor, optional adaptation state, time constant(s).

### 6.2 `Model`
Small MLP SNN, small convolutional SNN, optional recurrent SNN.
Exposes per-layer spikes and states for metrics and for learning rules. No large Spiking Transformers.

### 6.3 `LearningRule`
Interface: `init_state(model)`, `step(model, layer_states, modulators) -> updates`, `apply(updates)`, `reset()`.
Supports online (per time step / per sample) and minibatch updates.

Generic local update:
```
Δw_ij = f(x_i, x_j, z_i, z_j, w_ij, e_ij, m_t, …)
```
Three-factor form:
```
Δw_ij = e_ij(t) · M(t)       e_ij: local eligibility trace, M: modulatory signal
```

Every rule **declares its properties** (validated by tests, section 10):
```
uses_autograd: bool
locality: synapse_local | layer_local | global_modulated | global_gradient
online_capable: bool
state: list of traces it maintains
modulator_source: none | reward | error | prediction_error | learned
```

Initial rule types to anticipate: `surrogate_bptt`, `stdp`, `three_factor`, `predictive_local`, `learned_plasticity`.

### 6.4 `Encoder`
Static → spikes: rate, latency/time-to-first-spike, direct current input. Event data passed through with binning options. Exposes time steps `T` and resulting input event rate.

### 6.5 `Dataset` / `Stream`
Wrappers expose: temporal encoding, `T`, event rate, class/task metadata, stream order, split roles (`train | tune | test`).

### 6.6 `Shift`
Composable, seeded, with an explicit schedule. Taxonomy:
- timing: abrupt, gradual, recurring, incremental,
- type: new classes/tasks, covariate (sensor noise, illumination-like perturbations, event-rate changes), temporal-dynamics changes, concept/label changes, previously unseen conditions.

Class-frequency change must not be the only supported notion of drift.

### 6.7 `Protocol`
Defines how learning and evaluation interleave:
- offline training then test,
- task/domain sequence with evaluation on all seen tasks,
- **prequential (test-then-train) streaming evaluation**.
Records pre-shift, during-shift and post-shift quality.

### 6.8 `HardwareBackend`
Interface placeholder only (`export(model, rule)`, `run(...)`, `measure(...)`). No implementation now; document assumptions in `docs/neuromorphic_hardware.md`.

### 6.9 `RunRecord`
The single result object written by every run (section 10).

---

## 7. Metrics

- Task quality `Q`.
- Spike rate: `#spikes / (#neurons × T)`, per layer and overall.
- Temporal sparsity: fraction of inactive neuron–time pairs.
- Synaptic operations: spikes × fan-out (definition documented in `docs/metrics.md`).
- Forgetting: `F_a = Q_a^max − Q_a^final`.
- Recovery time: first time `Q_t ≥ Q_target` after a shift (with `Q_target` defined in config).
- Adaptation cost: number of updates, synaptic ops and spike activity during adaptation, reported separately, not as a single sum.
- Energy proxy: computed from synaptic ops / spikes with **explicit, configurable assumptions** (e.g. energy per operation), always labeled as proxy.
- Measured energy: only from a hardware backend.

---

## 8. Reproducibility

Every run saves: git commit + dirty flag, full resolved config, seed, framework versions, device, `T`, neuron parameters, learning-rule parameters and declared properties, dataset version, shift definition and schedule, protocol, energy-proxy assumptions, raw metric traces.

Multi-seed runs are the default for any comparison (seed count in config).

---

## 9. Testing

- All tests run on a laptop CPU in well under a minute.
- **Neuron dynamics:** LIF response to constant input matches the analytic solution; reset and threshold behavior; refractory period.
- **Learning rules:** sign and shape of STDP window on toy spike pairs; eligibility trace decay; three-factor update is zero when modulator is zero; online vs minibatch consistency where applicable.
- **Locality declarations:** see section 10.
- **Determinism:** same seed → same spikes and metrics.
- **Shifts and protocols:** schedules produce the expected stream; prequential evaluation never trains before testing on a sample.
- **Smoke run** of the reference experiment.

---

## 10. Invariants enforced in code

1. **Immutable results.** Each run writes to a new directory `results/<experiment>/<timestamp>_<short-hash>/`; writing into an existing one raises an error.
2. **Locality is verified, not claimed.** A rule declaring `uses_autograd=False` is executed in tests under conditions that fail if autograd graphs are built or `.grad` is populated. A rule declaring `synapse_local` may only access its documented local quantities (enforced by the interface it receives).
3. **Proxy vs measured energy.** The schema allows `energy_j` only from a hardware backend with measurement metadata; simulation results can only populate `energy_proxy` with mandatory `proxy_assumptions`.
4. **No tuning on test.** The data API exposes split roles; hyperparameter search utilities refuse to read data with role `test`.
5. **Comparable settings.** Comparison utilities refuse to place runs side by side if `T`, model size (neurons/synapses), encoding, or protocol differ — unless called with `allow_mismatch=True`, in which case mismatches are shown in the output.
6. **Pre/post shift always reported.** Protocols with shifts must record quality before, during and after each shift; forgetting and adaptation are reported separately.
7. **Plausibility wording.** Reports and docs describe rules by their declared locality/properties, not as "biologically plausible" by default.

---

## 11. Repository structure

```
src/snn_research/
  neurons/
  models/
  learning_rules/
  encoding/
  datasets/
  shifts/
  protocols/       # offline, task-sequence, prequential
  metrics/
  energy/          # proxies and assumptions
  backends/        # simulation backends; hardware interface placeholder
  records/         # RunRecord schema, validation, result store
  reporting/
  utils/

configs/
  models/  neurons/  datasets/  encoding/  learning_rules/  shifts/  protocols/  experiments/

experiments/
  reference/       # pipeline-validation experiment (section 14)
  juelich_pilot/   # collaboration pilot (section 15), created in Phase 6
  # other experiment folders are added by humans as research questions emerge

docs/
  research_framework.md
  learning_rules.md
  metrics.md
  neuromorphic_hardware.md
  juelich_pilot.md

tests/
results/
STATUS.md
```

Apart from `juelich_pilot/`, do not create experiment folders named after collaborators or specific papers.

---

## 12. Datasets

Start small and reproducible:
- static: MNIST, Fashion-MNIST; CIFAR only after smoke tests pass,
- event-based: N-MNIST, DVS Gesture; later event-camera datasets relevant to robotics.

Downloads are cached and versioned; tests use tiny synthetic or subsampled data and never require downloads.

---

## 13. Development phases (stop and report after each)

**Phase 0 — architecture proposal.**
Write `docs/research_framework.md` with module layout, interfaces (especially `LearningRule`, `Shift`, `Protocol`) and dependency plan. Skeleton only. Stop for review.

**Phase 1 — foundations.**
LIF neuron, small MLP/conv SNN, encoders, surrogate-gradient training, spike/activity metrics, result store, test suite.

**Phase 2 — local learning.**
`LearningRule` interface, STDP and one three-factor rule with eligibility traces, locality-declaration tests. Same model/data can be trained with either rule.

**Phase 3 — non-stationarity.**
`Shift` library, `Protocol` implementations including prequential evaluation, forgetting/recovery/adaptation-cost metrics.

**Phase 4 — efficiency accounting.**
Synaptic-op counting, energy proxies with explicit assumptions, optional fixed-point / bounded-precision weights for rules.

**Phase 5 — hardware readiness.**
Document hardware constraints, flesh out the `HardwareBackend` interface, add export of model + rule description. No hardware-specific implementation without explicit request.

**Phase 6 — Jülich pilot.**
Package the pilot described in section 15 using only framework components from Phases 1–5.

---

## 14. Reference experiment (pipeline validation, not research)

Purpose: prove the whole pipeline works end-to-end. Its numbers are not a research result.

```
dataset:  MNIST (rate-encoded) or a tiny N-MNIST subset
model:    small LIF MLP
rules:    surrogate BPTT and one local/three-factor rule
shift:    one simple, seeded shift (e.g. class-incremental split or input noise)
protocol: pre-shift → shift → post-shift evaluation
metrics:  accuracy, forgetting, recovery, spike rate, synaptic ops
output:   one figure + metrics table
```

Must run on a laptop CPU in a few minutes at its smallest configuration.

---

## 15. Jülich pilot (collaboration track)

### Purpose
A small, self-contained, reproducible package that demonstrates what the framework can do with local learning under non-stationarity, and that is designed from the start to be portable to neuromorphic hardware (primarily SpiNNaker2). It is the technical basis for approaching Forschungszentrum Jülich.

Working title: **Local plasticity for energy-efficient online adaptation in SNNs.**

Principle: **the scientific contribution must exist before hardware access.** The pilot shows a result in simulation; the collaboration is sought to deploy it on hardware, measure real energy/latency, and compare simulation proxies with measurements.

### Content
```
model:    small LIF network (size chosen to be realistic for on-chip deployment)
rules:    surrogate BPTT (reference)
          one three-factor / local rule
          optionally: learned plasticity
data:     one static or event-based benchmark (event-based preferred if feasible)
shift:    one controlled, seeded shift
protocol: pre-shift → shift → post-shift, online adaptation after the shift
metrics:  quality (pre/post), recovery time, forgetting,
          spike rate, synaptic operations, adaptation cost, energy proxy
seeds:    multiple (from config)
```

Comparisons should be quality-aware (Pareto or quality-matched), so that cheaper adaptation is not simply a consequence of a weaker model.

### Hardware-portability requirements
- The local rule uses only quantities available at the synapse/neuron plus a broadcast modulatory signal; no hidden global gradients (verified by the locality tests, section 10).
- Optional fixed-point / bounded-precision weights and traces; report the effect of quantization on results.
- Network size, fan-in/fan-out, time step and number of time steps documented explicitly.
- Assumptions about the target hardware (supported neuron models, plasticity support, precision, memory per core, I/O) are listed in `docs/juelich_pilot.md` as **to be confirmed with the partner**, not as facts.

### Deliverables
- `experiments/juelich_pilot/`: configs and one entry-point script reproducing all results from scratch.
- One main figure (quality and adaptation vs activity/synaptic ops for the compared rules) and one metrics table.
- `docs/juelich_pilot.md`, short and partner-facing:
  - the question and the simulated result,
  - what is needed from the partner (hardware access, porting support, energy measurement),
  - what the pilot provides (reproducible code, rule specification, export of model + rule),
  - open hardware questions,
  - a clear separation between simulation proxies and anything that would require measurement.

---

## 16. Agent working rules

1. Read this spec before substantial changes.
2. Propose architecture before implementing it; stop at phase boundaries.
3. Update `STATUS.md` after each work session: done, open questions, known limitations, scientific caveats.
4. Keep learning rules modular and independent of datasets and models.
5. Keep cheap smoke tests passing; add tests for every neuron model and learning rule.
6. Never present simulation proxies as hardware measurements.
7. Never call a rule local if it uses hidden global gradients.
8. Do not add experiments for the research directions unless asked.
9. Avoid expensive sweeps unless asked.
10. Prefer falsifiable, small experiments and readable code over architecture proliferation.
