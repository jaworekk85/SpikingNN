# Project status

Updated: 2026-10-11 (Europe/Warsaw).

## Current phase

**Phase 1 — foundations: reviewed and accepted (2026-10-11).**
Phase 0 was accepted on the same day. Phase 2 may start on request; no Phase 2 work
has started.

## Done

### Phase 0
- Architecture, interfaces and dependency plan in `docs/research_framework.md`;
  installable `src/` skeleton; config/experiment layout; repository instructions; CI.
- Local Git on `main`, `origin` = `https://github.com/jaworekk85/SpikingNN.git`.

### Phase 1
- `neurons.LIF`: exponential-Euler LIF (exact at sample points for piecewise-constant
  input), subtract/zero reset, refractory steps, fast-sigmoid surrogate derivative.
  State is explicit (`LIFState`); the neuron holds no hidden state.
- `encoding`: rate (seeded Bernoulli), latency (one spike at `round((1-x)(T-1))`),
  direct current (flagged as current; no event rate). Output shape `[T, batch, ...]`.
- `models.SpikingNet` with `build_model` for MLP and a small conv variant (one stride-2
  conv LIF layer). Exposes per-layer spikes and membrane potential; readout is linear
  on the last hidden layer's time-averaged spike rate. State is fresh per forward call.
- `learning_rules`: validated `RuleDeclaration`; `SurrogateBPTT` (Adam) following
  `init_state / step / apply / reset`, declared `global_gradient`, `uses_autograd=True`.
- `metrics`: accuracy, spike rate, temporal sparsity (per layer and size-weighted overall).
- `datasets`: role-validated `Split` (`train | tune | test`) and seeded synthetic
  prototype data. No downloads.
- `protocols.run_offline`: train on `train`, evaluate once on `test`.
- `records`: `RunRecord` validation (energy proxy needs assumptions; `energy_j` needs a
  hardware backend and measurement metadata) and `save_run` writing to a new
  `results/<experiment>/<UTC timestamp>_<hash>/` directory (`exist_ok=False`).
- `utils`: YAML -> nested frozen dataclasses (unknown keys rejected), resolved-config
  export, seeding, provenance (git commit/dirty, Python/torch versions, platform).
- `experiments/reference/run.py` + `configs/experiments/reference_tiny.yaml`:
  one record per seed.

## Validation

- Local: Python 3.12.3, PyTorch 2.14.1+cpu, `pytest`: 23 passed in ~9 s.
- Tests: analytic LIF response to constant input, threshold crossing, both resets,
  refractory period, surrogate gradient flow; encoder determinism/statistics/timing;
  MLP/conv shapes, neuron counts, no state carried between calls; activity metrics on
  hand-computed tensors; record validation, immutable store, config unknown keys,
  split roles; end-to-end smoke run (learns above chance, same seed -> same metrics).
- Reference tiny run (2 seeds, synthetic, 4 classes): test accuracy 0.66 / 0.70,
  spike rate ~0.08. Pipeline validation only, not a result.
- CI (Python 3.11 and 3.12, CPU PyTorch wheels) is configured but has not run yet.

## Decisions

- Python 3.11+ (raised from 3.10): PyTorch 2.14 is the last release with Python 3.10
  wheels; 2.15 (GA 2026-10-28) requires 3.11+.
- snnTorch (MIT) is the single optional surrogate-gradient adapter, not yet needed.
- Tensor convention: `[T, batch, ...]`, `dt` in milliseconds.
- No registries or class hierarchies: encoders are functions, one `SpikingNet` class
  with a builder, and the learning-rule interface is a convention until Phase 2 adds
  a second rule.
- `RunRecord` stores neuron, rule, encoding and protocol parameters inside the full
  resolved `config` rather than duplicating them as separate fields.
- No license for now (all rights reserved); revisit before any public release.

## Phase 1 review decisions (2026-10-11)

- Readout stays a linear layer on the last hidden layer's time-averaged spike rate.
  Revisit (e.g. non-spiking leaky-integrator readout) only if local rules in Phase 2
  require it.
- Records keep `n_neurons` (LIF units) and `n_weights` (weight parameters). For conv
  layers `n_weights` is not the logical synapse count; the synapse definition
  (padding, weight sharing) is deferred to Phase 4 together with synaptic-operation
  counting.
- MNIST (torchvision) is deferred until an experiment needs real data, most likely in
  Phase 3. Tests stay download-free.
- The conv model stays minimal (one 3x3/stride-2 conv LIF layer) until an experiment
  needs more.

## Known limitations

- Only the offline protocol exists; no shifts, streams, prequential evaluation or
  forgetting/recovery metrics (Phase 3).
- No synaptic-operation counting or energy proxy (Phase 4).
- No local learning rules or locality guards (Phase 2).
- No reporting (comparison guards, figures, tables) yet.

## Scientific caveats

- Synthetic smoke runs validate the pipeline; they are not research results.
- Spike counts are not synaptic operations and not measured energy.
- Surrogate BPTT is declared and treated as global-gradient learning, not local.

## Next phase

Phase 2: `LearningRule` interface, STDP and one three-factor rule with eligibility
traces, locality-declaration and autograd-guard tests; same model and data trainable
with either rule.
