# Project status

Updated: 2026-10-12 (Europe/Warsaw).

## Current phase

**Phase 4 — efficiency accounting: complete, awaiting review.**
Phases 0–2 were accepted on 2026-10-11 and Phase 3 (with the latency-jitter addition)
on 2026-10-12. No Phase 5 work has started. Agreed next step after review: multi-seed
reporting (figure + metrics table), then Phase 5.

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

### Phase 2
- `LearningRule` protocol (interface only, no base class) and `LocalObservation`
  (`pre`, `post`, `post_potential`), the only input of local numerical kernels.
- `STDP`: pair-based, exponential traces, clipped weights; `synapse_local`, no modulator.
- `ThreeFactor`: presynaptic trace x pseudo-derivative -> eligibility trace, times an
  output-error modulator; hidden layers get the error through fixed random feedback.
  Readout uses the delta rule with its own `readout_lr`. `global_modulated`.
- Model observations now include presynaptic input and the pre-reset potential `u`
  (the value compared with threshold); `ModelConfig.init_gain` scales initial weights.
- `run_offline` accepts several rules; forward passes build a graph only if a rule
  uses autograd. Modulators: `loss` and detached `error`.
- Reference: `rule: surrogate_bptt | three_factor | stdp` (or `--rule`); `stdp` means
  STDP on hidden layers + three-factor delta rule on the readout. Records store
  `rule_declarations` for every rule used.
- `docs/learning_rules.md` documents both rules and their verified properties.

### Phase 3
- `shifts`: `Schedule` (`abrupt | gradual | incremental | recurring`, `start`, `end`,
  `steps`, `period`) giving per-index strength and `before | during | after` phase;
  `ShiftConfig` kinds `noise` (covariate), `intensity` (illumination-like / input event
  rate), `label_permutation` (concept) and `latency_jitter` (temporal dynamics: random
  per-channel delays after encoding); `shift_and_encode` orders sample-level shifts ->
  encoding -> temporal shifts.
- `datasets.Stream`: seeded order over a `train`-role split; yields encoded, shifted
  minibatches by stream index; `Split.subset(classes)` for task sequences.
- `evaluate(..., shifts=...)` evaluates a held-out split under shifts at full strength.
- `protocols`: `forward_and_learn` (one forward, then rule updates from it),
  `train_epochs`, `run_offline`, `run_prequential` (test-then-train; `adapt` flag) and
  `run_task_sequence` (accuracy matrix over all tasks after each task). Rules are now
  bound once by the caller, so their state persists across protocol stages.
- `metrics`: `phase_quality`, `recovery_samples` (trailing window fully after onset,
  `None` if unrecovered), `forgetting` (max - final). Adaptation cost inside the shift
  window: update steps and spikes, separate fields.
- Reference: offline pretraining -> prequential stream with one shift (default: abrupt
  `noise`, magnitude 0.3, samples 256–511 of 768) -> clean/shifted test accuracy.
  `RunRecord.shift` stores the resolved shift; metrics may be `None` (unavailable).

### Phase 4
- `metrics.synops`: `fan_out` (dense and conv, padding/edge aware), `synapse_count`
  (logical connections incl. readout), `synaptic_operations` (per-sample synops and,
  for direct-current input, separate `input_macs`).
- `energy.EnergyProxyConfig` / `energy_proxy_pj`: per-component proxy in pJ with saved
  coefficients, units, source and `kind: proxy (not measured)`. Defaults: 0.9 pJ/synop,
  4.6 pJ/MAC (45 nm FP32 arithmetic only), 0 for neuron and weight updates.
- `evaluate` reports spikes, synops, input MACs and neuron updates per sample;
  `forward_and_learn` counts weights written; prequential adaptation cost adds weights
  written, synops and input MACs (separate fields).
- `learning_rules.precision`: optional fixed-point weights (`weight_bits`, `w_max`) for
  STDP and three-factor, applied at binding and after every update.
- Reference: `energy` config section; `RunRecord.energy_proxy` = inference proxy per
  clean test sample after pretraining, with `proxy_assumptions`; adaptation energy proxy
  as separate metrics; `model.n_synapses` added.

## Validation

- Local: Python 3.12.3, PyTorch 2.14.1+cpu, `pytest`: 61 passed in ~12 s.
- Phase 4 tests: conv fan-out equals brute-force counting (odd/even sizes, with and
  without padding); dense fan-out and synapse count; synops = spikes x fan-out with
  direct input counted as MACs instead; energy-proxy arithmetic and assumption fields;
  fixed-point levels and validation; a rule keeps weights on the grid during a stream;
  adaptation cost counts updates, weights written and synops; reference record carries
  an energy proxy with assumptions and no measured energy.
- Phase 3 tests: strengths for all four timings, phases and schedule validation; shift
  kinds at zero/full strength and seeded noise; stream order, index-based shifting and
  `train`-role guard; latency jitter bounds, determinism and post-encoding application
  from the shift start; phase quality, recovery (incl. unrecovered `None`) and forgetting
  on hand-made traces; prequential logs forward -> update per batch and records quality
  from pre-update logits; `adapt=False` leaves weights unchanged; task-sequence matrix
  and forgetting; reference record contains shift, pre/during/post quality, recovery,
  forgetting and adaptation cost.
- Phase 2 tests: STDP window sign/shape, eligibility decay, zero-modulator update,
  minibatch = mean of per-sample updates (both rules), no graph / no `.grad` from local
  rules even after a grad-enabled forward, kernels accept only `LocalObservation`,
  declarations; every rule trains the same model and data, deterministically per seed.
- Tests: analytic LIF response to constant input, threshold crossing, both resets,
  refractory period, surrogate gradient flow; encoder determinism/statistics/timing;
  MLP/conv shapes, neuron counts, no state carried between calls; activity metrics on
  hand-computed tensors; record validation, immutable store, config unknown keys,
  split roles; end-to-end smoke run (learns above chance, same seed -> same metrics).
- Reference tiny run after Phase 2 (`init_gain: 5`, 2 seeds, synthetic, 4 classes),
  test accuracy / overall spike rate: surrogate BPTT 0.94, 0.98 / ~0.11; three-factor
  0.98, 0.87 / ~0.24; STDP + three-factor readout 0.90, 0.94 / ~0.04. Pipeline
  validation only, not a result or a rule comparison (single tiny config, no tuning
  protocol, no quality-matched comparison).
- Reference tiny run after Phase 3 (2 seeds; pretrain test acc; stream quality
  before/during/after the noise shift; recovery samples with window 64, target 0.8;
  clean test acc after the stream): surrogate BPTT 0.94/0.98; 0.97–0.99 / 0.84–0.85 /
  0.89; 64, 64; 0.98/0.99. Three-factor 0.98/0.87; 0.89–0.95 / 0.74–0.78 / 0.79; 186,
  193; 0.96/0.98. STDP + three-factor readout 0.90/0.94; 0.84–0.91 / 0.70–0.78 /
  0.78–0.81; 135, 368; 0.85/0.91. Pipeline validation only, not a rule comparison.
- Reference tiny run after Phase 4: ~5,000 synops per test sample (mostly input spikes
  x 32 hidden targets), energy proxy ~4,400–5,000 pJ per sample for all rules. A
  three-factor precision check (single seeds, `w_max: 8`): full precision 0.98/0.98
  pretrain test accuracy, 8 bits 0.96/0.99, 6 bits 0.97/0.98, 4 bits 0.78/0.78;
  `w_max: 2` alone (no quantization) already drops to 0.73/0.84 because readout weights
  exceed it. Pipeline validation only, not a result.
- CI runs on every push to `main` (Python 3.11 and 3.12, CPU PyTorch wheels).

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
  counting. Resolved in Phase 4: records also store `n_synapses` (logical connections;
  see `docs/metrics.md`).
- MNIST (torchvision) is deferred until an experiment needs real data, most likely in
  Phase 3. Tests stay download-free.
- The conv model stays minimal (one 3x3/stride-2 conv LIF layer) until an experiment
  needs more.

## Phase 2 notes (accepted)

- Default PyTorch init left hidden LIF layers almost silent, so STDP had nothing to
  act on. The tiny config now uses `init_gain: 5.0`. A principled alternative
  (threshold/init calibration, homeostasis) can wait until real data is used.
- In the `stdp` setting the readout is trained by the three-factor delta rule; STDP
  alone is unsupervised. Its accuracy mainly reflects random-like hidden features plus
  a trained readout; whether STDP features help is a research question, not tested.
- Three-factor modulator arrives once per sample, so updates are per sample or
  minibatch, not per timestep; per-timestep online learning signals are not implemented.
- Hidden-layer error feedback uses fixed random matrices drawn from the run seed.

## Phase 3 notes (accepted)

- Prequential evaluation is per minibatch (default 8): the whole batch is predicted
  before any update from it. Batch size 1 gives strict per-sample test-then-train at
  higher cost.
- Recovery resolution is bounded by the window (64 samples); BPTT's value of 64 means
  "within the first full window", not exactly 64.
- Forgetting in the reference uses two checkpoints (before/after the stream); finer
  histories would need periodic held-out evaluation during the stream.
- Latency jitter (max 10 steps) barely affects the tiny reference: with rate coding and
  a rate readout, spike timing carries little information. Under latency encoding the
  tiny config does not learn at all (pretrain ~0.32), so temporal shifts cannot yet be
  shown to matter; a timing-sensitive setup needs model/encoding tuning (later work).
- Not implemented: event data, a hyperparameter-search utility with a test-role guard
  (no tuning utility exists yet; protocols check split roles instead), multi-seed
  aggregation/reporting (planned right after Phase 4).

## Open questions (Phase 4, for review)

- Energy proxy defaults are arithmetic-only 45 nm FP32 costs; memory access is excluded
  and would likely dominate. A memory-aware or target-hardware model (e.g. SpiNNaker2
  figures from a partner) can replace the coefficients without code changes.
- In the tiny reference, input spikes dominate synops, so rule differences in hidden
  activity barely move the total; per-layer synops are not yet reported separately.
- Neuron-update and weight-update energy are 0 by default (counted, not costed).
- Quantization uses deterministic rounding; stochastic rounding and bounded-precision
  traces are not implemented. STDP now clips weights to `w_max` at binding (previously
  on its first update), which slightly changed one STDP seed (0.914 -> 0.906).

## Known limitations

- No reporting (comparison guards, figures, tables) yet.

## Scientific caveats

- Synthetic smoke runs validate the pipeline; they are not research results.
- Spike counts are not synaptic operations and not measured energy.
- Prequential quality is measured on the training stream (adaptation data), not on a
  held-out split; held-out clean/shifted test accuracy is reported separately.
- Surrogate BPTT is declared and treated as global-gradient learning, not local.
- The three-factor rule is `global_modulated`: weights update from local traces and a
  broadcast error signal; it is not synapse-local. Only STDP is `synapse_local`.

## Next phase, after review

Multi-seed reporting (aggregation, comparison guards, reference figure and metrics
table), then Phase 5: document hardware constraints, flesh out the `HardwareBackend`
interface, export of model + rule description (no hardware-specific implementation).
