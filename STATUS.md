# Project status

Updated: 2026-10-11 (Europe/Warsaw).

## Current phase

**Phase 0 — architecture proposal: reviewed and accepted (2026-10-11).**
Phase 1 may start on request. No Phase 1 implementation has started.

## Done

- Preserved `SNN_AGENT_RESEARCH_SPEC.md` as the authoritative specification.
- Proposed module responsibilities, data flow, interfaces and dependency plan in
  `docs/research_framework.md`.
- Created an installable `src/` package skeleton with the specified module layout.
- Added configuration directories and a reference-experiment placeholder.
- Documented learning-rule contracts, metric definitions and hardware assumptions.
- Added repository instructions, Git exclusions and a lightweight CI packaging check.
- Initialized local Git on `main` and configured `origin` as
  `https://github.com/jaworekk85/SpikingNN.git` (existing empty remote repository).

## Validation

- Python 3.10.11: source compilation passed (initial Phase 0 validation).
- After the review, recreated the local `.venv` on Python 3.12.3: source compilation,
  wheel build/install of `snn-research==0.0.1` and isolated imports of all 13 modules passed.
- Imported the installed package and all 13 component modules in isolated Python mode.
- Confirmed Git ignores the virtual environment, build artifacts, bytecode and results.
- CI is configured for Python 3.11 and 3.12; local validation covers 3.12 only.
- Scientific tests are deferred until concrete implementations exist in Phase 1.

## Decisions (Phase 0 review, 2026-10-11)

- Python 3.11+ (raised from 3.10): PyTorch 2.14 is the last release with Python 3.10
  wheels; 2.15 (GA 2026-10-28) requires 3.11+. Use PyTorch, PyYAML and typed
  standard-library dataclasses.
- snnTorch (MIT) is the single optional surrogate-gradient adapter; the reference
  LIF remains our own PyTorch implementation. Add the adapter only when needed.
- Tensor convention accepted: `[T, batch, ...]`, with `dt` in milliseconds.
- Keep public interfaces in this phase as documentation; add concrete code in the
  corresponding implementation phase, without registries or backend class hierarchies.
- No license for now (all rights reserved); revisit before any public release.

## Open questions

- Specify convolutional synaptic-operation counting for padding and weight sharing
  when convolutional metrics are implemented.

## Known limitations

- Package modules are placeholders; there is no experiment CLI, training or simulation.
- YAML configs, RunRecord validation and scientific tests are planned, not implemented.
- No datasets have been downloaded. No hardware support or pilot has been implemented.

## Scientific caveats

- Architecture and pipeline validation do not constitute research results.
- Spike counts and synaptic operations do not constitute measured energy.
- Locality must be verified by interface restrictions and tests before making claims.

## Next phase

Phase 1: reference LIF dynamics, small MLP/conv SNN, encoders, surrogate-gradient
training, spike/activity metrics, validated immutable result storage and CPU tests.
The complete reference comparison awaits local learning and shifts in later phases.
