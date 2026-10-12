# Test plan

Run with `python -m pytest -q`. All tests use CPU and tiny synthetic data and must
run well under one minute (Phases 1–4 + reporting: ~16 s).

- Phase 1: analytic LIF response, threshold/reset/refractory, deterministic encoding,
  activity metrics, record schema and immutable-store checks, synthetic smoke run.
- Phase 2: STDP window, eligibility decay, zero-modulator update, locality/autograd
  guards, online/minibatch consistency where applicable.
- Phase 3: seeded schedules, split-role guards, prequential test-before-train order,
  shift-phase reporting, forgetting/recovery/adaptation metrics.
- Phase 4+: operation counting, proxy/measurement separation and comparison guards.

CI installs the package with CPU PyTorch and runs the suite on Python 3.11 and 3.12.
