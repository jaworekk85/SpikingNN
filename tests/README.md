# Test plan

Scientific tests are introduced with concrete implementations, starting in Phase 1.
All tests use CPU and tiny synthetic data and must run well under one minute.

- Phase 1: analytic LIF response, threshold/reset/refractory, deterministic encoding,
  activity metrics, record schema and immutable-store checks, synthetic smoke run.
- Phase 2: STDP window, eligibility decay, zero-modulator update, locality/autograd
  guards, online/minibatch consistency where applicable.
- Phase 3: seeded schedules, split-role guards, prequential test-before-train order,
  shift-phase reporting, forgetting/recovery/adaptation metrics.
- Phase 4+: operation counting, proxy/measurement separation and comparison guards.

Phase 0 CI verifies source compilation and building/installing/importing the skeleton;
it does not claim scientific coverage.
