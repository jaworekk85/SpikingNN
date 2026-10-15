# Project status

Updated: 2026-10-15 (Europe/Warsaw). This file describes the **current state** only and
stays short (target < 120 lines); history is in the Git log (one commit per phase).

## Current state

- Phases 0–5 of the spec are implemented and accepted; real data (MNIST, Fashion-MNIST,
  N-MNIST) and tune-split hyperparameter tuning are in place. Phase 6 (Jülich pilot)
  has not started.
- Paused step: apply the tune-split results (below) to the MNIST/N-MNIST configs and
  re-run the test evaluation, replacing the earlier, optimistic numbers.
- Pending user decision: keep STDP in the pilot (needs homeostasis / lateral
  inhibition) or base the pilot on the three-factor rule vs BPTT.
- 84 tests, ~16 s on CPU, no downloads; CI green on Python 3.11 and 3.12.

## What exists (details in `docs/`)

- LIF neuron; MLP/conv SNN with per-layer observations; rate, latency, direct and
  binned-event encodings.
- Rules: surrogate BPTT (global gradient), pair STDP (synapse-local), error-modulated
  three-factor with eligibility traces and random feedback (global-modulated);
  optional fixed-point weights.
- Data: synthetic prototypes, MNIST/Fashion-MNIST (IDX), N-MNIST (event zips, binned);
  seeded shifts (noise, intensity/event rate, label permutation, latency jitter) with
  abrupt/gradual/incremental/recurring schedules.
- Protocols: offline, prequential (test-then-train), task sequence; train/tune-only
  grid search.
- Metrics: accuracy, spike rate, sparsity, pre/during/post-shift quality, recovery,
  forgetting, adaptation cost, logical synaptic operations, energy proxy (pJ, with
  saved assumptions, arithmetic only).
- Records/reports: immutable run directories, multi-seed reports (tables + figure) with
  a comparability guard. Hardware: model + rule export, constraint check against a
  SpiNNaker2 draft (to be confirmed), `HardwareBackend` protocol (no backend).

## Latest results (pipeline validation, not research results)

Test-split numbers from 2026-10-13 used local-rule settings chosen on **test** accuracy
(a violation of "no tuning on test"); they are optimistic for the local rules and are
only indicative until re-run with the tuned settings.

| Data (3 seeds) | BPTT test acc | Three-factor | STDP + 3-factor readout |
| --- | --- | --- | --- |
| MNIST 6k/1k, noise shift | 0.941 ± 0.007 | 0.924 ± 0.003 | 0.710 ± 0.024 |
| N-MNIST 6k/1k, event rate halved | 0.926 ± 0.011 | 0.854 ± 0.007 | 0.107 (chance) |

N-MNIST stream quality before/during/after: BPTT 0.995/0.895/0.898, three-factor
0.872/0.826/0.831 (smallest drop) with about half of BPTT's spikes (707 vs 1292 per
sample). Synaptic operations are dominated by input spikes.

Tune-split tuning (2026-10-14; objective = mean of tune accuracy after pretraining and
shifted-tune accuracy after the stream; seeds 100/101; outputs in local `results/`):

| Data | BPTT | Three-factor | STDP |
| --- | --- | --- | --- |
| MNIST | lr 0.003 -> 0.893 | lr 0.03, readout 0.25 -> 0.894 (lower grid edge) | a+ 1e-4, a- 1.5e-4, readout 1.0 -> 0.681 |
| N-MNIST | lr 0.03 -> 0.903 | lr 0.3, readout 1.0 -> 0.881 (upper grid edge) | a+ 1e-4, a- 3e-4, readout 1.0 -> 0.284 |

The three-factor optimum lies on the grid edge on both datasets, so the grids should be
extended (lower on MNIST, higher on N-MNIST) before final numbers.

## Decisions (dated)

- 2026-10-11: Python 3.11+ (PyTorch 2.15 drops 3.10); tensors `[T, batch, ...]`, `dt` in
  ms; own LIF, snnTorch only as an optional adapter; no license yet (revisit before
  sharing code); no registries or class hierarchies.
- 2026-10-11: rate readout (linear on time-averaged spikes); minimal conv model.
- 2026-10-11: work directly on `main` (no branches/PRs).
- 2026-10-12: synaptic operations count logical connections (conv padding emits
  nothing, weight sharing does not reduce counts); records store `n_synapses`.
- 2026-10-12: energy proxy defaults 0.9 pJ/synop, 4.6 pJ/MAC (45 nm FP32, Horowitz
  2014), memory access excluded; neuron/weight updates counted but not costed.
- 2026-10-13: datasets read from original files with numpy (no torchvision); N-MNIST
  binned 300 ms -> 30 x 10 ms; `neuron.dt_ms` must equal the bin width.
- 2026-10-13: keep our own JSON export; add NIR only if the partner prefers it.
- 2026-10-15: configs list only non-default values; STATUS describes state, not history.

## Known limitations and caveats

- STDP without homeostasis is unstable on 784+ inputs (activity runs away or dies);
  it fails on N-MNIST. Homeostasis / lateral inhibition (Diehl & Cook 2015) is the
  known remedy.
- Results are sensitive to tiny numerical differences per seed; report multi-seed only.
- Prequential evaluation is per minibatch (8); recovery resolution equals the window
  (64); forgetting uses two checkpoints (before/after the stream).
- The three-factor modulator arrives once per sample (not per timestep); local rules
  do not update biases; conv synapses stay fixed under local rules.
- Quantization: deterministic rounding, weights only (no bounded-precision traces);
  local rules still learn float weights by default (SpiNNaker2 draft: 8-bit).
- Hardware memory estimate is rough; all hardware facts are drafts to be confirmed.
- No quality-matched / Pareto comparison yet (required for the pilot).
- Prequential quality is measured on the adaptation stream (training data); held-out
  clean/shifted accuracy is reported separately.
- Energy proxies are not measurements; spike counts are not synaptic operations.
- The three-factor rule is `global_modulated` (local traces + broadcast error); only
  STDP is `synapse_local`; BPTT is global-gradient learning.

## Next steps

1. Extend the three-factor grids, apply tuned settings to the MNIST/N-MNIST configs,
   re-run the test evaluation (3 seeds) and replace the table above.
2. Decide on STDP for the pilot (fix with homeostasis/inhibition or drop).
3. Phase 6 pilot in simulation: 5–10 seeds, 8-bit weights, chip-realistic size,
   quality-matched/Pareto comparison, short English `docs/juelich_pilot.md`.
4. Before sharing code with a partner: choose a license.
