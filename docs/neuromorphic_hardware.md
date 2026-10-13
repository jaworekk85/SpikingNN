# Neuromorphic hardware boundary

Phase 5: the boundary is defined and a model + rule export exists, but **no
hardware backend is implemented** and no hardware has been used. Everything below
about specific chips is compiled from public sources and is **to be confirmed with the
hardware partner**. The framework itself assumes no chip capability; capabilities enter
as data (`HardwareConstraints`, e.g. `configs/hardware/spinnaker2_draft.yaml`).

## What the framework provides (`snn_research.backends`)

- `export_network(model, rules, encoding, input_shape)` -> JSON description + float32
  weights; `save_export` writes `export.json` and `weights.npz` (never overwrites).
  The description contains:
  - timing: `dt_ms`, `T`; encoding and whether inputs are spikes or current;
  - neuron: LIF equations, `tau_mem_ms`, `beta`, input scale `1 - beta`, threshold,
    reset (`subtract`/`zero`), refractory steps, initial state;
  - layers: type (dense / conv2d with kernel, stride, padding), shapes, neurons,
    logical synapses, max fan-in and fan-out, weight/bias keys;
  - readout: non-spiking rate readout (`W_out * mean_T(spikes) + b`), classes;
  - rules: declaration (locality, autograd, modulator), parameters (incl. weight
    precision), state per synapse / pre- / postsynaptic neuron, target layers;
  - totals: neurons, synapses, stored weights, max fan-in.
- `HardwareBackend` protocol: `load(description, weights)`, `run(inputs)`, `measure()`.
- `Measurement(energy_j, source, method, boundary, units)`: the only way to fill
  `RunRecord.energy_j` (`record_fields()` sets `backend_kind="hardware"`).
- `HardwareConstraints` + `check_constraints`: findings (`error | warning | info`) for
  neuron model, reset, `dt`, fan-in, weight precision of learning rules, on-chip
  learning support, readout mapping and a rough memory estimate (logical synapses x
  weight bits + rule state; no routing, code or buffer overhead). Unknown limits
  (`null`) are skipped, not assumed.

Reference usage: `run.py ... --hardware configs/hardware/spinnaker2_draft.yaml` writes
the export and the check into every run directory.

## Mapping notes for a port

- Our LIF update is `u = beta v + (1 - beta) I`. Hardware LIFs often use
  `v = alpha v + I`; the export gives `beta` and the input scale `1 - beta` so weights
  or thresholds can be rescaled explicitly.
- Rate coding draws Bernoulli spikes per step; a port must use the same `T` and `dt`
  and an equivalent input generator, or feed recorded spike trains.
- Local rules need: STDP — one pre trace per input neuron and one post trace per
  neuron; three-factor — one eligibility trace per synapse, pre traces, and fixed
  random feedback weights (`n_classes` per hidden neuron) plus a broadcast error at
  the end of each sample. Surrogate BPTT is an offline reference, not an on-chip rule.
- The readout is not spiking: it needs spike counters and an affine map per class
  (on a core or the host).
- Learned weights are float unless a rule sets `weight_bits`/`w_max`; deployment with
  8-bit weights should be simulated first (Phase 4 precision option).

## Public information on candidate hardware (to be confirmed)

| Topic | SpiNNaker2 (draft) | Loihi 2 (for context) |
| --- | --- | --- |
| Compute | 152 processing elements per chip, each Arm Cortex-M4F + accelerators (exp, RNG, MAC) | 128 neurocores (secondary source); up to ~1M neurons / 120M synapses per chip reported |
| Memory | 128 kB SRAM per element (~19 MB per chip) + 2 GB LPDDR4 | per-core synaptic memory, configurable precision |
| Neurons | py-spinnaker2: IF, LIF, CuBa-LIF; reset by subtraction or to zero | programmable microcode neurons |
| Weights | py-spinnaker2: 8-bit signed weights, float32 neuron state (reported) | configurable; e.g. 8-bit weights in reported studies |
| Learning | software on the Arm cores; no high-level learning API confirmed | programmable synaptic learning with traces and third factors |
| Timestep | 1 ms commonly used | — |
| Interchange | NIR (Neuromorphic Intermediate Representation) supported in reported work | NIR supported |

Sources: arXiv 2312.09084, 2401.04491, 2607.24396 (SpiNNaker2 architecture); arXiv
2406.17049 and py-spinnaker2 documentation (neuron models, 8-bit weights); arXiv
2311.14641 (NIR; Loihi 2 figures), 2303.09503 and 2511.01553 (Loihi 2 learning).

## Open questions for the partner

- Neuron model and parameter format on the target (`alpha` vs `beta`, fixed-point
  state?), supported `dt`, maximum `T` per sample.
- Weight precision and layout; whether learning can update 8-bit weights directly
  (rounding scheme) and how traces are stored.
- How a broadcast modulatory signal (error/reward) reaches cores, and its latency.
- Neurons and synapses per core in practice; routing limits on fan-in/fan-out.
- Spike input path (host streaming vs on-chip generation) and readout path.
- Energy-measurement setup: what is measured (chip, board), sampling, idle baseline.
- Whether NIR export is the preferred route instead of (or besides) our own export.

Simulation results contain proxies with assumptions. Measured `energy_j` is accepted
only from a hardware backend with measurement source, method and boundary metadata.
