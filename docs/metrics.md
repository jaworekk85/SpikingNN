# Metrics — definitions and checks

Implemented (`snn_research.metrics`, `energy`, `protocols`): accuracy, spike rate and
temporal sparsity (Phase 1); shift-phase quality, recovery, forgetting and adaptation
cost (Phase 3); synaptic operations, input MACs, synapse counts and energy proxies
(Phase 4). Measured energy needs a hardware backend and is never produced here.

Phase 3 conventions:

- Prequential quality is per-sample correctness recorded *before* the model learns
  from that sample. `before / during / after` are means over the shift schedule's
  phases (`during` = `[start, end)`, the adaptation window).
- Recovery = samples after shift onset until a trailing window of `window` samples,
  lying fully after onset, first reaches `q_target`; `None` if never. Its resolution is
  bounded below by `window` (the smallest reportable value).
- Forgetting = `max - final` over a task's history, from when it was first learned.
  In the reference run the clean test accuracy before and after the stream forms
  that history (two checkpoints).
- Adaptation cost inside `[start, end)`, each a separate field: update steps
  (minibatches), weights written by rules, spikes, synaptic operations, input MACs,
  and an adaptation energy proxy with its components.

Phase 4 conventions (`metrics/synops.py`, `energy/proxy.py`):

- Synaptic operation = one spike delivered over one existing connection; each
  presynaptic spike counts its fan-out, summed over layers, timesteps and samples.
- Dense fan-out = `out_features`. Convolutional fan-out of an input position = number
  of (output channel, output position) pairs whose receptive field contains it. Padding
  positions are not neurons and emit nothing. Weight sharing does not reduce counts:
  shared weights still drive separate logical connections. The synapse count of a conv
  layer is the sum of these fan-outs (logical connections, not stored weights).
- The readout accumulates the last hidden layer's spikes: `n_classes` operations per
  spike. Its time-averaging/division is not counted.
- Direct-current input is not spikes: the first layer then counts `input_macs` (one per
  non-zero input value, target and timestep), never mixed into synaptic operations.
- Neuron updates = neurons x `T` per sample. Weights written = elements of all updates
  returned by rules (dense count, including zero-valued updates).
- Energy proxy (pJ) = synops x `pj_per_synop` + MACs x `pj_per_mac` + neuron updates x
  `pj_per_neuron_update` + weights written x `pj_per_weight_update`. Defaults 0.9 / 4.6 /
  0 / 0 pJ: 45 nm FP32 arithmetic (Horowitz, ISSCC 2014; Han et al. 2015), as widely
  reused in SNN papers. They exclude memory access, data movement, I/O, idle power and
  implementation overhead, which often dominate on real hardware. Coefficients and
  their source are saved in `proxy_assumptions` with `kind: proxy (not measured)`.

| Quantity | Definition / requirement |
| --- | --- |
| Task quality | Explicit configured score, initially classification accuracy |
| Spike rate | Spikes / (neurons × timesteps), per layer and overall; include batch count in the denominator |
| Temporal sparsity | Inactive neuron–time pairs / all observed neuron–time pairs |
| Synaptic operations | For each emitted input spike, count connected outgoing synaptic events; sum across layers and timesteps |
| Forgetting | Maximum observed quality on task `a` minus its final quality |
| Recovery | First post-shift time/index at which quality reaches configured `Q_target`; record unrecovered explicitly |
| Adaptation cost | Separate update count, synaptic operations and spike activity over an explicit adaptation interval |
| Energy proxy | Estimated activity/operation cost with saved coefficients, units and assumptions |
| Measured energy | Hardware-sourced joules with measurement metadata |

For a dense layer, fan-out is the number of connected postsynaptic neurons. Masked
connections use actual fan-out. Counts represent logical spike-triggered operations,
not dense GPU multiply-add counts or runtime. Convolutional padding and shared-weight
accounting follow the Phase 4 conventions above. Event count and neuron count are
separate from number of synapses.

Aggregate rate by total spikes divided by total neuron–time opportunities; do not
average differently sized layers without weighting. Current-input operations (MACs)
and learning-update operations (weights written) are counted separately, as above.

Report pre/during/post quality for every shift, with explicit windows. Forgetting
and adaptation are separate metrics. Energy proxies cannot be described as measured
energy. Comparisons require matching `T`, neuron/synapse counts, encoding and protocol,
or explicit mismatch disclosure (`reporting.check_comparable`; `allow_mismatch=True`
lists the mismatches in the report). Reports give mean ± sample standard deviation
over seeds and count unavailable values (e.g. unrecovered) instead of dropping them
silently.
