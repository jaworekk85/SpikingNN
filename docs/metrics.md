# Metrics — definitions and planned checks

Implemented (`snn_research.metrics`, `protocols`): accuracy, spike rate and temporal
sparsity (Phase 1); shift-phase quality, recovery, forgetting and adaptation cost
without synaptic operations (Phase 3). Synaptic operations and energy proxies are
Phase 4; measured energy needs a hardware backend.

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
- Adaptation cost inside `[start, end)`: number of update steps (minibatches) and total
  spikes, reported as separate fields; synaptic operations will be added in Phase 4.

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
accounting must be specified before implementation. Event count and neuron count
are separate from number of synapses.

Aggregate rate by total spikes divided by total neuron–time opportunities; do not
average differently sized layers without weighting. Current-input operations and
learning-update operations need their own documented treatment.

Report pre/during/post quality for every shift, with explicit windows. Forgetting
and adaptation are separate metrics. Energy proxies cannot be described as measured
energy. Comparisons require matching `T`, neuron/synapse counts, encoding and protocol,
or explicit mismatch disclosure.
