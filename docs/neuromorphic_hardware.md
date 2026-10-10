# Neuromorphic hardware boundary

Phase 0 placeholder and assumptions. No hardware backend, exporter or energy
measurement is implemented. Development starts with ordinary CPU/GPU simulation.

Future hardware boundary: `export(model, rule)`, `run(...)`, `measure(...)`.
Hardware-specific code stays in `backends/`. Local rules must have documented local
state and any broadcast modulators. Export readiness includes network dimensions,
fan-in/fan-out, timesteps, time constants, reset/refractory semantics, weight/trace
precision and memory needs.

Supported neuron models, programmable plasticity, fixed-point precision, routing,
memory/core, I/O and energy-measurement interfaces are **to be confirmed with the
hardware partner**. Do not assume SpiNNaker2 or Loihi capabilities in the framework.
Phase 5 will elaborate this interface; implementations need an explicit request.

Simulation results contain proxies with assumptions. Measured `energy_j` is accepted
only from a hardware backend with measurement source, method and boundary metadata.
The Jülich pilot and partner-facing document are deferred to Phase 6.
