# Learning rules — proposed contract

Implemented in Phase 1: `RuleDeclaration` (validated declared properties) and
`SurrogateBPTT` (`uses_autograd=True`, `locality=global_gradient`, `modulator_source=error`;
the loss is passed as the `"loss"` modulator). See `research_framework.md` for interfaces.

Rules declare `uses_autograd`, `locality`, `online_capable`, maintained state and
`modulator_source`. Local kernels receive only their declared synaptic/neuron
quantities. The orchestration layer binds parameters; it must not expose arbitrary
model internals to local kernels.

Planned order: surrogate BPTT in Phase 1; STDP and one eligibility-based three-factor
rule in Phase 2. Local implementations use `torch.no_grad`; tests must additionally
detect built graphs or populated gradients. A zero modulator must produce zero
three-factor update, and eligibility decay must be tested independently.

Online and minibatch semantics, trace reset policy and modulator provenance must
be documented per implementation. Describe rules by verified properties, without
assuming biological plausibility. Predictive/local and learned plasticity remain
future options, not implementations in this skeleton.
