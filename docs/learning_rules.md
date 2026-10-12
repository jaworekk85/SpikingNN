# Learning rules

See `research_framework.md` for the overall contract. Rules follow the
`LearningRule` protocol (`snn_research.learning_rules.base`):

```text
declaration: RuleDeclaration   uses_autograd, locality, online_capable, state, modulator_source
init_state(model)              bind target parameters (model is used for binding only)
step(model, layer_states, modulators) -> updates
apply(updates)                 commit updates to the bound parameters
reset()                        clear rule state, never learned weights
```

Local numerical kernels receive a `LocalObservation` with only `pre` (presynaptic
activity), `post` (postsynaptic spikes) and `post_potential` (membrane potential before
reset), each `[T, batch, n]`. They never receive the model, other layers or gradients.
The offline protocol passes two modulators: `loss` (with a graph only if a rule uses
autograd) and `error = onehot(label) - softmax(logits)` (detached). Updates are
averaged over the minibatch; per-sample traces start at zero.

## Implemented rules

| Rule | Locality | Autograd | Modulator | Targets |
| --- | --- | --- | --- | --- |
| `SurrogateBPTT` | `global_gradient` | yes | `error` (loss) | all parameters (Adam) |
| `STDP` | `synapse_local` | no | `none` | dense hidden synapses |
| `ThreeFactor` | `global_modulated` | no | `error` | dense hidden synapses + readout |

**STDP** (pair-based, exponential traces): a pre spike `d >= 0` steps before a post
spike gives `+a_plus exp(-d dt/tau_plus)`; a post spike `d > 0` steps before a pre
spike gives `-a_minus exp(-d dt/tau_minus)`. Weights are clipped to `[-w_max, w_max]`.
STDP is unsupervised, so the reference `stdp` setting pairs it with `ThreeFactor`
on the readout only (`train_hidden=False`).

**ThreeFactor** (error-modulated eligibility rule): presynaptic trace
`x_i(t) = beta_pre x_i(t-1) + pre_i(t)`, pseudo-derivative
`psi_j(t) = 1 / (1 + k |u_j(t) - threshold|)^2`, eligibility
`e_ij(t) = lambda_e e_ij(t-1) + psi_j(t) x_i(t)`, update
`dW_ij = lr * M_j * mean_t e_ij(t)`. The readout uses `M = error` with eligibility
equal to its input rate (the delta rule, `readout_lr`). Hidden layers receive the error
through fixed random feedback matrices (`M = error @ B^T`), not transported weights or
backpropagated gradients. The modulator is computed once per sample (at its end), so
updates are per sample/minibatch, not per timestep.

Neither local rule updates biases. Convolutional synapses stay fixed under local rules.

**Bounded precision (optional, Phase 4).** Both local rules accept `weight_bits` and a
bound (`w_max`): bound weights are clipped to `[-w_max, w_max]` and rounded to the
nearest of `2**bits - 1` symmetric fixed-point levels at binding and after every update
(`learning_rules/precision.py`). Rounding is deterministic round-to-nearest, so updates
below half a step are lost; stochastic rounding and bounded-precision traces are not
implemented. `ThreeFactor` needs `w_max > 0` to quantize (default: unbounded).

## Verified properties (tests/test_learning_rules.py)

- STDP window sign and exponential shape on toy spike pairs.
- Exponential eligibility-trace decay.
- Zero modulator gives a zero three-factor update.
- Minibatch update equals the mean of per-sample updates (both rules).
- Local rules build no autograd graph and populate no `.grad`, even when the forward
  pass itself was run with gradients enabled.
- Local kernels accept only `LocalObservation` (no model argument).

Rules are described by their declared, tested properties, not as biologically
plausible. Predictive/local and learned-plasticity rules remain future options.
