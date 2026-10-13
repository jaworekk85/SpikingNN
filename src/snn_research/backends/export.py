"""Hardware-neutral export of a trained model and its learning rule(s).

``export_network`` describes everything a hardware port must reproduce: timing, the
exact neuron equations and parameters, layer shapes, logical synapse counts and
fan-in/fan-out, the readout, and each rule's declaration, parameters, per-synapse /
per-neuron state and modulator. ``save_export`` writes ``export.json`` plus
``weights.npz`` (float32 arrays keyed by parameter name). Nothing here targets a
specific chip; mapping onto hardware is a backend's job.
"""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

import snn_research

from ..encoding import EncodingConfig
from ..metrics import fan_out

FORMAT = "snn_research.export/1"

# State each rule keeps, by where it lives; used to estimate on-chip memory.
RULE_STATE = {
    "surrogate_bptt": {"note": "needs stored activity of all T steps and a backward pass; not an on-chip rule"},
    "stdp": {"per_presynaptic_neuron": ["pre_trace"], "per_postsynaptic_neuron": ["post_trace"]},
    "three_factor": {
        "per_synapse": ["eligibility_trace"],
        "per_presynaptic_neuron": ["pre_trace"],
        "per_postsynaptic_neuron": ["feedback_weights (n_classes values)"],
    },
}


def _synapse(layer: nn.Module) -> nn.Module:
    return next(m for m in layer.modules() if isinstance(m, (nn.Linear, nn.Conv2d)))


def _fan_in(synapse: nn.Module, input_shape: tuple[int, ...]) -> int:
    """Largest number of presynaptic connections onto one postsynaptic neuron."""
    if isinstance(synapse, nn.Linear):
        return synapse.in_features
    ones = torch.ones(1, *input_shape)
    kernel = torch.ones(1, synapse.in_channels, *synapse.kernel_size)
    return int(F.conv2d(ones, kernel, stride=synapse.stride, padding=synapse.padding).max())


def _plain(value: Any) -> Any:
    return asdict(value) if is_dataclass(value) else value


def export_network(model: nn.Module, rules: Sequence, encoding: EncodingConfig,
                   input_shape: tuple[int, ...]) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Return the JSON description and the float32 weights of a ``SpikingNet``."""
    p = model.lif.params
    layers, weights = [], {}
    shape = tuple(input_shape)
    for i, layer in enumerate(model.layers):
        synapse = _synapse(layer)
        with torch.no_grad():
            out_shape = tuple(layer(torch.zeros(1, *shape)).shape[1:])
        fan = fan_out(layer, shape)
        layers.append({
            "name": f"layer{i}",
            "type": "dense" if isinstance(synapse, nn.Linear) else "conv2d",
            "input_shape": list(shape),
            "output_shape": list(out_shape),
            "flatten_input": any(isinstance(m, nn.Flatten) for m in layer.modules()),
            "neurons": int(np.prod(out_shape)),
            "synapses": int(fan.sum()),
            "max_fan_in": _fan_in(synapse, shape if isinstance(synapse, nn.Conv2d) else (int(np.prod(shape)),)),
            "max_fan_out": int(fan.max()),
            **({"kernel_size": list(synapse.kernel_size), "stride": list(synapse.stride),
                "padding": list(synapse.padding)} if isinstance(synapse, nn.Conv2d) else {}),
            "weight": f"layer{i}.weight",
            "bias": f"layer{i}.bias" if synapse.bias is not None else None,
        })
        weights[f"layer{i}.weight"] = synapse.weight.detach().numpy().astype(np.float32)
        if synapse.bias is not None:
            weights[f"layer{i}.bias"] = synapse.bias.detach().numpy().astype(np.float32)
        shape = out_shape
    readout = model.readout
    weights["readout.weight"] = readout.weight.detach().numpy().astype(np.float32)
    weights["readout.bias"] = readout.bias.detach().numpy().astype(np.float32)

    description = {
        "format": FORMAT,
        "framework_version": snn_research.__version__,
        "timing": {"dt_ms": p.dt_ms, "T": encoding.T},
        "encoding": {"kind": encoding.kind, "input_is_spikes": encoding.kind != "direct",
                     "input_shape": list(input_shape)},
        "neuron": {
            "model": "LIF",
            "equations": [
                "u[t] = beta * v[t-1] + (1 - beta) * I[t]   (I[t] = W x[t] + b; not integrated while refractory)",
                "z[t] = 1 if u[t] >= threshold else 0",
                "v[t] = u[t] - z[t] * threshold   (reset=subtract)   |   v[t] = u[t] * (1 - z[t])   (reset=zero)",
            ],
            "tau_mem_ms": p.tau_mem_ms, "beta": p.beta, "input_scale": 1 - p.beta,
            "threshold": p.threshold, "reset": p.reset, "refractory_steps": p.refractory_steps,
            "initial_state": "v = 0 at the start of every sample",
        },
        "layers": layers,
        "readout": {
            "type": "non-spiking rate readout",
            "equation": "logits = W_out (mean over T of last hidden spikes) + b_out; prediction = argmax",
            "inputs": layers[-1]["neurons"], "classes": readout.out_features,
            "synapses": layers[-1]["neurons"] * readout.out_features,
            "weight": "readout.weight", "bias": "readout.bias",
        },
        "rules": [{
            "declaration": asdict(rule.declaration),
            "parameters": _plain(rule.config),
            "state": RULE_STATE.get(rule.declaration.name, {}),
            "targets": (sorted(rule.synapses) if hasattr(rule, "synapses") else ["all parameters"])
                       + (["readout"] if hasattr(rule, "readout") else []),
        } for rule in rules],
        "totals": {
            "neurons": sum(layer["neurons"] for layer in layers),
            "synapses": sum(layer["synapses"] for layer in layers) + layers[-1]["neurons"] * readout.out_features,
            "stored_weights": int(sum(w.size for k, w in weights.items() if k.endswith("weight"))),
            "max_fan_in": max(layer["max_fan_in"] for layer in layers),
        },
    }
    return description, weights


def save_export(description: dict[str, Any], weights: dict[str, np.ndarray], directory: str | Path) -> Path:
    """Write ``export.json`` and ``weights.npz`` into ``directory``; never overwrite."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / "export.json"
    if target.exists() or (directory / "weights.npz").exists():
        raise FileExistsError(f"an export already exists in {directory}")
    np.savez(directory / "weights.npz", **weights)
    target.write_text(json.dumps(description, indent=2), encoding="utf-8")
    return target
