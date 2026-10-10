# Configuration layout

Phase 0 directories only. YAML configurations are introduced alongside their
implementations and validated using typed dataclasses. There is no config loader yet.

`experiments/` will hold complete experiment configurations; the other directories
will hold component examples. Prefer explicit composition in plain Python over
configuration inheritance or registries. Always save a fully resolved run config.

Tiny configurations must default to CPU and synthetic data. Comparisons declare
multiple seeds; individual smoke checks may use one. No experiment is runnable yet.
