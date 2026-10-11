# Configuration layout

YAML configurations are loaded into frozen typed dataclasses by
`snn_research.utils.load_config`; unknown keys and invalid values are rejected, and
the fully resolved config (including defaults) is saved with every run.

`experiments/` will hold complete experiment configurations; the other directories
will hold component examples. Prefer explicit composition in plain Python over
configuration inheritance or registries. Always save a fully resolved run config.

Tiny configurations must default to CPU and synthetic data. Comparisons declare
multiple seeds; individual smoke checks may use one. Component directories stay
empty until a component needs a standalone example.
