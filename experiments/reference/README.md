# Reference experiment (pipeline validation)

Validates the whole pipeline end to end; its numbers are not research results.

- `pipeline.py` — `ReferenceConfig` and one seed's run: offline pretraining ->
  evaluation (clean and with the shift at full strength) -> prequential stream with one
  scheduled shift, adapting online -> evaluation again (forgetting) -> synaptic
  operations and energy proxy (inference and, separately, adaptation) -> `RunRecord`.
- `run.py` — CLI: `--rule surrogate_bptt | three_factor | stdp | all`, `--report`,
  `--export`, `--hardware <yaml>`.
- `report.py` — multi-seed comparison of saved runs (tables + figure); refuses runs
  with different `T`, encoding, model size or protocol unless `--allow-mismatch`.
- `tune.py` — grid search on a seeded `tune` split carved from the training data,
  tuning seeds distinct from evaluation seeds; objective = mean of tune accuracy after
  pretraining and shifted-tune accuracy after the stream. The test split is discarded.

Configs: `reference_tiny.yaml` (synthetic, seconds), `reference_mnist.yaml`,
`reference_nmnist.yaml`. For N-MNIST, `encoding` must be `events` with `T` equal to the
number of bins and `neuron.dt_ms` equal to the bin width; the config refuses otherwise.
