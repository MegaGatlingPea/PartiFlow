# Resource and execution checks

```bash
partiflow check
partiflow check configs/tasks/example.yaml --model partiflow --model autoencoder
partiflow check --all --model-dir /path/to/models
partiflow check --model proteinmpnn --load
partiflow check configs/tasks/example.yaml --model partiflow --smoke
```

Default checks are read-only: they inspect packaged resources, requested checkpoint paths, and optional basic task schema/input paths. They do not import model implementations, initialize accelerators, download assets, or run inference. `--model NAME` is repeatable and mutually exclusive with `--all`. Without a selection, the general PartiFlow model and autoencoder are checked.

`--load` explicitly constructs the requested model on CPU and exercises its real weight loader. Merely reading a checkpoint container is not counted as loading a model. `--smoke` explicitly runs the controlled generation/validation path; a missing implementation, asset, or execution failure reports failure. Loading and smoke execution are mutually exclusive. Passing path checks alone does not establish either capability.

`--model-dir` overrides the shared checkpoint root. `--data-dir` or `--config` can additionally check a local PepInter dataset. Task checks are labeled as basic schema/input validation; generation performs the complete task compilation.
