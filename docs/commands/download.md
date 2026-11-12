# Download resources

Download the general PartiFlow checkpoint and its autoencoder:

```bash
partiflow download partiflow autoencoder
partiflow check
```

PartiFlow checkpoints and the raw PepInter archive come from [MegaGatlingPea/PartiFlow](https://huggingface.co/MegaGatlingPea/PartiFlow). AF2 and MPNN use the sources listed below.

## Select targets

```bash
partiflow download --list
partiflow download partiflow-aux partiflow-dock partiflow-ss
partiflow download af2 proteinmpnn solublempnn
```

Targets are `partiflow`, `partiflow-aux`, `partiflow-dock`, `partiflow-ss`, `autoencoder`, `af2`, `proteinmpnn`, `solublempnn`, and `pepinter`. Every target has a default source. `--list` is read-only; `--all` downloads all nine targets, including the PepInter archive.

In a terminal, run `partiflow download` without targets to open the resource selector. Use Up/Down to browse, Space to select, and Enter to download. Tab moves to the Download and Back buttons. Esc or Ctrl-D returns without downloading; Ctrl-C exits with status 130. An empty selection stays in the selector.

The presence tick is separate from the selection checkbox. Existing complete downloads are skipped unless `--force` is supplied. With redirected input/output or `TERM=dumb`, provide target names, `--all`, or `--list` explicitly. `NO_COLOR` uses a monochrome selector and plain reports.

## Download PepInter

```bash
partiflow download pepinter --data-dir train_data/pepinter
python -m partiflow.training.prepare_data \
  --archive train_data/pepinter/pepinter.tar.xz --root train_data/pepinter
```

The download command saves `DATA_DIR/pepinter.tar.xz`. It does not extract the archive or build training graphs. Preprocessing is a separate step and covers all 171,019 source structures by default. It preserves the train, validation, and test splits, mapping `valid` to `val`. See [training data](../resources.md#prepare-training-data) before starting a training run.

`--data-dir` selects the dataset root. Alternatively, `--config` composes a training YAML, including its Hydra defaults, and resolves the destination using the training path rules. Explicit `--data-dir` takes precedence. A downloaded archive does not satisfy the split-index and processed-graph checks used by `partiflow check`.

## Sources and destinations

AF2 uses the [official AlphaFold parameter archive](https://storage.googleapis.com/alphafold/alphafold_params_2022-12-06.tar). ProteinMPNN uses the pinned [official weight directory](https://github.com/dauparas/ProteinMPNN/tree/8907e6671bfbfc92303b5f79c4b5e6ce47cdef57/vanilla_model_weights), and SolubleMPNN uses the [IPD distribution](https://files.ipd.uw.edu/pub/ligandmpnn/solublempnn_v_48_020.pt).

`--model-dir` selects the [model root](../resources.md#choose-a-model-directory). For example:

```bash
partiflow download partiflow autoencoder --model-dir models
partiflow check --model-dir models
```

Use `--url URL` to override the HTTP or HTTPS source for exactly one target. If the pinned GitHub host is unavailable, the vanilla checkpoint is also available from IPD:

```bash
partiflow download proteinmpnn \
  --url https://files.ipd.uw.edu/pub/ligandmpnn/proteinmpnn_v_48_020.pt --model-dir models
```

Failed or incomplete transfers do not replace an existing destination. The command continues through independent targets and reports each outcome. Terminal output shows selected targets, progress by completed target, and a result table; redirected output contains one JSON record per target.
