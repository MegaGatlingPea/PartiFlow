# Download resources

Download the four recommended weights: `autoencoder`, `partiflow`, `partiflow-ss`, and `partiflow-aux`:

```bash
partiflow download
partiflow check
```

PartiFlow checkpoints and the prepared PepInter subset come from [MegaGatlingPea/PartiFlow](https://huggingface.co/MegaGatlingPea/PartiFlow). AF2 and MPNN use the sources listed below.

## Select targets

```bash
partiflow download --list
partiflow download partiflow-aux partiflow-dock partiflow-ss
partiflow download af2 proteinmpnn solublempnn
```

Targets are `partiflow`, `partiflow-aux`, `partiflow-dock`, `partiflow-ss`, `autoencoder`, `af2`, `proteinmpnn`, `solublempnn`, and `pepinter`. Every target has a default source. `--list` is read-only; `--all` downloads all nine targets, including the prepared PepInter subset.

With no targets, a terminal opens the resource selector with those four weights preselected; noninteractive use downloads the four directly. Explicit target names replace the default set. `partiflow-dock` remains optional. Use Up/Down to browse, Space to select, and Enter to download. Tab moves to the Download and Back buttons. Esc or Ctrl-D returns without downloading; Ctrl-C exits with status 130. An empty selection stays in the selector.

The presence tick is separate from the selection checkbox. Existing complete downloads are skipped unless `--force` is supplied. With redirected input/output or `TERM=dumb`, the bare command downloads the four default weights directly. Use target names, `--all`, or `--list` for another selection. `NO_COLOR` uses a monochrome selector and plain reports.

## Download the PepInter subset

```bash
partiflow download pepinter
partiflow check --data-dir train_data/pepinter_subset
```

This target downloads and unpacks `pepinter-subset.tar.xz`. The default destination is `train_data/pepinter_subset/`, containing 64 training and 16 validation PDB structures, their processed graphs, and `indexes/split.csv`. The archive is retained as `DATA_DIR/pepinter-subset.tar.xz`. No graph preprocessing step is needed for this download.

The samples retain their original train/validation assignments and were selected by sorted ID for examples. They do not include a test set. The full training dataset is not included, and training on this subset does not reproduce the published models.

`--data-dir` selects another dataset root. Alternatively, `--config` composes a training YAML, including Hydra defaults, and resolves the destination using the training path rules. Explicit `--data-dir` takes precedence. If you change the destination, point the training configuration at its `indexes/split.csv` and `processed/` directories. See [training data](../resources.md#prepare-training-data) for the layout.

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
