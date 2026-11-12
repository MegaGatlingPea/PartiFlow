# Download or import resources

List targets and their current locations before acquiring them:

```bash
partiflow download --list
```

Targets are `partiflow`, `partiflow-aux`, `partiflow-dock`, `partiflow-ss`, `autoencoder`, `af2`, `proteinmpnn`, `solublempnn`, and `pepinter`. `--list` is read-only. `--all` selects every target, including those needing a local source.

## Import local files

```bash
partiflow download partiflow autoencoder --from-local weights
partiflow download partiflow-ss --from-local weights/partiflow-ss.ckpt
partiflow download af2 --from-local weights/alphafold_parameters.tar
partiflow download pepinter --from-local datasets/pepinter --data-dir train_data/pepinter
```

`--from-local` accepts a matching file, a directory containing the assets, or a supported archive. Checkpoint containers are copied intact. AF2 requires all configured arrays in its directory or archive. PepInter requires `indexes/split.csv` and nonempty `processed/` contents.

PartiFlow checkpoints and PepInter do not have configured public URLs. Import them locally or supply an explicit source. Without either, the target reports `unconfigured` and the command returns a nonzero status.

## Use configured downloads

```bash
partiflow download af2 proteinmpnn solublempnn --model-dir models
```

AF2 uses the [official AlphaFold parameter archive](https://storage.googleapis.com/alphafold/alphafold_params_2022-12-06.tar). ProteinMPNN uses the pinned [official weight directory](https://github.com/dauparas/ProteinMPNN/tree/8907e6671bfbfc92303b5f79c4b5e6ce47cdef57/vanilla_model_weights), and SolubleMPNN uses the [IPD distribution](https://files.ipd.uw.edu/pub/ligandmpnn/solublempnn_v_48_020.pt).

`--url URL` accepts HTTP, HTTPS, or a local `file:` URL for exactly one target. If the pinned GitHub host is unavailable, the vanilla checkpoint is also available from IPD:

```bash
partiflow download proteinmpnn \
  --url https://files.ipd.uw.edu/pub/ligandmpnn/proteinmpnn_v_48_020.pt --model-dir models
```

Existing complete targets are skipped unless `--force` is supplied. Failed or incomplete transfers do not replace an existing destination. The command continues through independent targets and reports each outcome.

`--model-dir` selects the [model root](../resources.md#choose-a-model-directory). For PepInter, `--data-dir` selects the destination. Alternatively, `--config` composes a training YAML, including its Hydra defaults, and resolves the dataset path using the training path rules. Imports use the standard `indexes/` and `processed/` layout; see [training data](../resources.md#prepare-training-data). Terminal output shows progress by processed target and a result table; redirected output contains one JSON record per target.
