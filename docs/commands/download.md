# Downloads and imports

```bash
partiflow download --list
partiflow download partiflow autoencoder --from-local /path/to/ckpts --model-dir /path/to/models
partiflow download af2 --from-local /path/to/af2_parameters.tar --model-dir /path/to/models
partiflow download proteinmpnn solublempnn --model-dir /path/to/models
partiflow download pepinter --from-local /path/to/pepinter --data-dir /path/to/data/pepinter
```

Targets are `partiflow`, `partiflow-aux`, `partiflow-dock`, `partiflow-ss`, `autoencoder`, `af2`, `proteinmpnn`, `solublempnn`, and `pepinter`. `--all` selects every target; `--list` reports configured sources and current files without changing them. `--force` replaces an existing destination after successful acquisition.

`--from-local` accepts a matching checkpoint, a directory containing the target assets, or a supported archive. PyTorch checkpoint containers are copied intact. `--url URL` supplies an explicit source for exactly one target. Missing published URLs produce `unconfigured`; local import remains available. Files are staged before publication, and archive paths, links, and special-file members are validated before extraction.

AF2 uses the [official AlphaFold parameter archive](https://storage.googleapis.com/alphafold/alphafold_params_2022-12-06.tar). Vanilla ProteinMPNN uses the pinned [official ProteinMPNN weight directory](https://github.com/dauparas/ProteinMPNN/tree/8907e6671bfbfc92303b5f79c4b5e6ce47cdef57/vanilla_model_weights); SolubleMPNN uses the [IPD checkpoint distribution](https://files.ipd.uw.edu/pub/ligandmpnn/solublempnn_v_48_020.pt).

`--model-dir` selects the root described in the [resource layout](../resources.md). For PepInter, `--data-dir` selects the dataset root, or `--config` reads its dataset path. A local dataset needs `indexes/split.csv` and the processed structure files under `processed/`.


When the pinned GitHub host is unreachable, the same vanilla weights are also
available from the official [IPD distribution](https://files.ipd.uw.edu/pub/ligandmpnn/proteinmpnn_v_48_020.pt):

```bash
partiflow download proteinmpnn --url https://files.ipd.uw.edu/pub/ligandmpnn/proteinmpnn_v_48_020.pt --model-dir /path/to/models
```

A failed or incomplete transfer does not replace an existing destination.
