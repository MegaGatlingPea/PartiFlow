# Installation and local resources

Use `partiflow.yaml` to create the independent `partiflow-official` environment from the checkout root. Its Linux runtime pins include Python 3.11.13, NumPy 1.26.4, Torch 2.6.0 with CUDA 12.4, JAX/JAXLIB and the CUDA 12 plugin 0.4.29, CuDNN 9.1.0.70, OpenMM 8.2.0, and PyRosetta `2025.51+release.612b6ef9e9`. USalign 2024.07.30 is installed from Bioconda. The complete package list is in [partiflow.yaml](../partiflow.yaml).

PyRosetta must be obtained under its license before creating the environment. Put `pyrosetta-2025.51+release.612b6ef9e9-cp311-cp311-linux_x86_64.whl` in the checkout's `.cache/packages/` directory. The recipe searches this local directory; it does not provide a PyRosetta download. Then run:

```bash
conda env create -f partiflow.yaml
conda activate partiflow-official
python -m partiflow --help
```

The recipe includes `-e .`, which installs the checkout as an editable package. Installing `partiflow` alone does not supply the full training, molecular-mechanics, and AF2 runtimes or model weights. After source edits, `python -m pip install -e . --no-deps` can refresh the editable installation.

## Installing a wheel and working outside the checkout

Build a wheel in the prepared environment, then install it into an environment with the same runtime dependencies. `--no-deps` retains those prepared versions.

```bash
python -m build --wheel
cd /path/to/workdir
python -m pip install --no-deps /path/to/partiflow-0.1.0-py3-none-any.whl
```

The wheel contains code, configurations, example PDBs, small chemical resources, and bundled workflow implementations. Checkpoints and training datasets remain external. Locate the installed resources when using a bundled task or training configuration:

```bash
PARTIFLOW_RESOURCES=$(python -c 'from partiflow.paths import resource_root; print(resource_root())')
partiflow design "$PARTIFLOW_RESOURCES/configs/tasks/example.yaml" --task klk2_hotspot \
  --model-dir /path/to/models --out /path/to/workdir/inference/klk2
partiflow train fm --config "$PARTIFLOW_RESOURCES/configs/training.yaml" \
  --model-dir /path/to/models --show-config
```

A bundled task resolves its bundled PDB paths from the resource tree. Copy task examples into a working directory before editing them and set their `pdb_path` values for that directory: relative paths in a user task file resolve from the invocation directory.

## Model and dataset roots

The model directory is selected in this order: `--model-dir`, `PARTIFLOW_MODEL_DIR`, the source checkout root, or the current working directory for a wheel installation. Configuration and small chemical/example resources come from the package independently of that model directory. `train`, `design`, `validate`, `postprocess`, `check`, and `download` accept `--model-dir`; clustering uses saved structures and an alignment executable.

```text
MODEL_DIR/
  ckpts/
    partiflow.ckpt
    partiflow-aux.ckpt
    partiflow-dock.ckpt
    partiflow-ss.ckpt
    autoencoder.ckpt
  af2/params_model_*.npz
  proteinmpnn/vanilla/v_48_020.pt
  proteinmpnn/soluble/v_48_020.pt
  train_data/pepinter/indexes/split.csv
  train_data/pepinter/processed/
```

The AF2 asset target contains the configured parameter files from the official archive: models 1–5 in original, pTM, and Multimer v3 forms. Standalone validation defaults to pTM model 1; its `af2.models` setting accepts an ordered unique subset of `[1, 2]`. AF2 reward uses Multimer v3 model 1 by default. Vanilla ProteinMPNN and SolubleMPNN use separate checkpoints.

```bash
partiflow download partiflow autoencoder --from-local /path/to/existing/checkpoints --model-dir /path/to/models
partiflow download af2 proteinmpnn solublempnn --model-dir /path/to/models
partiflow check --all --model-dir /path/to/models
```

PartiFlow model weights and PepInter have no configured public download URL in this release. Import available local files or explicitly provide a URL. The downloader reports `unconfigured` when neither is provided. Inference and checks use local assets without downloading them automatically.

PepInter can live outside the model directory: use `download pepinter --data-dir /path/to/pepinter --from-local /path/to/source`, or supply a dataset config containing `datamodule.processed_dir`. A complete local dataset includes `indexes/split.csv` and nonempty `processed/` contents. A training run additionally needs the split-specific fields and preprocessing settings in its dataset config.

For training, relative pretrained model and autoencoder paths resolve from the model root. Other relative paths in a bundled training config resolve from the checkout root in an editable installation, or the invocation directory in a wheel installation. For a user training config, those other paths resolve from the config file's directory. Explicit design `--ckpt` paths resolve from the invocation directory; use an absolute path when selecting a checkpoint outside it.
