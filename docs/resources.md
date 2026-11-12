# Installation and resources

## Create the runtime environment

Run from the source checkout root. The [environment recipe](../partiflow.yaml) targets Linux x86_64 and includes Python 3.11.13, Torch 2.6.0 with CUDA 12.4, JAX/JAXLIB 0.4.29, OpenMM 8.2.0, and USalign 2024.07.30.

PyRosetta is licensed separately. Obtain `pyrosetta-2025.51+release.612b6ef9e9-cp311-cp311-linux_x86_64.whl` under its license and put it in `.cache/packages/`. The environment recipe finds it there; it does not download PyRosetta for you.

```bash
conda env create -f partiflow.yaml
conda activate partiflow-official
partiflow --help
```

The recipe installs the checkout with `-e .`. The command-line package alone does not install the full training, AF2, and molecular-mechanics runtimes. Generation uses the configured accelerator; the bundled design settings request one GPU with mixed bfloat16 precision.

## Prepare the checkpoints

For a first denovo or motif design, import `partiflow.ckpt` and `autoencoder.ckpt` from your local `weights/` directory:

```bash
partiflow download partiflow autoencoder --from-local weights
partiflow check
```

Import other PartiFlow checkpoints when needed:

```bash
partiflow download partiflow-aux partiflow-ss --from-local weights
```

The general model serves denovo and motif tasks; the auxiliary model serves docking, inverse folding, and packing. Use `partiflow-ss.ckpt` for secondary-structure, scaffold-topology, and disulfide conditioning. `partiflow-dock.ckpt` is the specialist used for the CPSet-separated docking comparison. Each uses the same autoencoder. The design command keeps the default general checkpoint until you select another with `--ckpt`.

PartiFlow checkpoints and PepInter have no configured public download URL. Use `--from-local` or provide your own `--url`. Configured AF2 and MPNN sources are available through:

```bash
partiflow download af2 proteinmpnn solublempnn
partiflow check --model af2 --model proteinmpnn --model solublempnn
```

`download --list` shows target names, local status, and download availability. AF2 reward needs Multimer v3 parameters; paired validation uses pTM parameters. The AF2 target prepares models 1–5 in original, pTM, and Multimer v3 forms. Use [check --load](commands/check.md) when you want to test actual CPU loading after the files are present.

## Choose a model directory

The model root comes from `--model-dir`, then `PARTIFLOW_MODEL_DIR`. Without either, a source installation uses the checkout root and a wheel installation uses the current working directory. Configuration files, example structures, and small chemical resources come from the package separately.

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
```

Training data is configured separately. Moving weights with `--model-dir` does not move the dataset used by training.

For example, to keep weights in a separate `models/` directory:

```bash
partiflow download partiflow autoencoder --from-local weights --model-dir models
partiflow check --model-dir models
partiflow design configs/tasks/example.yaml --task klk2_hotspot --model-dir models \
  --out inference/klk2
```

An explicit `design --ckpt` path is relative to the directory where you run the command. With this layout, select the scaffold checkpoint as `--ckpt models/ckpts/partiflow-ss.ckpt`.

## Install a wheel

Use a prepared Linux x86_64 environment with the same runtime dependencies and licensed PyRosetta installation. The bundled DAlphaBall executable requires glibc 2.38 or newer. Build on Linux x86_64, then install the resulting platform-specific wheel:

```bash
python -m build --wheel
python -m pip install --no-deps dist/partiflow-0.1.0-py3-none-linux_x86_64.whl
```

The wheel includes the configurations, example PDBs, scaffold libraries, chemical resources, and workflow code. Weights and training datasets remain external. From another working directory, locate the installed examples with:

```bash
PARTIFLOW_RESOURCES=$(python -c 'from partiflow.paths import resource_root; print(resource_root())')
partiflow design "$PARTIFLOW_RESOURCES/configs/tasks/example.yaml" --task klk2_hotspot \
  --model-dir models --out inference/klk2
partiflow train fm --config "$PARTIFLOW_RESOURCES/configs/training.yaml" \
  --model-dir models --show-config
```

Bundled tasks resolve their PDB and scaffold-library paths inside the resource tree. When writing your own task file, use paths relative to the directory where you invoke PartiFlow, or use absolute paths. If you copy a bundled task into your working directory, update its paths before running it. For the README's one-sample task, copy the bundled PDB to `inputs/klk2.pdb` and set `pdb_path: ./inputs/klk2.pdb`:

```bash
mkdir -p inputs
cp "$PARTIFLOW_RESOURCES/pdbs/design/klk2/klk2_motif.pdb" inputs/klk2.pdb
```

## Prepare training data

A local PepInter dataset needs `indexes/split.csv` and nonempty `processed/` contents. Import it into the default layout or choose a separate data directory:

```bash
partiflow download pepinter --from-local datasets/pepinter --data-dir train_data/pepinter
partiflow check --data-dir train_data/pepinter
```

The import command expects a prepared dataset. For the raw PepInter archive, extract it and build atom37 graphs first:

```bash
python -m partiflow.training.prepare_data --archive train_data/archives/pepinter.tar.xz \
  --root train_data/pepinter
```

For an already extracted `train_data/pepinter/raw/` tree, replace `--archive ...` with `--skip-extract`. Without `--subset`, preprocessing covers the full source splits. The regular training configs use the full split index and need all indexed graphs.

`--config` on `download` and `check` composes the training YAML, including Hydra defaults, and uses its resolved `datamodule.processed_dir` to locate the dataset root. A top-level `data_dir` takes precedence; explicit `--data-dir` overrides both. These resource commands require `indexes/` and `processed/` under the selected data root. For another training layout, import to an explicit standard-layout directory and point the training config at the imported files. Confirm the split fields and preprocessing settings before [training](commands/train.md).

In training configurations, relative pretrained checkpoint and autoencoder paths use the model root. Other relative paths in a bundled config use the checkout root for source installations and the invocation directory for wheel installations. Other relative paths in a user training config use that config file's directory.
