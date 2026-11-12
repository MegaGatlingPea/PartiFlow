# PartiFlow

PartiFlow designs peptide binders from receptor structures. It can generate new peptides, grow a scaffold around a fixed motif, dock a known sequence, design a sequence on a fixed backbone, or pack side chains.

[![PartiFlow framework](docs/figures/partiflow-framework.png)](docs/figures/partiflow-framework.png)

## Install and prepare weights

The supplied environment targets Linux x86_64 with Python 3.11 and CUDA 12.4. Obtain the licensed PyRosetta wheel named `pyrosetta-2025.51+release.612b6ef9e9-cp311-cp311-linux_x86_64.whl` and place it in `.cache/packages/` before creating the environment. Run these commands from the checkout root:

```bash
conda env create -f partiflow.yaml
conda activate partiflow-official
partiflow --help
```

Download the recommended models from [Hugging Face](https://huggingface.co/MegaGatlingPea/PartiFlow):

```bash
partiflow download
partiflow check
```

The default set contains `autoencoder`, `partiflow`, `partiflow-ss`, and `partiflow-aux`. In a terminal, these four are preselected; press Enter to download them. `partiflow-dock` is an optional specialist.

The [installation and resource guide](docs/resources.md) covers the other checkpoints, AF2/MPNN weights, PepInter data, and installation from a wheel.

## Make a first design

Save the following as `first.yaml` in the checkout root. It asks for one 12-residue peptide at the interface in the bundled KLK2 reference complex.

```yaml
first_binder:
  task_type: denovo
  pdb_path: ./pdbs/design/klk2/klk2_motif.pdb
  is_complex: true
  pep_chain: L
  hotspots: reference
  motif_cfg: null
  lens_cfg: {nres: [12], min_len: null, max_len: null, step_len: null}
  sample_num: 1
  postprocess: false
```

Inspect the settings, then run generation on a prepared GPU:

```bash
partiflow design first.yaml --batch-size 1 --num-workers 0 --show-config
partiflow design first.yaml --batch-size 1 --num-workers 0 --out inference/first
```

Open `inference/first/first_binder/sample1.pdb` to inspect the generated complex. Its `result.csv` records the generated sequence, binder chain, constraints, and any error. `resolved_config.yaml` records the settings used. Choose a fresh `--out` directory for another design run.

Next, [validate](docs/commands/validate.md) candidates with paired AF2 predictions, [relax and filter](docs/commands/postprocess.md) structures, or [cluster](docs/commands/cluster.md) a passing cohort. A generated PDB, a high reward, and a passed computational filter describe different stages of the workflow.

## Choose a task and checkpoint

| Task | Reference conditions | What is generated |
| --- | --- | --- |
| [Denovo](docs/tasks/denovo.md) | Receptor context, requested length, optional conditions | Peptide sequence and all-atom coordinates |
| [Motif](docs/tasks/motif.md) | Selected motif identities and all-atom geometry | Surrounding peptide scaffold |
| [Docking](docs/tasks/docking.md) | Peptide sequence | Bound coordinates |
| [Inverse folding](docs/tasks/inverse-folding.md) | Peptide CA geometry | Sequence and side chains |
| [Packing](docs/tasks/packing.md) | Peptide sequence and CA geometry | Side chains |

| Checkpoint | Use |
| --- | --- |
| `partiflow.ckpt` | General denovo and motif design |
| `partiflow-aux.ckpt` | Docking, inverse folding, and packing |
| `partiflow-dock.ckpt` | Docking specialist used in the CPSet-separated comparison |
| `partiflow-ss.ckpt` | Secondary-structure, scaffold-topology, and disulfide-conditioned generation |

All four models use `autoencoder.ckpt`. Select the checkpoint explicitly when changing model capabilities. For the bundled [scaffold examples](docs/scaffolds.md):

```bash
partiflow download partiflow-ss
partiflow design configs/tasks/scaffold.yaml --task klk2_scaffold_linear \
  --ckpt ./ckpts/partiflow-ss.ckpt --out inference/scaffold_linear
```

The seven commands are `train`, `design`, `validate`, `postprocess`, `cluster`, `check`, and `download`. Terminals use cyan framed help, grouped resource reports, and execution context and results. `partiflow download` opens a keyboard resource selector; use Up/Down, Space, Enter, and Esc. Explicit command arguments remain available for scripts. Redirected JSON/text and `--show-config` output stay plain; `NO_COLOR` disables colored reports. `python -m partiflow` accepts the same arguments. See the [guide index](docs/README.md), [task fields](docs/task_schema.md), [Best-of-N search](docs/reward.md), and [topology guide](docs/topology.md) for the next steps.

The [performance figures](docs/performance.md) describe the original study and its benchmark settings.
