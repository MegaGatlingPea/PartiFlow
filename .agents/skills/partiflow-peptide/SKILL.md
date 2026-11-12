---
name: partiflow-peptide
description: "Run PartiFlow peptide-design workflows: prepare task YAML, choose models, generate binders, validate with AF2, relax and filter structures, and cluster results. Also use for installing resources, interpreting outputs, or running PartiFlow VAE/FM training."
---

# PartiFlow peptide workflows

Help the user turn a peptide-design goal and input structures into a runnable PartiFlow task. Use their installed environment, chosen inputs, and output location. Run the stages they request; use the recipes below to choose the next step.

## Choose the task and model

Establish the input structure, receptor chains, any reference peptide, desired length/count, and requested sequence, motif, or chemical conditions. Ask for missing information only when it changes the task.

| Goal | Task and checkpoint |
| --- | --- |
| Generate a peptide sequence and structure | `denovo`, `partiflow.ckpt` |
| Grow a scaffold around reference residues | `motif`, `partiflow.ckpt` |
| Dock a sequence, redesign a fixed backbone, or pack side chains | `docking`, `inv_folding`, or `packing`; `partiflow-aux.ckpt` |
| Apply secondary-structure, scaffold-topology, or disulfide conditioning | Appropriate task with explicit `partiflow-ss.ckpt` |

All models need `autoencoder.ckpt`. The optional `partiflow-dock.ckpt` is a docking specialist.

## Prepare resources

Use the existing PartiFlow installation. The repository environment is activated with `conda activate partiflow-official`. For installation help, see [Installation and resources](https://github.com/MegaGatlingPea/PartiFlow/blob/main/docs/resources.md).

```bash
partiflow download --model-dir models
partiflow check --model-dir models
```

The default download set is `autoencoder`, `partiflow`, `partiflow-ss`, and `partiflow-aux` from the public Hugging Face repository. A terminal opens a selector with those four checked; Enter accepts them. A noninteractive invocation downloads the four directly. Explicit targets replace this set. `--all` also downloads optional models and training data.

Download AF2/MPNN resources when the selected workflow needs them:

```bash
partiflow download af2 solublempnn --model-dir models
```

## Prepare a task YAML

Task files map names to settings. For a receptor-only input, a minimal denovo task is:

```yaml
my_binder:
  task_type: denovo
  pdb_path: ./inputs/target.pdb
  is_complex: false
  pep_chain: null
  hotspots: null
  motif_cfg: null
  lens_cfg: {nres: [12], min_len: null, max_len: null, step_len: null}
  sample_num: 1
  postprocess: false
```

Replace the input path and requested length/count with the user's choices. With a reference complex, set `is_complex: true` and its actual peptide chain in `pep_chain`. Use `hotspots: reference` only with a reference peptide; explicit hotspot selectors use source PDB chain IDs and residue numbers.

`sample_num` applies to each requested length/condition. Relative paths in a user task file resolve from the command's working directory. For motifs, scaffold libraries, and chemical bonds, read [Task variants](references/workflows.md#task-variants).

## Generate candidates

```bash
partiflow design tasks.yaml --task my_binder --model-dir models --out inference/my_run
```

Select the model explicitly for other capabilities, for example `--ckpt models/ckpts/partiflow-ss.ckpt`. Each selected task output directory must be absent or empty; use a fresh `--out` for another generation run.

`--show-config` prints the resolved design settings without generation. Use `--config FILE` for an overlay and repeat `--override KEY=VALUE` for individual settings when needed.

Each task writes PDBs, `result.csv`, `resolved_config.yaml`, and logs. Read actual structures from `result.csv`'s `filename` column. It also records sequences, chain identities, optional rewards/ranks, postprocessing status, and errors; failed samples can remain as rows.

## Validate, relax, or cluster

Choose the stage that answers the user's question:

```bash
partiflow validate inference/my_run --model-dir models --criteria af2_ig_peptide
partiflow postprocess inference/my_run --filter true
partiflow cluster inference/my_run --criteria af2_ig_peptide
```

- `validate` uses original and SolubleMPNN sequences by default, with paired complex/monomer AF2 predictions. Without `--criteria`, it reports metrics only. See [Validation and saved results](references/workflows.md#validation-and-saved-results).
- `postprocess RUN` relaxes and updates generated structures. Explicit PDB/refold inputs produce separate structures and retain the originals. `--filter true` applies the default Rosetta peptide preset; `--filter false` relaxes only. See [Postprocessing and chemistry](references/workflows.md#postprocessing-and-chemistry).
- `cluster RUN` selects AF2-passing sequence branches and clusters their corresponding design backbones. It deduplicates multiple sequences from one backbone. See [Clustering](references/workflows.md#clustering).

A generated structure, a high reward, and a passed criterion describe different outcomes. Explain the saved measurements and selected rules, including any failed or unavailable metrics.

For [Best-of-N reward](references/workflows.md#best-of-n-reward), [PepInter preparation and training](references/workflows.md#pepinter-and-training), and [installed example paths](references/workflows.md#installed-examples), read only the relevant recipe.
