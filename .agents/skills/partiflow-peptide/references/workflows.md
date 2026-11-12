# PartiFlow workflow recipes

## Task variants

Keep the named task YAML format. Use the user's actual PDB chain IDs, residue numbering, sequences, and output paths.

- **Motif:** use a reference complex, `task_type: motif`, and `motif_cfg` such as `{contigs: 2-8/L6-11/2-8, min_len: 12, max_len: 20}` with `lens_cfg: null`. The retained segment is source-chain L residues 6–11; use the actual reference chain and selected motif. The general model grows the surrounding scaffold.
- **Docking:** select `partiflow-aux.ckpt`. A reference complex supplies the peptide sequence, or a receptor-only task supplies an explicit `sequence`.
- **Inverse folding / packing:** use a reference complex and the auxiliary checkpoint. Inverse folding retains peptide CA geometry; packing retains sequence and CA geometry.
- **Scaffold conditions:** select `partiflow-ss.ckpt` explicitly. `binder_topology.mode: sample` uses a scaffold library/category/preset; requested lengths must match entries. Bundled libraries cover linear, head-to-tail amide-cyclic, and terminal-disulfide patterns. Their presets describe geometry, separately from quality-filter presets.

Source residue selectors refer to the input PDB. Constraint and topology positions refer to the generated peptide, starting at one. For example:

```yaml
constraint:
  restype:
    - {resid: 3, aa: C}
    - {resid: 7, aa: C}
topology:
  - {type: disulfide, begin_resid: 3, end_resid: 7}
postprocess: {enabled: true, method: auto}
```

Chemical bonds require topology declarations and restoration; a distance condition alone does not create a bond. Special topology and registered nonstandard chemistry require postprocessing. A terminal disulfide connects cysteine SG atoms; head-to-tail amide cyclization connects peptide termini.

Use the [task-field guide](https://github.com/MegaGatlingPea/PartiFlow/blob/main/docs/task_schema.md), [scaffold guide](https://github.com/MegaGatlingPea/PartiFlow/blob/main/docs/scaffolds.md), and [topology guide](https://github.com/MegaGatlingPea/PartiFlow/blob/main/docs/topology.md) for the chosen condition.

## Best-of-N reward

Prepare AF2 parameters and request replicas explicitly:

```bash
partiflow download af2 --model-dir models
partiflow design tasks.yaml --task my_binder --model-dir models \
  --override search=best_of_n --override reward=af2 \
  --override search.best_of_n.replicas=4 --out inference/my_best_of_n
```

The default reward uses the generated sequence with AF2 Multimer v3 initial guess, model 1, three recycles, and no MSA search. The score is `0.25 * (design_plddt / 100) + 0.75 * interface_iptm`.

Replicas multiply the task's length/sample budget. All successfully evaluated candidates remain, ranked within task and actual length; Best-of-N does not keep only one winner per base request. Rank 1 is the best score in its group. Reward values describe the prediction before subsequent task relaxation.

## Validation and saved results

```bash
partiflow download af2 solublempnn --model-dir models
partiflow validate inference/my_run --model-dir models --criteria af2_ig_peptide
```

Validation defaults to one original-sequence branch and one SolubleMPNN branch. Each uses pTM paired complex and monomer predictions, model 1, one sample, and three recycles. This differs from the Multimer reward protocol.

To choose all three supported sequence sources, also download ProteinMPNN:

```bash
partiflow download proteinmpnn --model-dir models
partiflow validate inference/my_run --model-dir models \
  --source gt:1 --source proteinmpnn:1 --source solublempnn:1 \
  --criteria af2_ig_peptide
```

For an existing validation run, changed source/prediction settings require `--rerun`. Repeating the same request fills missing work and reuses matching completed predictions. `--refilter` reapplies criteria to saved metrics; `--recompute-metrics` recalculates metrics from saved predictions. Neither starts AF2/MPNN. Use only one of `--plan`, `--refilter`, or `--recompute-metrics`; the latter two do not combine with `--rerun`.

Results are under `TASK/validate/af2_ig/`: `metrics.csv`, `success.csv`, `summary.json`, and `refolds/`. An empty criteria list produces metrics without a passing cohort. Missing required predictions/metrics are unavailable.

AF2 requires supported standard-protein chemistry. Nonstandard residues or unrepresentable atoms, ligands, and bonds can be unsupported. Cyclic positional encoding alone does not physically close a refolded ring. Explain these limits for the actual input rather than treating a substituted structure as validation of it.

## Postprocessing and chemistry

```bash
partiflow postprocess inference/my_run --filter false
partiflow postprocess inference/my_run --filter true
```

Standard peptides use Rosetta `beta_nov16`, one global FastRelax repeat, and up to 200 minimizer iterations, with no sequence design. OpenMM handles supported nonstandard residues and broader chemical connections. Automatic routing retains chemical identities and declared bonds; explicit routes must support the input.

Filtering uses `rosetta_bindcraft_peptide` by default: shape complementarity, interface hydrogen bonds, interface unsatisfied hydrogen bonds, and surface hydrophobicity. Rules and thresholds come from the selected preset YAML. A failed threshold retains the relaxed structure. A missing required metric is unavailable and cannot pass. BAL currently lacks the surface metric while other supported scores remain available.

`postprocess RUN` updates design PDBs and their result-table identities, preserving reward values/ranks. For AF2 predictions, pass selected refold paths or an explicit `AF2success` directory; derived structures are saved separately. Reports are under `TASK/postprocess/`, or a separate refold output directory for explicit structures.

Repeating a compatible request can reuse completed relaxation and add missing metrics. Threshold-only changes can reapply stored metrics. After editing a structure at the same path or changing the relaxation protocol, use `--rerun`. Keep inputs fixed during a run.

To set CPU concurrency, save a YAML overlay and use `--config FILE`:

```yaml
workers: 8
threads: 1
seed: 2026
timeout: 1800
```

`workers` caps concurrent persistent Ray workers; `threads` is each worker's CPU budget. Defaults choose workers from the available allocation and use one thread per worker. Rosetta runs on CPU; `--platform` selects the OpenMM platform.

## Clustering

```bash
partiflow cluster inference/my_run --criteria af2_ig_peptide --tm-threshold 0.5
```

Run/task inputs select the saved AF2-passing cohort and cluster current design backbones using US-align. The default threshold is 0.5; multiple passing sequence branches from one design count once. Reports are `clusters.csv` and `cluster_summary.csv` under `TASK/validate/af2_ig/`.

For a Rosetta-filtered cohort, pass its saved postprocessing directory explicitly and select `--criteria rosetta_bindcraft_peptide`. Its cluster reports go under `TASK/postprocess/clusters/`. An empty passing cohort produces empty results.

## PepInter and training

Download training data only when preparing a training run:

```bash
partiflow download pepinter --data-dir train_data/pepinter
python -m partiflow.training.prepare_data \
  --archive train_data/pepinter/pepinter.tar.xz --root train_data/pepinter
```

The download saves the raw archive only. Preprocessing defaults to all 171,019 structures, preserving 166,019 train, 3,750 validation, and 1,250 test examples; `valid` maps to `val`. A raw archive does not satisfy the processed-data checks.

Use the installed/bundled VAE or FM configuration, adjusting data paths, hardware, run name, and output location for the requested run:

```bash
partiflow train vae --config configs/training_ae.yaml --override run_name_=my_vae
partiflow train fm --config configs/training.yaml --model-dir models --override run_name_=my_fm
```

FM needs the shared autoencoder. `--show-config` resolves settings without training. Configuration `resume: auto` can continue an existing run; `resume: none` starts fresh. An explicit `resume` checkpoint restores training state. `pretrain_ckpt_path` loads weights for a warm start with resume disabled. Use a new run name to keep outputs separate.

Training writes effective settings, checkpoints, and logs under `<output_dir>/<run_name_>/`. The supplied setup uses one GPU; multi-GPU training has not been validated for this release. See the [training guide](https://github.com/MegaGatlingPea/PartiFlow/blob/main/docs/commands/train.md) for compilation and configuration details.

## Installed examples

Repository examples use paths relative to its root. A wheel installation includes small resources inside the package; locate them with:

```bash
PARTIFLOW_RESOURCES=$(python -c 'from partiflow.paths import resource_root; print(resource_root())')
partiflow design "$PARTIFLOW_RESOURCES/configs/tasks/scaffold.yaml" \
  --task klk2_scaffold_linear --model-dir models \
  --ckpt models/ckpts/partiflow-ss.ckpt --out inference/scaffold
```

Bundled tasks resolve their packaged PDBs/libraries automatically. After copying a task to your own file, update paths for the working directory. The same resource root supplies bundled training configurations.
