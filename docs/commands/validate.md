# Validate with paired AF2 predictions

Validation predicts the receptor–peptide complex and the isolated peptide using the same sequence. This compares confidence in the bound pose with the peptide's independently predicted conformation.

Prepare `af2` and `solublempnn` weights, inspect the work, then run:

```bash
partiflow validate inference/first --plan
partiflow validate inference/first --criteria af2_ig_peptide
```

Inputs can be run directories, task directories, `result.csv` files, or supported structure files. Repeat `--task` to select tasks and use `--limit N` to limit rows within each task. `--model-dir` selects local model weights. `--config FILE` overlays the [validation settings](../../configs/validate.yaml), and `--presets FILE` selects a quality-preset YAML.

## Sequences and prediction settings

The defaults request two sequence branches for each design: one original sequence (`gt`, labeled `self`) and one SolubleMPNN sequence. Both run paired pTM complex and monomer predictions with model 1, one sample, and three recycles. The default criteria list is empty, so add `--criteria af2_ig_peptide` when you want pass/fail decisions.

To compare all three supported sequence sources, also prepare the ProteinMPNN weights and select the branches explicitly:

```bash
partiflow validate inference/first \
  --source gt:1 --source proteinmpnn:1 --source solublempnn:1 \
  --criteria af2_ig_peptide --rerun
```

`--source` replaces the default source list. `--rerun` is needed when changing an existing run's branch settings; omit it for a fresh validation run. Source aliases include `self`, `protein_mpnn`, and `soluble_mpnn`. An MPNN source ending in `-fi` also retains original interface identities, in addition to the task's fixed identities.

Use a YAML overlay for other prediction settings. For example:

```yaml
af2:
  models: [1, 2]
  samples: 1
  recycles: 3
```

`af2.models` accepts an ordered, unique subset of `[1, 2]`. Each configured model/sample produces a complex prediction and a separate monomer prediction. Models remain loaded across requests in one process. To choose another visible GPU, set its visibility before launching a separate command.

## Read the results

Each task writes:

```text
TASK/validate/af2_ig/
  metrics.csv
  success.csv
  summary.json
  refolds/
```

`metrics.csv` contains prediction and paired-comparison metrics. `success.csv` records the sequence branches passing each requested criterion. `summary.json` summarizes the requested and observed work and criterion counts. `refolds/` contains the saved sequence branches, structures, and confidence data. Validation leaves design PDBs and the design result table intact.

The [AF2 quality preset](../presets.md#paired-af2-criteria) requires confidence thresholds and agreement between bound and unbound peptide structures. A completed prediction can fail those rules. Missing required predictions or metrics produce `unavailable`.

AF2 supports standard protein residues and representable protein topology in every chain. Unsupported nonstandard chemistry, ligands, or atoms produce `not_applicable` with an `unsupported_chemistry` reason. MPNN redesign does not remove this input restriction. Cyclic positional encoding alone does not physically close a refolded peptide ring.

## Continue or update a run

Repeat the original command to fill missing work and reuse completed predictions with matching inputs and settings. Keep any custom `--source` and `--config` settings the same. After changing an input structure or prediction protocol, use `--rerun`.

For a run made with the default sequence settings:

```bash
partiflow validate inference/first --criteria af2_ig_peptide --refilter
partiflow validate inference/first --criteria af2_ig_peptide --recompute-metrics
```

`--refilter` reapplies criteria to saved metrics. `--recompute-metrics` recalculates metrics from saved predictions and confidence arrays. Neither runs AF2 or MPNN, and missing predictions remain incomplete. `--plan`, `--refilter`, and `--recompute-metrics` are mutually exclusive. Do not combine `--rerun` with either saved-result update mode.

Use a criterion during validation or refiltering before [clustering](cluster.md) an AF2 passing cohort.
