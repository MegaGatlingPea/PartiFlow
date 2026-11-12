# Quality presets

Quality presets turn calculated metrics into a selected cohort. Their names, thresholds, comparison operators, and reduction rules live in [configs/preset.yaml](../configs/preset.yaml).

## Paired AF2 criteria

Request `af2_ig_peptide` during validation or apply it later to saved metrics:

```bash
partiflow validate inference/first --criteria af2_ig_peptide
partiflow validate inference/first --criteria af2_ig_peptide --refilter
```

| Scope | Metric | Required value |
| --- | --- | --- |
| Complex | pLDDT | ≥ 0.80 |
| Complex | pTM | ≥ 0.55 |
| Complex | ipTM | ≥ 0.40 |
| Complex | Normalized interface PAE | ≤ 0.30 |
| Monomer | pLDDT | ≥ 0.80 |
| Paired complex/monomer | Bound–unbound RMSD | ≤ 2.5 Å |

The `all_and_mean` reduction requires the applicable individual predictions and their means to pass. Each sequence branch is judged using its own paired predictions. Confidence fractions use 0–1 values; normalized PAE differs from PAE in angstroms. Missing predictions or required metrics produce `unavailable`.

By default, validation collects metrics with an empty criteria list. Add a criterion to produce a passing cohort for [clustering](commands/cluster.md). When continuing a customized validation run, keep its `--source` and `--config` settings the same.

## Rosetta interface criteria

Enable filtering after relaxation with:

```bash
partiflow postprocess inference/first --filter true --criteria rosetta_bindcraft_peptide
```

| Metric | Required value |
| --- | --- |
| Shape complementarity | > 0.55 |
| Interface hydrogen bonds | > 1 |
| Interface unsatisfied hydrogen bonds | < 3 |
| Surface hydrophobicity | < 0.50 |

The strict operators matter: equality fails these four rules. A failed quality rule retains the relaxed structure and measured values. Missing required metrics produce `unavailable`; an interface preset on a monomer is `not_applicable`.

For supported nonstandard residues, metric availability depends on the chemistry. BAL uses B3G beta-backbone scoring, for which `surface_hydrophobicity` is currently unavailable. It therefore cannot pass this four-rule preset. See [postprocessing](commands/postprocess.md) for chemical support.

## Define a custom preset

Save the following as `quality.yaml` and select it with `--presets`:

```yaml
presets:
  my_af2:
    methods: [af2_ig]
    reduction: all_and_mean
    rules:
      - {metric: plddt, op: '>=', value: 0.85, scope: complex}
      - {metric: bound_unbound_rmsd, op: '<=', value: 2.0, scope: paired}
```

```bash
partiflow validate inference/first --presets quality.yaml --criteria my_af2 --refilter
```

Use `methods: [rosetta]` and the appropriate Rosetta metric names for a postprocessing preset. Changing thresholds can reuse saved metrics; changing which metrics are required may require calculating the missing values. Filtering is a computational selection step, and experimental binding remains a separate measurement.
