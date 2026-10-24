# Validation

```bash
partiflow validate inference/klk2 --plan
partiflow validate inference/klk2 --source gt:1 --source soluble_mpnn:1 --model-dir /path/to/models
partiflow validate inference/klk2 --criteria af2_ig_peptide --refilter
```

Inputs are one or more run/task paths, `result.csv` files, or supported structure files. Repeat `--task` to select tasks, `--source SOURCE:COUNT` to select sequence branches, and `--criteria NAME` to select named criteria. `--config FILE` merges validation settings, `--presets FILE` supplies criteria definitions, `--limit N` limits input rows within each task, and `--model-dir` resolves local checkpoints. `--rerun` explicitly recomputes requested work after an input or protocol change.

Validation executes serially in one owner process. AF2 and MPNN models remain resident across requests and close at the end. There are no per-worker CPU/GPU allocation options. Execution uses the active process's visible device; plan separate invocations explicitly when separate devices are needed.

`--plan`, `--refilter`, and `--recompute-metrics` are mutually exclusive. Planning inspects the selected work. Refiltering reevaluates saved metrics; metric recomputation reads saved predictions and confidence arrays without running AF2 or MPNN. Missing predictions stay incomplete. `--rerun` cannot be combined with either cache-update mode. Completed prediction slots resume under matching saved inputs and settings; changed inputs or protocol settings require `--rerun`.

[configs/validate.yaml](../../configs/validate.yaml) defaults to one original sequence (`gt`, label `self`) and one SolubleMPNN sequence, pTM model 1, one sample, and three recycles. `af2.models` accepts an ordered unique subset of `[1, 2]`; each model/sample runs an initial-guess complex protocol and an independent binder monomer protocol using the same sequence. Sequence sources are `gt`, `proteinmpnn`, and `solublempnn`, with documented aliases such as `self`, `protein_mpnn`, and `soluble_mpnn`. Appending `-fi` to an MPNN source retains its original interface identities in addition to fixed task identities.

AF2 accepts standard protein residues and representable protein topology in the design and every context chain. Nonstandard residues, unsupported atoms, ligands, and other unsupported chemistry are reported explicitly as `not_applicable` with an `unsupported_chemistry` reason. They are not silently replaced by canonical residues, even when an MPNN redesign was requested. Source numbering, insertion codes, chain identity, and confidence arrays remain associated with each supported prediction. Cyclic offset encoding does not establish physical ring closure in a refold.

Each task writes `validate/af2_ig/metrics.csv`, `success.csv`, `summary.json`, and per-sample sequence slots under `refolds/`. Original design structures and `result.csv` identities remain intact. The default criteria list is empty, which collects metrics without a pass/fail preset. Add `--criteria af2_ig_peptide` during validation or refiltering to publish that cohort for [clustering](cluster.md).

The exact thresholds and comparison operators live in [configs/preset.yaml](../../configs/preset.yaml). `af2_ig_peptide` uses `all_and_mean`: every configured complex, monomer, and paired comparison must satisfy the applicable rules, together with their means. Confidence fractions are on a 0–1 scale, normalized PAE is distinct from PAE in angstroms, and paired bound/unbound RMSD is in angstroms. Missing required metrics yield `unavailable`; metrics from different predictions are never mixed to manufacture a pass.
