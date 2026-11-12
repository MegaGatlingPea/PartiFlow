# Relax structures and apply quality rules

`postprocess` relaxes saved structures and can evaluate interface quality. For design inputs, it updates the published PDBs and their result-table identities; reports go under each task's `postprocess/` directory.

```bash
partiflow postprocess inference/first --plan
partiflow postprocess inference/first --filter false --platform CPU
partiflow postprocess inference/first --filter true --criteria rosetta_bindcraft_peptide
```

Inputs include run/task directories, `result.csv` files, explicit PDB files, and explicit `AF2success` directories. Repeat `--task` and `--criteria` for selections. `--config FILE` overlays the [postprocessing configuration](../../configs/postprocess.yaml); `--presets FILE` supplies quality rules. `--plan` lists the work without relaxing or scoring structures.

## Relaxation and chemistry

The standalone command defaults to `method: auto`, CPU, linear relaxation enabled, and filtering disabled. Standard Rosetta relaxation uses `beta_nov16`, one FastRelax repeat, and up to 200 minimizer iterations. Backbone and side chains can move globally, rigid-body jumps stay fixed, and the default adds no coordinate or motif restraints.

Head-to-tail and disulfide tasks retain their declared chemical bonds during relaxation. Final geometry and PDB connection records follow the restored chemistry. Relaxation-only penalties are removed before quality scoring.

Registered nonstandard residues and broader covalent connections use OpenMM. An explicit backend choice must support the input chemistry. In `auto` mode, Rosetta runtime failures can fall back to OpenMM. See [topology](../topology.md) for bond declarations and supported routes.

For supported nonstandard residues, native Rosetta energy scoring uses mapped residue types at fixed coordinates, without Rosetta relaxation, minimization, or packing. The [support matrix](../../partiflow/postprocess/rosetta_ncaa/support_matrix.csv) lists the 45 supported energy-scoring types. BAL uses B3G beta-backbone chemistry and currently lacks `surface_hydrophobicity`; its other supported metrics remain available.

Task-level postprocessing during design has a separate default: ordinary linear tasks skip relaxation unless requested. Special topology and registered nonstandard chemistry still require restoration. Standalone postprocessing lets you relax published candidates later.

## Optional filtering

`--filter false` requests relaxation without quality metrics. `--filter true` calculates the metrics needed by the selected criteria. The default criterion is `rosetta_bindcraft_peptide`, with shape complementarity, interface hydrogen bonds, interface unsatisfied hydrogen bonds, and surface hydrophobicity rules. See the [preset guide](../presets.md#rosetta-interface-criteria) for exact operators and thresholds.

A failed threshold retains the relaxed structure and measured values. A missing required metric is `unavailable`, and a monomer receives `not_applicable` for an interface preset. BAL cannot pass the default four-rule preset while its surface metric is unavailable.

## Runtime settings

Use `--platform CPU|CUDA|OpenCL` to select the OpenMM platform. Rosetta calculations run on CPU. A YAML overlay can set CPU concurrency:

```yaml
workers: 8
threads: 1
seed: 2026
timeout: 1800
```

Save it as `postprocess-local.yaml`, then run:

```bash
partiflow postprocess inference/first --config postprocess-local.yaml --filter true
```

The default `workers: null` chooses a cap from the selected work and available CPU allocation. `threads: 1` is the per-worker CPU budget. `timeout: 1800` bounds startup and individual sample calls in seconds.

## Results and reruns

`TASK/postprocess/summary.json` and sample reports record relaxation, metrics, and criterion decisions. Design `result.csv` keeps its reward values and ranks while updating `postprocess_status`, final identities, and atom locators. A relaxation failure leaves the previous valid structure available.

Explicit PDB/refold inputs instead write separate structures under `postprocess/refold/` and keep the originals. If a metric calculation fails after relaxation, completed geometry may remain available while its missing metrics are marked `unavailable`.

```bash
partiflow postprocess inference/first --filter false
partiflow postprocess inference/first --filter true
partiflow postprocess inference/first --filter true --rerun
```

The first two commands can reuse compatible completed relaxation and add missing metrics. Threshold-only preset changes reevaluate stored metrics. Use `--rerun` after editing a structure at the same path, changing the relaxation protocol, or opening results from an older incompatible report format. Keep structures fixed while a command is running.
