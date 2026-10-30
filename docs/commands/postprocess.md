# Postprocessing

`partiflow postprocess` relaxes saved structures and optionally applies the
quality rules in a preset. It uses the original PartiFlow task and topology
metadata and keeps quality metrics separate from design reward values.

```bash
partiflow postprocess inference/klk2 --plan
partiflow postprocess inference/klk2 --filter false --platform CPU
partiflow postprocess inference/klk2 --filter true --criteria rosetta_bindcraft_peptide
```

## Inputs and execution settings

Inputs are one or more run/task paths, `result.csv` files, explicit PDB files,
or an explicit `AF2success` directory. The default processes published design
structures. `--filter` accepts exactly `true` or `false` and defaults to false.
The other options are `--config`, repeatable `--task` and `--criteria`,
`--presets`, `--rerun`, `--plan`, `--platform CPU|CUDA|OpenCL`, and `--model-dir`.
`--plan` prepares the selected work without running relaxation or scoring.

Defaults come from [configs/postprocess.yaml](../../configs/postprocess.yaml).
One command shares a pool of persistent Ray workers across its selected tasks.
Each worker handles a sample through relaxation, required metrics, preset
decisions, and publication before accepting another compatible sample.
An idle worker immediately takes the next compatible sample; tasks can compute
concurrently even though each task retains its own result table.

Use a small YAML overlay to control CPU resources:

```yaml
# postprocess-local.yaml
workers: 8
threads: 1
seed: 2026
timeout: 1800
```

```bash
partiflow postprocess inference/klk2 --config postprocess-local.yaml --filter true
```

`workers: null` automatically chooses an upper bound from the selected work and
allocated CPU resources, including affinity and job allocation. An explicit
worker count is a concurrency cap. `threads: 1` is the default per-worker CPU
budget. `timeout: 1800` bounds worker startup and individual sample calls in
seconds. Workers reset their Rosetta random stream before each sample using
`seed`; the metric phase has its own reset. Standard Rosetta work runs on CPU.
GPU-requested OpenMM work also observes available GPU resources.

## Relaxation and chemical support

The default standard-peptide Rosetta protocol uses `beta_nov16`, global
backbone and side-chain mobility, fixed rigid-body jumps, and repacking without
sequence changes. FastRelax uses one repeat and at most 200 minimizer
iterations. It adds no coordinate or motif restraints. Existing configuration
overrides for mobility and iteration count remain available.

PartiFlow's disulfide and head-to-tail logic remains part of the calculation:
topology is constructed, real chemical connections and required relaxation
penalties are retained, final geometry is checked, and the corresponding PDB
connection records are written. Relaxation-only penalties are removed before
quality scoring.

Registered nonstandard residues and broader covalent connections continue
through OpenMM. Explicit backend requests are strict. In `auto` mode, a
Rosetta runtime failure can fall back to OpenMM. OpenMM retains its force fields
and harmonic restraints, and its CPU thread setting follows `threads`.

Standard `beta_nov16` work and native NCAA `ref2015` scoring use separate SDK
profiles in separate worker processes. A process keeps its profile for its
lifetime, keeping standard beta correction settings separate from native NCAA
chemistry. Each worker initializes its required SDK at most once.
Unfiltered OpenMM work in the `native_ref2015` profile does not initialize the
native scoring SDK.

NCAA scoring uses exact mapped residue types at fixed coordinates, with
Rosetta relaxation, minimization, and packing disabled for those types. The
[native support matrix](../../partiflow/postprocess/rosetta_ncaa/support_matrix.csv)
contains 45 supported energy-scoring types. BAL maps to Rosetta B3G. In the
current LayerSelector implementation, its beta backbone leaves
`surface_hydrophobicity` unavailable; other supported metrics and native energy
scoring remain available.

## Optional filtering

`--filter false` performs relaxation without requesting quality metrics.
`--filter true` calculates only the metrics required by the selected presets
and their calculation dependencies. A standard Rosetta sample reuses its
relaxed Pose for scoring. An OpenMM result is converted to the appropriate
native scoring Pose once when scoring is needed.

The default `rosetta_bindcraft_peptide` preset keeps PartiFlow's four rules:

- `shape_complementarity`
- `interface_hbonds`
- `interface_unsat_hbonds`
- `surface_hydrophobicity`

Their exact thresholds and operators are maintained in
[configs/preset.yaml](../../configs/preset.yaml). The standard relaxation
protocol update does not replace this preset. Extra PackStat, SASA, energy, or
composition calculations run only when the requested metrics need them.
Custom criteria belong in a preset YAML selected with `--presets`.

A missing required metric is `unavailable` and cannot pass its preset. BAL
therefore cannot pass the default four-rule preset while its required surface
metric is unavailable. Monomers receive `not_applicable` for an interface
preset. A scientific threshold failure retains the relaxed structure and its
measured values.

## Outputs, failures, and resume

For design inputs, a completed staged structure atomically replaces its source.
Its task's `result.csv` refreshes the actual design identity and atom locators;
reward values and ranks retain their original meaning. Reports and preset
decisions are written under `TASK/postprocess/`. A relaxation failure retains
the previous valid input.

An explicit `AF2success` input instead writes separate
`postprocess/refold/` artifacts and preserves sequence-slot paths in its reports.
The original refolds, validation reports, and design result table remain intact.

Geometry is published with a small completion record before optional native
metric calculation. If a native metric calculation crashes its worker, the
completed geometry remains available and the missing metrics and decisions
are reported as `unavailable`. Other samples continue. The command does not
automatically repeat the failed sample in the same invocation.

`result.csv` publication retains a task-level rollback boundary. If publication
fails, that task's structures, reports, and table are restored. Tasks already
published successfully remain committed even if a later task fails; the
computations for different tasks may have overlapped.

```bash
# Reuse compatible completed geometry and fill missing work.
partiflow postprocess inference/klk2 --filter false
# Add metrics without relaxing completed samples again.
partiflow postprocess inference/klk2 --filter true
# Explicitly repeat relaxation after an input or protocol change.
partiflow postprocess inference/klk2 --filter true --rerun
```

Version 2 records store effective settings, identity metadata, output locations,
metrics, and decisions. They do not store copies of every input/output atom or
coordinate. Resume compares the saved settings and metadata and checks that
the published output exists. Keep inputs fixed during a command. After an
external edit to a structure at the same path, use `--rerun`; lightweight
records do not automatically detect same-path coordinate changes.

A later filtered invocation can reuse successful relaxation. Threshold-only
preset changes rejudge stored metrics without repeating relaxation. Earlier
report layouts require explicit `--rerun`; the command does not migrate them
or remove unselected outputs.
