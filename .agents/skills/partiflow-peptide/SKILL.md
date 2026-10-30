---
name: partiflow-peptide
description: Maintain the selected PartiFlow official checkout, including its CLI, VAE/FM training, design and structure workflows, tests, resources, and packaging. Use for PartiFlow official development or release validation; exclude general peptide design, paper writing, and development of other models.
---

# PartiFlow Official Development

Maintain the selected PartiFlow official checkout and its public behavior. First confirm the checkout, branch, existing changes, and current task. Verify commands against current source, configurations, and tests; past acceptance records do not establish the current state. All project paths below are relative to the selected checkout. Do not assume a particular machine, remote connection, or checkout location.

## Working Entry Points

- Use a dedicated environment based on the repository's `partiflow.yaml`. Inspect the existing interpreter and resources before making environment changes; keep other project environments outside the task.
- Start from `partiflow/cli.py`, `configs/`, and relevant tests. Read `docs/commands/` as needed. Preserve the original named PartiFlow task YAML files and their task semantics.
- Preserve model mathematics, parameter hierarchy, and checkpoint ABI. Ordinary weights, EMA weights, and full training restoration must each remain valid.
- Original PartiFlow, PartiDesign comparison code, and paper projects are read-only comparison material when the task needs them. Modifying another project requires an explicit task for that project.

## Command Routing

The public entry point is `partiflow`, with these seven commands. Confirm arguments using the current parser and `--help`.

| Request | Entry point and implementation |
| --- | --- |
| VAE or flow matching training | `train vae --config FILE` or `train fm --config FILE`; `partiflow/training/` |
| Generation, tasks, search, and reward | `design TASK.yaml`; `partiflow/design/`, `configs/search/`, `configs/reward/` |
| Paired AF2-IG validation | `validate PATH...`; `partiflow/workflows/validate.py` |
| Relaxation and optional quality filtering | `postprocess PATH...` with `--filter true` or `--filter false`; `partiflow/postprocess/` |
| Structural clustering | `cluster PATH...`; `partiflow/cluster/` |
| Resource status or explicit loading and execution checks | `check`; the default checks paths, while `--load` and `--smoke` execute models |
| Resource retrieval or import | `download`; `--list` shows the manifest, and current resource definitions determine model and data resolution |

For `train` and `design`, `--show-config` validates configuration resolution only. The default for `postprocess` is `--filter false`. Thresholds come exclusively from the selected `configs/preset.yaml` or `--presets` file.

Postprocessing defaults to a shared persistent Ray pool, with runtime resource
settings in `configs/postprocess.yaml`. Standard peptides use the
`beta_nov16` global relaxation protocol; OpenMM and native NCAA scoring retain
their separate chemistry paths and immutable SDK profiles. Preserve disulfide
and head-to-tail support and the existing four-rule peptide filter preset.
Lightweight postprocess records require `--rerun` after an external same-path
structure edit; they do not automatically detect coordinate changes.

For output contracts, reward, cached-result restoration, training, compilation, noncanonical amino acids (NCAA), or packaging, read the relevant sections of [Development and Validation](references/development.md). That reference identifies implementation files and focused tests.

## Validation and Delivery

- Run CPU contract tests directly related to the change before task-specific execution checks. Keep executable tests and small-data configurations in the project.
- Before a GPU experiment, inspect device availability and run experiments serially. Do not stop other processes. Leaf agents must leave GPU allocation and real-model validation to the main thread.
- Distinguish configuration resolution, resource presence, weight loading, software execution, structural quality, and model convergence. Passing tiny training establishes an executable software path only.
- Report changes, relevant code and figure sources, commands and configurations actually used, sample counts, observations, and unverified items in the requested language. Keep private run logs, generated reports, and development acceptance records outside release commits and uploads, in an external workspace or a user-selected private location. Exclude temporary figures, model weights, training data, and temporary assets from release content.
- Export release figures from canonical paper sources, follow the user's selected figure set, and identify figure numbers and source files. Performance claims must be supported by their cited results.
- Commit only within the current user's authorization. Do not push automatically or modify other projects.
