# PartiFlow Official Development and Validation

Read the sections relevant to the current task. All project paths are relative to the selected official checkout. When implementation changes, inspect current source and tests before updating the workflow; this reference does not replace the source.

## Source Map

| Behavior | Inspect first |
| --- | --- |
| Public commands and lazy loading | `partiflow/cli.py`, `tests/test_cli_public.py` |
| Original task YAML and configuration composition | `partiflow/design/config.py`, `partiflow/design/tasks.py`, `configs/tasks/`, `docs/task_schema.md` |
| Public CSV output | `partiflow/common/results.py`, `partiflow/design/runtime.py` |
| Reward and sequence policies | `partiflow/workflows/reward.py` and neighboring `reward_backend.py` and `sequence.py`; `configs/reward/af2.yaml`, `configs/sequence.yaml` |
| Paired validation and restoration | `partiflow/workflows/validate.py` and neighboring `af2.py`; `configs/validate.yaml` |
| Filtering semantics | `partiflow/common/presets.py`, `configs/preset.yaml` |
| Training, EMA, and compilation | `partiflow/training/`, `partiflow/utils/ema_callback.py` |
| Production postprocessing and native NCAA | `partiflow/postprocess/` (`runner.py`, `dispatch.py`, `source.py`, `chemistry.py`, `metrics.py`, `native.py`); `partiflow/utils/nonstandard_residue_registry.py` |
| Installed resources and wheel packaging | `partiflow/paths.py`, `partiflow/download.py`, `pyproject.toml`, and the active build configuration |

## Task and Result Contracts

Preserve the named task mapping. Supported `task_type` values are `denovo`, `motif`, `docking`, `inv_folding`, and `packing`. Retain the meanings of `pdb_path`, `pep_chain`, `lens_cfg`, `motif_cfg`, `constraint`, `topology`, `postprocess`, and other original fields. Do not replace them with another project's schema. Read sample counts and length expansion from the actual task plan.

`partiflow/common/results.py` currently defines 13 leading fields, actual numeric metric fields sorted by name, and 2 trailing fields:

```text
filename,design_chains,base_sample_id,design_modality,design_chain_id,token_length,reward_method,global_rank,design_sequence,contig_string,tip_atoms,fixed_restypes,bond_constraints,[sorted metric fields],postprocess_status,error
```

Without reward, the CSV has 15 columns. Leave missing metrics blank. `design_chains`, `fixed_restypes`, and `bond_constraints` are JSON arrays describing the final published structure; keep position, chain, and atom references synchronized. Raw samples use `sample<base>.pdb`; Best-of-N results use `sample<base>_<chain>_rank<rank>.pdb`. Do not invent filenames for failed rows.

Compute reward before task postprocessing. CSV reward values remain predictions from before relaxation. Standalone validation and postprocessing store their metrics and decisions in their own reports without overwriting the design identity contract.

## Reward and Paired Validation

- Preserve the reward protocol used for PartiDesign comparisons: AF2 **Multimer v3 with initial guess**, model 1, 3 recycles, no MSA, and generated sequences by default. The implementation converts `0.25 * (design_plddt / 100) + 0.75 * interface_iptm` to `float32`. Check current configuration and backend behavior; do not confuse pLDDT units of 0-100 with 0-1.
- `structure_source: fold` publishes the reward prediction. `structure_source: design` requires retaining the generated sequence; do not attach a changed sequence to coordinates whose side chains have not been rebuilt. Use only the original sequence, ProteinMPNN, SolubleMPNN, and their existing aliases as sequence sources. Preserve fixed task identities and optional fixed interface identities.
- Search supports raw sampling and Best-of-N. Expand copies per base request, retain all valid copies, and rank within task and actual-length groups. Break ties by stable candidate order. Preserve base identities and records for bases whose candidates all fail. Do not add `keep_top_n` truncation. On one GPU, complete generation and move or release the FM model's GPU allocation before running resident AF2/MPNN.
- Determine fixed identities from actual `fixed_restypes` and identity-dependent topology. The presence of `contig_string` or `tip_atoms` alone must not turn purely geometric conditioning into fixed sequence identities. The public reward method label is `af2multimer`; the standalone validation method remains `af2_ig`.
- Standalone validation uses **pTM paired complex and monomer predictions** for the same sequence. Current defaults are model 1, 1 sample, and 3 recycles. `af2.models` accepts an ordered, duplicate-free subset of `[1, 2]`. By default, evaluate one original sequence and one SolubleMPNN sequence; criteria are empty.
- Reward rank does not establish validation success. Read thresholds and operators only from the selected preset. For `af2_ig_peptide`, `all_and_mean` must not combine metrics from different predictions to manufacture a passing result. A missing required metric is `unavailable`.
- Report nonstandard residues, atoms, ligands, or topology unsupported by AF2 as `not_applicable` / `unsupported_chemistry`. Do not substitute standard residues and claim that the original input passed.

## Cached Results and Restoration

For validation restoration, directly compare current structure contents, residue and atom identities, chains and insertion codes, coordinates, topology, and effective configuration. External PSSM/tied files currently use path, size, and modification-time checks. Models currently record root paths and execution configuration; those checks do not compare all resource contents or weight tensors. Read and compare old plans and states before writing replacements. Do not overwrite an old plan to erase a mismatch.

Validation restores completed slots only when input and protocol agree. For changed input or protocol, use the current CLI's `--rerun` semantics. `--plan`, `--refilter`, and `--recompute-metrics` are mutually exclusive. The latter two process saved results without running AF2 or MPNN; missing predictions remain incomplete. Do not combine `--rerun` with either update mode. Currently, `--plan` returns before writing: it establishes plan compatibility, not prediction completeness. Refiltering or recomputing updates slots, metrics, decisions, and state reports while retaining design structures and prediction PDB/NPZ files. Recheck execution paths when implementation changes.

Postprocessing uses version 2 lightweight records. Compare effective settings, identity metadata, topology, and the published output location before reuse, and require the output to exist. Failed or old results must not hide the backend requested by a new run. Do not retain full input/output atom and coordinate copies in the records. Inputs are fixed for an invocation; external edits at the same path require explicit `--rerun`, and same-path coordinate changes are not automatically detected. Document this boundary rather than adding a replacement content-tracking mechanism. Earlier report layouts require explicit rerun and are not migrated automatically.

Changing from `filter: false` to `filter: true` reuses completed geometry. Threshold-only preset changes reuse stored metrics for rejudging. A metric-stage native crash retains already published geometry and marks missing metrics unavailable; a later invocation can finish scoring without relaxing that geometry again.

## Training, Checkpoints, and Compilation

- Use `configs/training_ae.yaml`, `configs/training.yaml`, and the current `train vae|fm` adapters. Preserve data transforms, VAE/FM losses, optimizer behavior, and parameter keys. Code organization changes must preserve model mathematics and checkpoint ABI.
- `resume: auto` selects the highest-version ordinary `last*.ckpt` in the current run. `resume: none` starts a new run. An explicit path performs full restoration. `pretrain_ckpt_path` only warm-starts weights and must not coexist with an effective resume request.
- For an ordinary checkpoint paired with EMA and for direct EMA restoration, check optimizer moments, EMA parameters and counters, global step, scheduler, and the next actual parameter update. Successful loading does not establish full restoration or promise bitwise-identical random trajectories.
- Preserve the narrow repair in `partiflow/training/loops.py` for Lightning's completed-epoch boundary. Validate completed and partial epochs separately, avoiding skipped or repeated boundaries while retaining native partial-epoch behavior.
- Compile only forward calls while preserving module hierarchy and strict weight keys. In `partiflow/training/compilation.py`, training and evaluation forwards use separate code objects and independent specialization banks. Inspect both entry points through `forward.compiled_forwards`. Do not globally increase Dynamo cache limits or silently fall back to eager execution on failure.
- For real tiny training checks, use `prepare_data.py` to select a fixed **64 training / 16 validation** subset from the original real PepInter splits, with `configs/training_vae_acceptance.yaml` and `configs/training_fm_acceptance.yaml`. Report the actual splits, training and validation counts, optimization steps, loss values and finiteness, and checkpoint/restoration behavior. These are software acceptance checks, not model convergence or benchmark results.

## Chemical Identity and Postprocessing

Preserve production OpenMM/PyRosetta routing and mandatory chemical restoration. Select applicability for standard residues, cyclization, disulfides, and broader nonstandard chemistry through `partiflow/postprocess/chemistry.py`. Required restoration of special chemistry must not be skipped because `postprocess: false` is set.

The default standard-peptide protocol is `beta_nov16`, global backbone and side-chain mobility, fixed jumps, repacking without sequence design, one FastRelax repeat, and 200 maximum minimizer iterations. Add no coordinate or motif restraints. Preserve the existing head-to-tail and disulfide construction, required relaxation penalties, geometry checks, and PDB connection serialization; relaxation penalties must not enter the reported base metric energies. Keep explicit mobility and iteration overrides functional.

Standalone postprocessing shares one persistent Ray pool across selected tasks. Use `workers` as the concurrency cap, `threads` as the per-actor CPU budget, and the configured seed/timeout. Workers retain an immutable SDK profile: `standard_beta` for standard Rosetta work and `native_ref2015` for OpenMM/native NCAA scoring. Keep their processes separate because beta correction flags change global chemical parameters. Reset the Rosetta random stream per sample. Unfiltered OpenMM actors in the `native_ref2015` profile do not initialize its scoring SDK.

Pass one process-local PDB source/context through identity extraction, backend preparation, locator updates, and scoring. Reuse a standard sample's relaxed Pose; construct the native NCAA scoring Pose once after OpenMM. Cache immutable native parameter/alias data and matching OpenMM ForceField definitions. Build Systems and Contexts for the actual structure instead of reusing them across incompatible topologies. OpenMM CPUThreads must follow the configured per-worker thread budget.

`partiflow/postprocess/rosetta_ncaa/support_matrix.csv` currently lists 45 entries supporting native scoring at fixed coordinates; none claims Rosetta relaxation or packing support. Check the source atom graph, formal and net charge, connectivity, aliases/native patches, and actual residue identity. Score-only execution must not pack, relax, or minimize. Do not replace NCAA with canonical approximations.

BAL uses the native B3G beta backbone. The current default `LayerSelector` path has caused a native crash for this type, so the implementation marks `surface_hydrophobicity` as unavailable. This limits the current implementation path; it does not establish that all Rosetta surface calculations reject beta backbones. Preserve other computable metrics and keep missing values `unavailable`. The default interface preset requires this metric, so it cannot pass when the metric is missing. Computable energy or interface metrics do not establish physical quality.

Calculate the selected PartiFlow preset's required metrics and dependencies. Its default remains the existing four-rule `rosetta_bindcraft_peptide` preset, with thresholds and strict comparison operators from `configs/preset.yaml`; the standard protocol change does not import another project's additional rules. Keep full native energy entry points available for explicit chemistry checks.

Publish completed geometry and a small record before optional native metrics, so a C++ metric crash cannot erase successful relaxation. Keep process-local source and Pose objects out of Ray results and saved records. Preserve task-level `result.csv` publication/rollback: earlier successfully published tasks survive a later task's publication failure, while concurrent uncommitted work is stopped before rollback. Scientific filter failure retains the structure. Preserve the public identity, constraint locators, reward values, and ranks.

## Focused Validation and Releases

Choose existing tests relevant to the change. This table is a routing aid, not a mandatory full suite for every task. All filenames below are within `tests/`.

| Change | Focused tests |
| --- | --- |
| CLI and output | `test_cli_public.py`, `test_design_contract.py`, `test_workflow_contract.py` |
| Reward, sequence, and AF2 | `test_reward_configuration.py`, `test_reward_candidate_failure.py`, `test_sequence_schema.py`, `test_af2_contract.py` |
| Cached results and postprocessing | `test_validate_structure_state.py`, `test_postprocess_resume.py`, `test_postprocess_backend_override.py`, `test_postprocess_publication.py`, `test_postprocess_dispatch.py` |
| Training | `test_training_contract.py`, `test_training_epoch_resume.py`, `test_training_compile.py`, `test_data_contract.py` |
| Native chemistry | `test_native_sources.py`, `test_native_acceptance.py`, `test_postprocess_metrics.py`, `test_postprocess_routing.py` |
| Packaging | `test_package_resources.py`, `test_wheel_records.py`, `test_download_assets.py` |

For example, after a CLI-only change, run from the checkout:

```bash
PYTHONDONTWRITEBYTECODE=1 python -B -m unittest discover -s tests -p test_cli_public.py -v
```

The main thread coordinates model/GPU checks within the user's task. Inspect idle devices first, execute serially, and do not preempt or kill other processes.

For releases, verify installed resource resolution and packaged configurations and chemistry resources. Retain executable tests, small-data configurations, and vendor copyright and source-attribution files. Keep model weights, training data, run outputs, private logs and generated reports, development acceptance records, and temporary figures/assets out of release commits. Do not upload logs as repository documentation. Export release figures from canonical paper sources according to the user's selected display set, and identify their sources. Do not invent benchmarks or turn tiny training results into performance claims. Commit only within the current user's authorization; do not push automatically.

In the requested reporting language, distinguish code/configuration checks, actual loading, real execution, tiny training, quality filtering, and convergence. Report only commands executed for the current task and their observed results, with necessary code and figure source references. Do not attach historical pass tables or full machine logs.

Apply the project's source and release cleanup constraints without adding content digests, fingerprints, or tracking wrappers, including renamed or alternate-algorithm versions of the same mechanism, to owned or vendored source. Compare and retain computation-relevant chemical parameters, atoms, bonds, charges, structure mappings, configurations, and restoration state directly. Keep cleanup-check commands and project-specific exclusion lists outside the repository. Do not monkeypatch internals of externally installed dependencies.

For a formal distribution, confirm that wheel platform tags match bundled native binaries. Let build tools generate the standard `RECORD`; do not manually empty its fields to satisfy source-content cleanup constraints. A successful local pip installation does not establish compliance with publishing specifications. If an explicit zero-occurrence rule includes mandatory distribution metadata, explain the format conflict and resolve its scope before release. Do not restore application-level content-tracking wrappers.
