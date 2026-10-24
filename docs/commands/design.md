# Design

```bash
partiflow design configs/tasks/example.yaml --task klk2_hotspot --out inference/klk2
partiflow design configs/tasks/example.yaml --task klk2_docking --ckpt ckpts/partiflow-aux.ckpt --out inference/docking
partiflow design configs/tasks/example.yaml --task klk2_motif --show-config
```

The positional file contains named [tasks](../task_schema.md). Repeat `--task NAME` to select several entries; omitting it selects every entry in that file. `--config FILE` merges a design configuration, and repeatable `--override KEY=VALUE` options update the Hydra configuration. Named options are `--ckpt`, `--out`, `--run-name`, `--seed`, `--batch-size`, `--num-workers`, `--compile true|false`, and `--model-dir`. A named option and an override cannot both set its mapped field. `--show-config` resolves configuration without generation or full task compilation.

[configs/design.yaml](../../configs/design.yaml) composes generation, search, and optional reward settings. `--out` selects the exact run directory. Otherwise `output.root_dir` and `output.run_name` determine it. Selected task directories must be absent or empty at launch. Each task keeps its final PDB structures, `result.csv`, `resolved_config.yaml`, logs, and private workflow state.

Raw sampling writes `sample<base>.pdb`. Optional [best-of-N AF2 reward](../reward.md) retains ranked candidates as `sample<base>_<chain>_rank<rank>.pdb`. Base IDs start at one within each task and span its expanded length plan. Failed planned bases remain represented in the result table; a failed row does not invent a structure filename.

The raw `result.csv` header has exactly 15 columns, in this order:

```csv
filename,design_chains,base_sample_id,design_modality,design_chain_id,token_length,reward_method,global_rank,design_sequence,contig_string,tip_atoms,fixed_restypes,bond_constraints,postprocess_status,error
```

The writer inserts actual numeric reward metric columns, sorted by name, between `bond_constraints` and `postprocess_status`. The 13 leading fields and two trailing fields retain their order. Missing metrics stay blank. `design_chains`, `fixed_restypes`, and `bond_constraints` contain JSON arrays tied to the final published structure. Identity constraints and declared bond endpoints use its chain, residue, and atom locators; successful unconstrained samples use empty constraint arrays. Independent validation and postprocessing reports keep their own metrics and decisions.

Raw sampling and optional reward ranking share the generation implementation. Task postprocessing retains the production OpenMM/Rosetta chemistry routes and runs after reward calculation. Reward metrics therefore describe the reward prediction before subsequent relaxation. A completed generation does not establish structural quality or binding; use [validate](validate.md) and explicit [postprocess criteria](postprocess.md) for their computational assessments.
