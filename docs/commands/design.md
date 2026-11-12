# Generate designs

`design` reads a YAML file containing named tasks. Select one or more names with `--task`; without it, every task in the file runs.

```bash
partiflow design configs/tasks/example.yaml --task klk2_hotspot --out inference/klk2
partiflow design configs/tasks/example.yaml --task klk2_motif --show-config
partiflow design configs/tasks/example.yaml --task klk2_docking \
  --ckpt ./ckpts/partiflow-aux.ckpt --out inference/docking
```

Choose the task's [checkpoint](../../README.md#choose-a-task-and-checkpoint) explicitly. The default is the general model. [Scaffold and topology conditioning](../scaffolds.md) uses `partiflow-ss.ckpt`; docking, inverse folding, and packing normally use `partiflow-aux.ckpt`.

## Configuration

The [default configuration](../../configs/design.yaml) composes generation, search, and optional reward settings. `--config FILE` adds a YAML overlay. Repeat `--override KEY=VALUE` for Hydra updates:

```bash
partiflow design configs/tasks/example.yaml --task klk2_hotspot \
  --override seed=2026 --batch-size 4 --num-workers 0 \
  --compile false --out inference/klk2_small
```

A task's non-null `sample_num` takes priority over the global value, so change the task YAML when you need a different count for an existing task. Requested lengths multiply that count: `klk2_hotspot` requests one sample at each length from 11 through 20.

Named options include `--ckpt`, `--out`, `--run-name`, `--seed`, `--batch-size`, `--num-workers`, `--compile true|false`, and `--model-dir`. A named option and an override cannot both set the same configuration field. The defaults use batch size 16, 16 data-loader workers, and compilation disabled. `--show-config` prints the resolved settings and exits before generation.

`--out` chooses the exact run directory. Otherwise the default is `inference/design`, formed from `output.root_dir` and `output.run_name`. Each selected task directory must be absent or empty. For another run, choose a fresh output directory or a new run name.

## Read the output

Each task has its own directory:

```text
inference/klk2/klk2_hotspot/
  sample1.pdb
  sample2.pdb
  ...
  result.csv
  resolved_config.yaml
  logs/
```

Raw generation uses `sample<base>.pdb`. Base IDs start at one and span the task's expanded length and condition plan. [Best-of-N search](../reward.md) writes ranked candidates as `sample<base>_<chain>_rank<rank>.pdb`.

| Result field | Meaning |
| --- | --- |
| `filename` | Published PDB; empty if no structure was produced |
| `base_sample_id`, `token_length` | Planned sample identity and generated binder length |
| `design_sequence`, `design_chain_id`, `design_chains` | Sequence and chain identities in the published structure |
| `reward_method`, `global_rank` | Optional reward and rank within the task/length group |
| `contig_string`, `tip_atoms` | Retained task and geometric conditioning information |
| `fixed_restypes`, `bond_constraints` | Fixed identities and declared bonds, stored as JSON arrays |
| `postprocess_status`, `error` | Relaxation outcome and any failure details |

Without reward metrics, `result.csv` has 15 columns. Numeric reward metrics are added before `postprocess_status` and `error`; missing values stay blank. Failed planned samples remain in the table. Use the `filename` field to find actual structures.

Task postprocessing runs after reward calculation. Reward columns therefore describe the reward prediction before subsequent relaxation. Independent validation and postprocessing have their own metric reports. To assess saved candidates, continue with [AF2 validation](validate.md) or [relaxation and quality filtering](postprocess.md).
