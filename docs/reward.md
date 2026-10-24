# Best-of-N search and AF2 reward

Raw sampling uses `search.algorithm: null` and no reward. Optional `best_of_n` expands each planned input into the configured number of complete-trajectory replicas. The search and reward configurations compose through the same design command.

```bash
partiflow design configs/tasks/example.yaml --task klk2_hotspot \
  --override search=best_of_n --override reward=af2 \
  --override search.best_of_n.replicas=4 --out inference/klk2_best_of_n
```

`search.best_of_n.replicas` is a positive integer and `search.selection_scope` is `task_length_plan`. Base sample identities span the task's expanded length plan. Successfully evaluated replicas are retained and ranked within each task and actual binder length; rank 1 has the highest available reward in that group. This does not automatically select a validation-passing candidate or reduce publication to one structure per base. See [design outputs](commands/design.md) for filenames and the ordered CSV contract.

[configs/reward/af2.yaml](../configs/reward/af2.yaml) defaults to the generated sequence (`sequence.source: design`) and a Multimer v3 initial-guess prediction with no MSA search. Its default score is `0.25 * (design_plddt / 100) + 0.75 * interface_iptm`. `structure_source: fold` publishes the folded candidate before task postprocessing; `structure_source: design` retains the design coordinates and rejects a changed sequence on unchanged side chains.

The shared sequence policy can retain the generated sequence or use ProteinMPNN/SolubleMPNN while preserving fixed identities. AF2 reward supports standard protein chemistry and reports unsupported chemistry explicitly; it does not substitute canonical identities for nonstandard residues. Reward calculation precedes task relaxation, so its numeric CSV columns retain the pre-relaxation values. Missing values remain blank and actual metric names are sorted before the two trailing CSV fields.

Reward ranking and later quality criteria are separate operations. Standalone [validation](commands/validate.md) performs the paired complex/monomer protocol and evaluates named criteria; its default sequence sources differ from the reward defaults. Model weights are provisioned through [local resources](resources.md), with no runtime downloads.
