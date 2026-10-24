# Motif scaffolding

Motif tasks preserve selected sequence and all-atom motif coordinates from a reference complex while generating the surrounding scaffold.

```bash
partiflow design configs/tasks/example.yaml --task klk2_motif --out inference/motif
partiflow design configs/tasks/motif.yaml --task peptide01_top01_motif --show-config
```

Set `task_type: motif`, `is_complex: true`, and the reference `pep_chain`. An explicit `motif_cfg` such as `{contigs: 2-8/L6-11/2-8, min_len: 12, max_len: 20}` retains source residues L6-11 and expands compatible generated flanks. Motif lengths come from `motif_cfg`; leave `lens_cfg` null. With `contigs: reference`, the percentage/segment settings control motif selection. Topology sampler anchors stay within generated/redesignable positions; explicit constraints cannot conflict with fixed motif positions. See [task schema](../task_schema.md).

[configs/tasks/motif.yaml](../../configs/tasks/motif.yaml) contains reference-derived peptide motif examples using the same `design` command and task grammar.
