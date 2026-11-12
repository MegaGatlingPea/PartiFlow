# Denovo design

Denovo design generates a peptide sequence and all-atom coordinates from receptor context, a requested length, and optional hotspots.

```bash
partiflow design configs/tasks/example.yaml --task klk2_hotspot --out inference/denovo
```

This bundled task requests one candidate at each length from 11 through 20. For a single candidate, use the [one-length task in the README](../../README.md#make-a-first-design).

Set `task_type: denovo`, `pdb_path`, and `lens_cfg`. Use `nres` for exact lengths or a `min_len`/`max_len`/`step_len` range. `hotspots` can be null, derived from a reference peptide, or specified with source residue selectors such as `A124-125/A153`.

The general checkpoint is the default. For secondary-structure, scaffold-topology, or disulfide conditioning, explicitly select `partiflow-ss.ckpt` and follow the [scaffold guide](../scaffolds.md). For chemical bonds and sampled topologies, see [topology](../topology.md) and the [shared task fields](../task_schema.md).
