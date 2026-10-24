# Denovo generation

Denovo tasks generate binder sequence and coordinates from receptor context, optional hotspots, and a length schedule.

```bash
partiflow design configs/tasks/example.yaml --task klk2_hotspot --out inference/denovo
```

Use `task_type: denovo`, a source `pdb_path`, and `lens_cfg` with exact `nres` values or a min/max/step interval. `sample_num` applies to each expanded length condition. `hotspots` can be null, reference-derived, or explicit source residue selectors. Registered topology samplers can add conditions at redesignable binder positions. See the [shared task schema](../task_schema.md).
