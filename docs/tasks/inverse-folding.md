# Inverse folding

Inverse folding fixes binder CA coordinates from a reference complex and generates a compatible sequence and all-atom side chains.

```bash
partiflow design configs/tasks/example.yaml --task klk2_invfold \
  --ckpt ckpts/partiflow-aux.ckpt --out inference/inverse_folding
```

Set `task_type: inv_folding`, `is_complex: true`, and the reference `pep_chain`. Optional motif settings retain selected residue identities. Preserve the existing reference backbone/identity constraints; later postprocessing may move coordinates according to its own configured restraints. The [task schema](../task_schema.md) explains hotspots, explicit identity constraints, and sample counts.
