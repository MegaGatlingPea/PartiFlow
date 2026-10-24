# Side-chain packing

Packing fixes the binder sequence and CA coordinates and generates its all-atom side-chain coordinates.

```bash
partiflow design configs/tasks/example.yaml --task klk2_packing \
  --ckpt ckpts/partiflow-aux.ckpt --out inference/packing
```

Set `task_type: packing`, `is_complex: true`, and the reference `pep_chain`. Both binder identity and backbone conditioning come from the reference structure. Model-based packing and optional topology-aware relaxation are separate stages with separate coordinate constraints. See [task schema](../task_schema.md) and [postprocessing](../commands/postprocess.md).
