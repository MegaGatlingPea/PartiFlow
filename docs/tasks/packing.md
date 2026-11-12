# Pack peptide side chains

Packing preserves the reference peptide's sequence, conditions on its CA geometry, and predicts side-chain coordinates. The final coordinates pass through the VAE decoder.

```bash
partiflow design configs/tasks/example.yaml --task klk2_packing \
  --ckpt ./ckpts/partiflow-aux.ckpt --out inference/packing
```

Use `task_type: packing`, `is_complex: true`, and the reference `pep_chain`. The auxiliary checkpoint supports this task. Both sequence identity and backbone conditioning come from the reference complex.

Packing and subsequent relaxation are separate operations. Declare any required chemical connections in the task's [topology](../topology.md), then use [postprocessing](../commands/postprocess.md) for relaxation and optional quality criteria.
