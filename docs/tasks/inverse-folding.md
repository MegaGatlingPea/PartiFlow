# Design a sequence on a fixed backbone

Inverse folding conditions on the reference peptide's CA geometry and designs a compatible sequence and side chains. The final coordinates pass through the VAE decoder.

```bash
partiflow design configs/tasks/example.yaml --task klk2_invfold \
  --ckpt ./ckpts/partiflow-aux.ckpt --out inference/inverse_folding
```

Use `task_type: inv_folding`, `is_complex: true`, and the reference `pep_chain`. Optional motif settings retain selected residue identities as well. The auxiliary checkpoint supports this combination of fixed backbone and generated sequence.

Later relaxation may move coordinates under its own settings. Read the generated sequence and final chain identities from `result.csv`; use [validation](../commands/validate.md) when assessing the designed sequence. The [task fields](../task_schema.md) describe hotspot selection, additional identity constraints, and sample counts.
