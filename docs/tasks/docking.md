# Dock a known sequence

Docking fixes the complete peptide sequence and generates its coordinates in the receptor context. Select the auxiliary checkpoint for ordinary docking:

```bash
partiflow design configs/tasks/example.yaml --task klk2_docking \
  --ckpt ./ckpts/partiflow-aux.ckpt --out inference/docking
```

Set `task_type: docking`. The sequence comes from the reference `pep_chain` in a complex, or from `sequence` when using a receptor/context-only input. A sequence such as `HK[MK8]LHQ[MK8]LQDS` retains the registered bracketed residue identities.

Declare crosslinks separately through [topology](../topology.md): a residue code supplies identity, while a topology entry supplies its inter-residue bond. The [CPSet task file](../../configs/tasks/cpset.yaml) includes known-sequence docking examples with different peptide chains and chemical connections.

`partiflow-dock.ckpt` is the docking specialist used for the CPSet-separated study comparison. CPSet remains a collection of docking tasks. Check the peptide chain and bond declarations for each entry before adapting it.
