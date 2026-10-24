# Known-sequence docking

Docking fixes the complete binder identity and generates receptor-bound coordinates. Use the auxiliary checkpoint for ordinary practical docking; the docking specialist serves its CPSet-separated comparison scope.

```bash
partiflow design configs/tasks/example.yaml --task klk2_docking \
  --ckpt ckpts/partiflow-aux.ckpt --out inference/docking
```

Set `task_type: docking`. The binder sequence comes from the reference `pep_chain` in a complex, or from explicit `sequence` with receptor/context-only input. Explicit sequences retain registered bracketed identities such as `HK[MK8]LHQ[MK8]LQDS`. Every docking binder position is a hard identity condition. Declare inter-residue chemistry through [topology](../topology.md); a bracketed residue alone does not create a crosslink.

CPSet is a collection of docking tasks, not an additional task family. Each entry's peptide chain and chemistry declarations are authoritative; chain L is not universal.
