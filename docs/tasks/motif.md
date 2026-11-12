# Motif scaffolding

A motif task uses selected sequence identities and all-atom geometry from a reference peptide to condition the surrounding scaffold. Use the general PartiFlow checkpoint for the bundled motif examples.

```bash
partiflow design configs/tasks/example.yaml --task klk2_motif --out inference/motif
partiflow design configs/tasks/motif.yaml --task peptide01_top01_motif --show-config
```

Coordinates pass through the VAE decoder, so the generated motif geometry can differ slightly from the reference. Subsequent relaxation can move it further.

The input must contain the reference peptide:

```yaml
my_motif:
  task_type: motif
  pdb_path: ./inputs/complex.pdb
  is_complex: true
  pep_chain: L
  hotspots: reference
  motif_cfg: {contigs: 2-8/L6-11/2-8, min_len: 12, max_len: 20}
  lens_cfg: null
  sample_num: 1
```

`L6-11` retains those source residues; `2-8` allows a generated flank in that length range. `motif_cfg` controls the resulting length plan. With `contigs: reference`, the motif percentage and segment-count settings instead choose source segments.

Sampled topology anchors are restricted to generated positions. Identity constraints must agree with retained motif residues. Optional relaxation has its own coordinate-movement settings; the motif's generation constraints and later relaxation are separate stages.

The [motif task file](../../configs/tasks/motif.yaml) contains further reference-derived examples. Detailed secondary-structure and adjacency conditions can be added with the specialized checkpoint described in [scaffold conditions](../scaffolds.md).
