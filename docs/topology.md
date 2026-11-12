# Chemical topology and nonstandard residues

Use `topology` to declare chemical connections in the generated peptide. Constraint distances guide geometry, residue constraints fix identities, and topology entries specify the bonds to restore.

For a head-to-tail amide ring:

```yaml
topology:
  - {type: head_tail, begin_resid: 1, end_resid: last}
postprocess: {enabled: true, method: auto}
```

For a disulfide at positions 3 and 7:

```yaml
constraint:
  restype:
    - {resid: 3, aa: C}
    - {resid: 7, aa: C}
  spatial:
    - {begin_resid: 3, end_resid: 7, distance: 5.5}
topology:
  - {type: disulfide, begin_resid: 3, end_resid: 7}
postprocess: {enabled: true, method: auto}
```

These positions refer to the generated binder, starting at one. Source hotspot and motif selectors instead use the input PDB numbering. A terminal cysteine disulfide joins SG atoms and is distinct from an N–C head-to-tail amide bond; the [scaffold guide](scaffolds.md) includes both kinds of examples.

## Connection types

| Task declaration | Connection |
| --- | --- |
| `head_tail` or `head_to_tail` | Head-to-tail amide closure |
| `disulfide` | Cysteine SG–SG bond |
| `isopep` | Supported side-chain amide connection |
| `bicyclic` | Supported three-anchor bicycle chemistry |
| `mk8_staple` | Registered MK8 hydrocarbon staple |
| `nle_staple` | Registered NLE hydrocarbon staple |
| `residue_template` | Registered residue replacement |

Registered RTY connections are also supported. The [example tasks](../configs/tasks/example.yaml) and [CPSet docking tasks](../configs/tasks/cpset.yaml) show full declarations with the required identities and anchors. When changing a task, keep its chemical residue types, anchors, and distance conditions consistent.

For example, a pair of MK8 residues needs both identities and its bond declaration:

```yaml
constraint:
  restype:
    - {resid: 3, aa: MK8}
    - {resid: 7, aa: MK8}
topology:
  - {type: mk8_staple, anchors: [3, 7]}
```

An explicit docking sequence can use bracketed registered codes. Naming the residues alone does not create the staple.

## Relaxation and assessment

Rosetta supports applicable standard linear, head-to-tail, and disulfide relaxation. Broader covalent chemistry and registered nonstandard residues use OpenMM. Special chemistry requires restoration during design, so it cannot explicitly disable postprocessing or force an incompatible backend.

The final structures retain the declared atom identities and connections. Relaxation can move coordinates according to the selected backend and restraints. Use the final PDB and its updated result-table identity fields when examining a candidate.

The [native NCAA matrix](../partiflow/postprocess/rosetta_ncaa/support_matrix.csv) lists 45 fixed-coordinate energy-scoring types. It does not provide Rosetta relaxation or packing for those types. BAL uses B3G beta-backbone chemistry; its missing `surface_hydrophobicity` metric prevents passing the default four-rule interface preset. Other supported metrics can still be reported.

AF2 accepts representable standard protein chemistry. Unsupported nonstandard residues, ligands, atoms, or bonds are reported explicitly; they are not replaced by canonical residues. Use [postprocessing](commands/postprocess.md) and an appropriate [quality preset](presets.md) for supported chemical assessment.
