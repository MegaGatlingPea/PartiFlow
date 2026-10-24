# Explicit topology and nonstandard residues

Topology declares chemical connections; residue names alone do not create inter-residue bonds. Keep fixed identities, CA-distance conditions, topology type, and anchors consistent. Task anchor positions refer to generated binder positions; source hotspot selectors retain source PDB numbering.

```yaml
topology:
  - {type: head_tail, begin_resid: 1, end_resid: last}
  - {type: disulfide, begin_resid: 3, end_resid: 7}
postprocess: {enabled: true, method: auto}
```

Supported declarations include head-to-tail closure, disulfide, isopeptide/stapled connections, three-anchor bicycle chemistry, registered RTY connections, and MK8/NLE hydrocarbon staples. `head_tail` and `head_to_tail` identify the same closure. `mk8_staple` and `nle_staple` retain their executable registry names. Passive residue replacement uses `residue_template` with a position and registered `resname`.

```yaml
constraint:
  restype:
    - {resid: 3, aa: MK8}
    - {resid: 7, aa: MK8}
topology:
  - {type: mk8_staple, anchors: [3, 7]}
```

The production OpenMM path applies the declared graph, registered templates, atom/charge assignments, and geometric checks. Rosetta supports applicable standard linear, head-to-tail, and disulfide relaxation; broader chemistry remains OpenMM-owned. Backend restraints and local-subset choices retain their existing meanings and permit their configured coordinate movement. Mandatory special chemistry cannot opt out through `postprocess: false` or an incompatible explicit backend.

All 45 rows in the [native NCAA support matrix](../partiflow/postprocess/rosetta_ncaa/support_matrix.csv) support fixed-coordinate energy scoring and declare no Rosetta relaxation/packing support. Strict loading retains project aliases and native patches, checks real atom identities and source graphs/charges, restores declared connections, and compares coordinates directly. These scoring rules do not replace the production chemistry-restoration route.

Native interface metrics are calculated without NCAA packing. BAL uses native B3G beta-backbone chemistry and lacks only `surface_hydrophobicity` among the four canonical filter metrics; shape complementarity, interface hydrogen bonds, and interface unsatisfied hydrogen bonds remain available. The default interface preset requires the missing metric and therefore reports `unavailable` for that case. See [postprocessing](commands/postprocess.md) for metric and criterion behavior. Scoring computability does not establish physical quality, and AF2 validation does not silently convert this chemistry to a standard residue.
