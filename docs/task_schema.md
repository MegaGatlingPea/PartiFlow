# Write a task YAML

A task file maps names to task settings. Select names with `design --task NAME`; omit `--task` to run all entries.

```yaml
my_binder:
  task_type: denovo
  pdb_path: ./inputs/complex.pdb
  is_complex: true
  pep_chain: L
  hotspots: reference
  motif_cfg: null
  lens_cfg: {nres: [12, 16], min_len: null, max_len: null, step_len: null}
  sample_num: 5
  postprocess: false
```

This requests five candidates at each of two lengths. The five task types are [denovo](tasks/denovo.md), [motif](tasks/motif.md), [docking](tasks/docking.md), [inv_folding](tasks/inverse-folding.md), and [packing](tasks/packing.md). Scaffold conditioning is an option within these tasks, rather than another task type.

## Input structures and positions

`pdb_path` points to a receptor or receptor–peptide complex. Relative paths in your own task files start at the directory where you invoke PartiFlow. Bundled tasks resolve their bundled structures and scaffold libraries from the package resources.

Set `is_complex: true` when the input includes a reference peptide, and identify it with `pep_chain` (default `L`). Motif, inverse-folding, and packing tasks need this reference. Docking can instead use a receptor-only input and an explicit `sequence`.

`hotspots` accepts:

- `null` for no hotspot condition.
- `reference` to derive the interface from the reference peptide.
- PDB residue selectors such as `A124-125/A153` to select source-chain residues.

Source selectors use the input PDB's chain IDs and residue numbering. Constraint and topology positions use generated binder positions, starting at one.

## Lengths and sample counts

For denovo design, use exact `lens_cfg.nres` values or `min_len`, `max_len`, and `step_len`. Motif tasks use the length range and generated spans in `motif_cfg`; leave `lens_cfg` null.

`sample_num` is a positive count for each expanded length/condition. A task value takes priority; the global design value fills a missing or null task value. Global `binder_secondary_class`, `binder_topology`, `condition_sampler`, and `omit_aa` settings apply to all selected tasks when provided.

## Sequence and geometry conditions

`constraint.restype` fixes residue identities, and `constraint.spatial` requests CA distances in angstroms:

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

The distance conditions guide generation; the topology declaration creates the chemical connection during postprocessing. Keep residue identities and bond anchors consistent. Registered nonstandard residues use their registry codes in identity constraints. Explicit docking sequences use bracketed codes, for example `HK[MK8]LHQ[MK8]LQDS`.

`binder_secondary_class` accepts `alpha`, `beta`, or `coil`. `binder_topology` provides detailed secondary-structure and adjacency conditions, including library selection. Use `partiflow-ss.ckpt` explicitly for these model capabilities; see [scaffold conditions](scaffolds.md).

## Motifs and topology sampling

A motif contig such as `2-8/L6-11/2-8` retains source residues L6–L11 and generates flanking spans. `motif_cfg.contigs: reference` samples source motif segments using its percentage and segment-count settings. See the [motif guide](tasks/motif.md) for an example.

The randomized `sampler` choices include `head_to_tail`, `disulfide`, `isopep`, `bicyclic`, `alkane_staple`, and `alkene_staple`. Do not combine `sampler` with `condition_sampler`. Nonempty explicit `constraint` settings take priority over randomized sampling. In motif tasks, sampled anchors must lie in generated positions; explicit constraints must agree with retained motif identities.

Use explicit [topology](topology.md) when you need bonds at particular positions. `constraint.cyclize` can also request shortcut geometry using `begin_resid`, `end_resid`, and `offset`; `last` selects a supported terminal position.

## Postprocessing a task

`postprocess` accepts a boolean or a mapping with `enabled`, `method`, and `relax_linear`. Ordinary linear generation skips relaxation unless requested. `postprocess: true` requests it. Special topology and registered nonstandard chemistry require restoration, so they cannot explicitly disable postprocessing.

`method: auto` chooses a supported route. Explicit routes are `openmm` and `rosetta_topology`. Global `postprocess_runtime` settings control execution details. Later, the standalone [postprocess command](commands/postprocess.md) can relax published structures and optionally apply quality criteria with `--filter true`.
