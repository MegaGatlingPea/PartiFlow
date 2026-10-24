# Task schema

A task YAML is a mapping from task names to task dictionaries. The five task types and their existing PartiFlow semantics are preserved. Select entries with repeatable `design --task NAME` options.

```yaml
my_binder:
  task_type: denovo
  pdb_path: ./pdbs/design/klk2/klk2_motif.pdb
  is_complex: true
  pep_chain: L
  hotspots: reference
  motif_cfg: null
  lens_cfg: {nres: [15, 20], min_len: null, max_len: null, step_len: null}
  sample_num: 5
  binder_secondary_class: null
  binder_topology: null
  omit_aa: null
  constraint:
    spatial:
      - {begin_resid: 3, end_resid: 7, distance: 5.5}
    restype:
      - {resid: 3, aa: C}
      - {resid: 7, aa: C}
  topology:
    - {type: disulfide, begin_resid: 3, end_resid: 7}
  postprocess: {enabled: true, method: auto, relax_linear: false}
```

The template demonstrates fields rather than guaranteed physical geometry. `pdb_path` is a receptor or receptor-binder complex PDB; user-relative paths resolve from the invocation directory. `is_complex: true` identifies a reference peptide in the source and is required by motif, inverse-folding, and packing tasks. `pep_chain` locates that reference chain and defaults to `L`. Docking can instead supply `sequence` with receptor/context-only input.

`hotspots` accepts null, `reference`, or slash-separated PDB residue selectors such as `A124-125/A153`. `reference` requires a reference peptide; explicit selectors use source PDB numbering.

`lens_cfg.nres` lists exact requested denovo lengths. Alternatively use `min_len`, `max_len`, and `step_len`. Motif lengths come from `motif_cfg` and contig expansion, not `lens_cfg`. A contig such as `2-8/L6-11/2-8` combines generated spans with fixed source residues `L6-11`. `motif_cfg.contigs: reference` samples motif segments using its percentage and segment-count settings.

`sample_num` is the number of samples per expanded length/condition. A positive task value wins; a global value fills only a missing/null task value. Global `binder_secondary_class`, `binder_topology`, `condition_sampler`, and `omit_aa` overrides apply to selected tasks. Supported coarse secondary classes are `alpha`, `beta`, and `coil`.

`constraint.restype` fixes generated-position identities; registered nonstandard residues use their registry codes. Docking `sequence` uses canonical one-letter identities and bracketed nonstandard identities, for example `HK[MK8]LHQ[MK8]LQDS`. `constraint.spatial` declares CA-distance conditions in angstroms. The optional `constraint.cyclize` entries use `begin_resid`, `end_resid`, and `offset`; `last` is accepted where terminal positions are supported. Fixed motif positions cannot also receive conflicting residue constraints.

Randomized `sampler` values include `head_to_tail`, `disulfide`, `isopep`, `bicyclic`, `alkane_staple`, and `alkene_staple`. Do not combine `sampler` with `condition_sampler`. Nonempty explicit `constraint` takes precedence over randomized sampling. Use explicit [topology](topology.md) when anchors must occupy particular positions. Sampler anchors in motif tasks are restricted to generated/redesignable positions.

`postprocess` accepts a boolean or a mapping with task-local `enabled`, `method`, and `relax_linear`. A missing setting skips ordinary linear relaxation by default; explicit `postprocess: true` requests relaxation. Special topology and registered nonstandard chemistry require restoration and cannot be disabled explicitly. `method: auto` selects the supported production backend; explicit methods are `openmm` and `rosetta_topology`. Global `postprocess_runtime` settings control runtime details such as platform and relaxation subset. Standalone [postprocess](commands/postprocess.md) processes published outputs independently, with `--filter true` adding quality metrics after successful relaxation.

The existing low-level controls remain available for configs that need them: `residue_type_constraints: [{pos: 18, aa: C}]`, `optional_ca_pair_constraints: [{pos_i: 18, pos_j: 21, bin_idx: 5}]`, and `cyclization: {mode: pair, positions: [1, 20], shortcut_len: 1}`. Prefer the grouped `constraint` fields for new task files and avoid declaring inconsistent copies of the same condition.
