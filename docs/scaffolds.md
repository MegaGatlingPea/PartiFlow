# Design with scaffold conditions

Scaffold conditions describe a peptide's secondary-structure pattern and contacts between structural blocks. Use `partiflow-ss.ckpt` explicitly for these features. The general checkpoint remains the default for `design`.

The [bundled scaffold tasks](../configs/tasks/scaffold.yaml) each request one 12-residue design:

```bash
partiflow design configs/tasks/scaffold.yaml --task klk2_scaffold_linear \
  --ckpt ./ckpts/partiflow-ss.ckpt --out inference/scaffold_linear
partiflow design configs/tasks/scaffold.yaml --task klk2_scaffold_amide_cycle \
  --ckpt ./ckpts/partiflow-ss.ckpt --out inference/scaffold_amide_cycle
partiflow design configs/tasks/scaffold.yaml --task klk2_scaffold_terminal_disulfide \
  --ckpt ./ckpts/partiflow-ss.ckpt --out inference/scaffold_terminal_disulfide
```

Import `partiflow-ss` and `autoencoder` first, as described in [resources](resources.md). Add `--show-config` to inspect any example before generation. If your weights are under a separate `models/` root, use `--model-dir models --ckpt ./models/ckpts/partiflow-ss.ckpt`.

## Select a library and preset

| Library | Entries and lengths | Preset names | Category |
| --- | --- | --- | --- |
| [Linear](../configs/scaffolds/linpep_archetype_scaffold_lib.pt) | 49 entries, 6–20 residues | `alpha`, `beta_hairpin`, `double_alpha`, `random` | `linear_peptide` |
| [Amide cyclic](../configs/scaffolds/cycpep_archetype_scaffold_lib.pt) | 36 entries, 6–20 residues | `beta_hairpin`, `helical_hairpin`, `mixed_alpha_beta`, `single_helix`, `single_sheet` | `head_tail_amide_cyclic_peptide` |
| [Terminal disulfide](../configs/scaffolds/head_tail_disulfied_cyclic_archetype_scaffold_lib.pt) | 31 entries, 9–20 residues | `beta_hairpin`, `mixed_alpha_beta`, `single_helix` | `head_tail_disulfied_cyclic_peptide` |

The terminal-disulfide library joins the two terminal cysteine SG atoms. It does not create a head-to-tail N–C amide bond. Its endpoints stay fixed when choosing library entries; circular shifts apply only to entries that permit them. The amide-cyclic library supplies head-to-tail amide closure.

For example, add this block to a denovo task with a matching requested length:

```yaml
binder_topology:
  mode: sample
  lib_path: configs/scaffolds/linpep_archetype_scaffold_lib.pt
  category: linear_peptide
  preset: beta_hairpin
  systematic: true
```

Selection filters by category and preset, then requires an exact length match. `systematic: true` cycles through matching entries in order; otherwise entries are sampled using the run's random state. Preset names select library patterns; they are separate from the [AF2/Rosetta quality presets](presets.md).

If no entry matches, choose a supported preset/category and a length available for that combination. Removing a preset filter broadens selection within the chosen library. Copying a bundled task to your own file also requires updating its PDB and library paths for your working directory.

## Supply your own pattern

For an inline secondary-structure pattern, use one token per binder residue:

```yaml
binder_topology:
  mode: ss_token
  ss_token: [0, 0, 0, 0, 2, 2, 1, 1, 1, 1, 3, 3]
```

Secondary-structure tokens are `0` for helix, `1` for strand, `2` for loop, and `3` for an unspecified position. Optional `block_adj` is an L×L matrix with values `0` for no adjacency, `1` for adjacency, and `2` for unspecified pairs. A missing matrix leaves adjacency unspecified.

Use `mode: scaffold_pt` with `ss_path` and `adj_path` to load a pair of tensor files. `mode: scaffold_dir` with `scaffold_dir` selects matching `*_ss.pt` and `*_adj.pt` pairs. `mode: sample` reads a `.pt` library containing a list of entries or a mapping with an `entries` list. Each entry supplies `length`, `ss_token`, optional `block_adj`, and its pattern/category fields.

These geometry conditions leave sequence positions designable. A loop token does not fix a turn residue or a nonstandard identity. Chemical anchors, motif identities, and explicit residue constraints supply identity requirements separately. The terminal-disulfide example fixes its two anchor cysteines while `omit_aa` controls other generated positions. See [task fields](task_schema.md) and [topology](topology.md) when adding bonds or identity constraints.
