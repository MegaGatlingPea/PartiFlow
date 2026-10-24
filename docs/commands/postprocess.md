# Postprocessing

```bash
partiflow postprocess inference/klk2 --plan
partiflow postprocess inference/klk2 --filter false --platform CPU
partiflow postprocess inference/klk2 --filter true --criteria rosetta_bindcraft_peptide
```

Inputs are one or more run/task paths, `result.csv` files, or explicit PDB files. The default processes the published design structures. `--filter` accepts exactly `true` or `false` and defaults to false. Additional options are `--config`, repeatable `--task` and `--criteria`, `--presets`, `--rerun`, `--plan`, `--platform CPU|CUDA|OpenCL`, and `--model-dir`. Runtime defaults are in [configs/postprocess.yaml](../../configs/postprocess.yaml).

`--filter false` performs relaxation without requesting quality metrics. `--filter true` preserves the relaxation-then-metrics workflow: after successful relaxation, calculate metrics and evaluate the selected criteria. A later filtered invocation reuses matching completed relaxation. `--rerun` explicitly repeats it. Quality failure does not delete the structure or turn successful relaxation into a computation failure.

The topology implementation routes registered nonstandard residues and broader covalent connections through OpenMM. Supported standard linear/cyclic/disulfide work can use Rosetta. Explicit backend requests remain strict; automatic Rosetta work can fall back to OpenMM after a runtime failure. The existing harmonic restraints and local relaxation subsets retain their configured coordinate movement. Native NCAA scoring does not enable Rosetta NCAA relaxation or packing.

A validated staged structure atomically replaces its source and refreshes the public identity and atom locators in its task's `result.csv`. Relaxation failure keeps the original structure. Reports and decisions go under `TASK/postprocess/`; quality metrics are independent of the design table's reward columns. Monomers receive `not_applicable` for an interface preset.

Native scoring uses the exact mapped residue types and fixed input coordinates, with NCAA packing disabled. All 45 registered NCAA types support actual energy scoring. The canonical filter metrics are `shape_complementarity`, `interface_hbonds`, `interface_unsat_hbonds`, and `surface_hydrophobicity`; the supported alpha-backbone types provide all four. BAL maps to Rosetta B3G; its beta backbone lacks only `surface_hydrophobicity` among those four metrics because Rosetta's LayerSelector requires an alpha backbone. The other three metrics remain available. This metric limit does not imply a loss of energy-scoring support.

The default `rosetta_bindcraft_peptide` preset requires all four metrics. Its exact thresholds and operators remain in [configs/preset.yaml](../../configs/preset.yaml). A required missing metric yields `unavailable`, so a BAL-containing binder cannot pass that default preset merely because its other metrics pass. Explicit custom criteria belong in a preset YAML supplied with `--presets`; required metrics are never silently dropped.

Passing an explicit `AF2success` directory processes those refolds into separate `postprocess/refold/` artifacts. Original predictions, validation reports, and design results remain intact, with sequence-slot paths retained in the independent reports.
