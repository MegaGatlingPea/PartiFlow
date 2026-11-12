# Cluster binder structures

Clustering groups selected binder backbones by structural similarity using US-align. First validate with an explicit criterion so the run has a saved passing cohort:

```bash
partiflow validate inference/first --criteria af2_ig_peptide
partiflow cluster inference/first --plan
partiflow cluster inference/first --criteria af2_ig_peptide --tm-threshold 0.5 --workers 16
```

Run/task paths select passing sequence branches from `TASK/validate/af2_ig/success.csv`. Repeated branches referring to one generated backbone are deduplicated. Alignments use the current original design structures, while AF2 supplies the cohort selection. A metrics-only validation run with no criteria has no passing cohort to cluster.

Repeat `--task` and `--criteria` to select tasks and saved criteria. `--tm-threshold` defaults to 0.5 and accepts `(0, 1]`. `--workers` defaults to 16 and accepts 1–16; each active directory worker uses one CPU thread. `--plan` inspects the selected work without running alignment.

The prepared environment supplies USalign 2024.07.30. Use `--usalign FILE` for an explicit executable; otherwise PartiFlow looks beside the active Python interpreter and then on `PATH`.

The similarity score is the mean of the two length-normalized TM scores. Greedy representative selection groups each saved criterion independently. Reports are `clusters.csv` and `cluster_summary.csv` under `TASK/validate/af2_ig/`.

To cluster a saved postprocessing cohort instead, select its directory explicitly:

```bash
partiflow cluster inference/first/first_binder/postprocess --plan
partiflow cluster inference/first/first_binder/postprocess --criteria rosetta_bindcraft_peptide
```

These reports go under `TASK/postprocess/clusters/`. Cluster counts describe structural diversity in the selected cohort. When comparing with the [original performance figures](../performance.md), use their original thresholds and cohort definitions.
