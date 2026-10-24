# Structural clustering

```bash
partiflow cluster inference/klk2 --plan
partiflow cluster inference/klk2 --tm-threshold 0.5 --workers 16
partiflow cluster inference/klk2 --criteria af2_ig_peptide --usalign /path/to/USalign
```

Run validation with an explicit criterion before clustering an AF2 cohort. Inputs can be run/task paths or their saved AF2 success tables and directories. The command selects passing sequence branches from `TASK/validate/af2_ig/success.csv`, deduplicates their generated backbones, and compares the current original design structures. It does not cluster AF2 refolds by default. A validation run with no criteria has no saved passing cohort.

Repeat `--task` and `--criteria` to select tasks and criteria. `--tm-threshold` defaults to 0.5 and must be in `(0, 1]`. `--workers` defaults to 16 and accepts 1–16; each active directory worker uses one CPU thread. `--plan` inspects selected work without alignment. The prepared environment supplies Bioconda USalign 2024.07.30. `--usalign` selects an executable explicitly; otherwise the command looks beside the active Python interpreter, then on `PATH`.

Clustering uses binder structures, the mean of the two length-normalized TM scores, and greedy representative selection. Each saved criterion is clustered independently. Reports are `clusters.csv` and `cluster_summary.csv` in `TASK/validate/af2_ig/`. To select saved postprocessing results, pass `TASK/postprocess/` explicitly; those reports go into `TASK/postprocess/clusters/`.

A cluster count describes structural diversity within the selected computational cohort. It does not establish biological binding. Historical figures may use a different TM threshold or denominator; reproduce those settings explicitly when comparing with [the original performance plots](../performance.md).
