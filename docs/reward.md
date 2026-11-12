# Raw sampling and Best-of-N reward

Raw sampling is the default: `search.algorithm: null` and no reward. It generates the candidates requested by the task's length and sample plan.

Best-of-N runs several full generation trajectories for each planned sample and ranks the evaluated candidates with AF2 reward:

```bash
partiflow design configs/tasks/example.yaml --task klk2_hotspot \
  --override search=best_of_n --override reward=af2 \
  --override search.best_of_n.replicas=4 --out inference/klk2_best_of_n
```

`search.best_of_n.replicas` is a positive integer. The shipped Best-of-N configuration defaults to one replica, so set it to the number you want. Four replicas on the bundled task's ten requested lengths plan 40 trajectories.

Successfully evaluated candidates are retained and ranked within each task and actual peptide length. Rank 1 has the highest available reward in that group. The output is not reduced to a single winner per base sample. Filenames use `sample<base>_<chain>_rank<rank>.pdb`; `result.csv` records base identities, ranks, measured reward values, and failures.

## AF2 reward settings

The [default reward](../configs/reward/af2.yaml) uses the generated sequence and an AF2 Multimer v3 prediction with initial guess, model 1, three recycles, and no MSA search. It combines:

```text
reward = 0.25 × (design_plddt / 100) + 0.75 × interface_iptm
```

`structure_source: fold` publishes the reward-predicted structure before task postprocessing. `structure_source: design` keeps the design coordinates and requires its sequence to remain compatible with those side chains. The shared sequence policy can retain the generated sequence or use ProteinMPNN/SolubleMPNN while respecting fixed identities.

Prepare the AF2 parameters before running reward. Import MPNN weights as well if the selected sequence policy uses them; see [resources](resources.md).

Reward is calculated before task relaxation, so its stored metrics keep that meaning after relaxation. Later [paired AF2 validation](commands/validate.md) uses a different protocol and can apply [quality presets](presets.md). A reward rank by itself does not imply that a candidate passed those criteria.

AF2 reward supports representable standard protein chemistry. Unsupported residues, atoms, ligands, or bonds are reported explicitly rather than converted to canonical identities.
