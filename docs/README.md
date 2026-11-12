# Using PartiFlow

Start with [installation and resources](resources.md) and the [one-sample design in the README](../README.md#make-a-first-design). Then choose the workflow you need:

| Command | Guide | What it does |
| --- | --- | --- |
| `design` | [Generate candidates](commands/design.md) | Run named tasks from a task YAML |
| `validate` | [AF2 validation](commands/validate.md) | Compare bound-complex and binder-monomer predictions |
| `postprocess` | [Relaxation and filtering](commands/postprocess.md) | Restore declared chemistry, relax structures, and optionally apply quality rules |
| `cluster` | [Structural clustering](commands/cluster.md) | Group selected binder backbones with US-align |
| `train` | [Training](commands/train.md) | Train the VAE or flow-matching model |
| `check` | [Resource checks](commands/check.md) | Inspect local files or explicitly test model loading/execution |
| `download` | [Download resources](commands/download.md) | Prepare model weights and training data |

The five task families share the same [task YAML format](task_schema.md): [denovo](tasks/denovo.md), [motif](tasks/motif.md), [docking](tasks/docking.md), [inverse folding](tasks/inverse-folding.md), and [packing](tasks/packing.md).

Use [scaffold conditions](scaffolds.md) to request secondary-structure patterns or adjacency, [topology](topology.md) to declare chemical bonds, and [Best-of-N reward](reward.md) to rank generated candidates. The [quality preset guide](presets.md) explains AF2 and Rosetta filtering. [Performance](performance.md) contains the original study figures.
