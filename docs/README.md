# PartiFlow documentation

Start with [installation and resources](resources.md), then choose a command. Commands work from an installed package; model files and user inputs remain outside the package.

| Command | Guide | Purpose |
| --- | --- | --- |
| `train` | [Training](commands/train.md) | Train the VAE or flow-matching model |
| `design` | [Design](commands/design.md) | Generate candidates from a task YAML |
| `validate` | [Validation](commands/validate.md) | Paired AF2 initial-guess complex and monomer assessment |
| `postprocess` | [Postprocessing](commands/postprocess.md) | Topology-aware relaxation and optional quality criteria |
| `cluster` | [Clustering](commands/cluster.md) | Binder structural diversity with US-align |
| `check` | [Checks](commands/check.md) | Resource inspection and explicit model execution checks |
| `download` | [Downloads](commands/download.md) | Configured downloads or local asset imports |

The [task schema](task_schema.md) documents the preserved PartiFlow task syntax. Task examples are organized by [denovo](tasks/denovo.md), [motif](tasks/motif.md), [docking](tasks/docking.md), [inverse folding](tasks/inverse-folding.md), and [packing](tasks/packing.md). See [topology](topology.md) for covalent and nonstandard chemistry, [search and reward](reward.md) for optional best-of-N selection, and [historical performance](performance.md) for the original figures.
