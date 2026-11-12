# Check local resources

Start with a read-only file check:

```bash
partiflow check
partiflow check configs/tasks/example.yaml --model partiflow --model autoencoder
partiflow check --all --model-dir models
```

The default checks packaged resources and the `partiflow` and `autoencoder` checkpoint paths. `--model NAME` is repeatable; `--all` checks every configured model target. They are mutually exclusive. An optional task YAML also checks basic task settings and input structure paths.

Normal checks do not load models, initialize accelerators, or download files. A `present` result means the required files were found. Use explicit operations when you want to go further:

```bash
partiflow check --model proteinmpnn --load
partiflow check --model partiflow --model autoencoder --smoke
```

`--load` constructs selected models on CPU with their strict loaders. `--smoke` runs small diagnostic computations, including reduced-step generation for selected PartiFlow models; it can use the configured accelerator. These operations are mutually exclusive. Their reports distinguish missing files, loading failures, and execution failures, and include diagnostic output locations when available.

`--model-dir` selects the model root. Add `--data-dir` or a training `--config` to check PepInter's split and processed files. `check` on a task file checks its basic schema and paths; use `design --show-config` to inspect design settings before generation.

Terminal output uses readable status tables. Redirected output is a JSON report, suitable for scripts. A failed check returns a nonzero exit status; import the missing target or correct the reported path before repeating it.
