# Train the VAE or flow-matching model

Choose `vae` or `fm` and provide a training YAML. Inspect the resolved paths and settings before starting:

```bash
partiflow train vae --config configs/training_ae.yaml --show-config
partiflow train fm --config configs/training.yaml --show-config
```

The [VAE configuration](../../configs/training_ae.yaml) uses the PepInter VAE dataset settings. The [flow-matching configuration](../../configs/training.yaml) also needs an autoencoder checkpoint. Prepare the [data and model resources](../resources.md) first, then set hardware, split paths, batch sizes, and output settings for your run.

```bash
partiflow train vae --config configs/training_ae.yaml --override run_name_=my_vae
partiflow train fm --config configs/training.yaml --model-dir models --override run_name_=my_fm
```

Repeat `--override KEY=VALUE` for Hydra settings. `--model-dir` resolves relative pretrained and autoencoder checkpoint paths. Other path rules depend on whether the selected config is bundled or user-written; see [resources](../resources.md#prepare-training-data).

Outputs go to `<output_dir>/<run_name_>/`, including `effective_config.yaml`, configured checkpoints, and logs. The bundled defaults use `runs/ae_train` and `runs/fm_train`. A configuration print exits before training; the actual run needs the configured runtime and accelerator.

## Start fresh, warm start, or resume

The default `resume: auto` selects the latest versioned `last*.ckpt` in that run's checkpoint directory when one is available. Repeating a run name can therefore continue an existing run.

- Use `resume: none` for a fresh run.
- Set `resume` to a checkpoint path for a full training resume, including optimizer and scheduler state.
- Set `pretrain_ckpt_path` for a weight-only warm start, with resume disabled.

These are YAML settings, also available through `--override`. For example:

```bash
partiflow train fm --config configs/training.yaml \
  --override run_name_=fm_warm_start --override resume=none \
  --override pretrain_ckpt_path=./ckpts/partiflow.ckpt
```

A full resume continues training state but does not promise an identical random-number trajectory. Use a new run name when you want to keep a previous run's outputs separately.

## Compile the FM forward pass

FM compilation is off by default. To enable it, add `--override compile.enabled=true` to the training command. Training and validation compile separately; the first pass through a new shape can take longer. Checkpoints retain the same parameter names and can be resumed with compilation enabled or disabled.

The supplied configurations use one GPU. Multi-GPU training has not been validated for this release.
