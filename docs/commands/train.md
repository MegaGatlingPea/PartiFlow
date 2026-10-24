# Training

```bash
partiflow train vae --config configs/training_ae.yaml --show-config
partiflow train fm --config configs/training.yaml --model-dir /path/to/models --show-config
partiflow train fm --config configs/training.yaml --model-dir /path/to/models --override seed=2026
```

The stage is `vae` or `fm`; `--config` is required. Repeat `--override KEY=VALUE` for Hydra configuration updates. `--model-dir` selects the root for relative pretrained model and autoencoder paths. `--show-config` resolves and prints the selected configuration without starting training. The [resource guide](../resources.md) explains installed configuration paths and dataset path resolution.

Training keeps the dataset transforms, optimizer, VAE/FM losses, and Lightning checkpoint behavior. Outputs go to `<output_dir>/<run_name_>/`, with an `effective_config.yaml` and configured checkpoints and logs. Set dataset paths, hardware devices, distributed strategy, and output directories in the selected config before starting a real run.

`resume: auto` selects the latest versioned `last*.ckpt` in that run's checkpoint directory when available. Use `resume: none` for a fresh run or an explicit checkpoint path for a full resume. `pretrain_ckpt_path` performs a weight-only warm start and cannot be combined with an active resume. Resuming does not promise the same random-number trajectory as uninterrupted training.

Dataset files and pretrained checkpoints are local inputs; [download/import](download.md) does not synthesize missing training data. A successful configuration print is a configuration check and does not establish training or accelerator acceptance.
