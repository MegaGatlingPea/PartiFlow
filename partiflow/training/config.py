"""Hydra training composition without a working-directory dependency."""
from __future__ import annotations

from pathlib import Path

from partiflow.paths import checkout_root, config_dir, model_root

ASSET_PATH_KEYS = {"pretrain_ckpt_path", "autoencoder_ckpt_path"}
PATH_KEYS = {"split_csv", "structure_dir", "processed_dir", "pretrain_ckpt_path",
             "autoencoder_ckpt_path", "output_dir", "inductor_cache_dir", "archive", "root"}


def _relocate(value):
    if isinstance(value, dict):
        return {key: _relocate(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_relocate(item) for item in value]
    if isinstance(value, str) and value.startswith("src.partiflow."):
        return "partiflow.model." + value[len("src.partiflow."):]
    if isinstance(value, str) and value.startswith("src."):
        return "partiflow." + value[4:]
    return value


def default_work_root() -> Path:
    return (checkout_root() or Path.cwd()).resolve()


def _absolute_paths(value, base: Path, assets: Path):
    if not isinstance(value, dict):
        if isinstance(value, list):
            return [_absolute_paths(item, base, assets) for item in value]
        return value
    result = {}
    for key, item in value.items():
        if key in PATH_KEYS and isinstance(item, str) and item.strip():
            path = Path(item).expanduser()
            path_base = assets if key in ASSET_PATH_KEYS else base
            item = str(path if path.is_absolute() else (path_base / path).resolve())
        result[key] = _absolute_paths(item, base, assets)
    return result


def load_config(path: Path, overrides=None, *, stage: str = "fm", model_directory=None):
    import hydra
    from omegaconf import OmegaConf
    path = Path(path).expanduser().resolve()
    bundled_configs = config_dir().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Training configuration not found: {path}")
    if path.suffix not in {".yaml", ".yml"}:
        raise ValueError("Training configuration must be a YAML file")
    overrides = list(overrides or [])
    source = OmegaConf.load(path)
    if not OmegaConf.is_dict(source):
        raise ValueError("Training configuration must be a mapping")
    # Historical AE configs refer to a group file that was never shipped.
    defaults = source.get("defaults", [])
    if OmegaConf.is_config(defaults):
        defaults = OmegaConf.to_container(defaults, resolve=False)
    if any(isinstance(item, dict) and item.get("network") == "transformer_ae_large" for item in defaults):
        if not any(str(item).startswith("network=") for item in overrides):
            overrides.append("network=transformer_ae")
    if path.parent != bundled_configs:
        overrides.append(f"hydra.searchpath=[file://{bundled_configs}]")
    with hydra.initialize_config_dir(version_base=None, config_dir=str(path.parent)):
        config = hydra.compose(config_name=path.name, overrides=overrides)
    plain = _relocate(OmegaConf.to_container(config, resolve=True))
    base = default_work_root() if path.is_relative_to(bundled_configs) else path.parent
    config = OmegaConf.create(_absolute_paths(plain, base, model_root(model_directory)))
    if stage not in {"vae", "fm"}:
        raise ValueError(f"Unsupported training stage: {stage}")
    if not config.get("run_name_"):
        config.run_name_ = stage + "_train"
    name = str(config.run_name_)
    if Path(name).name != name or name in {".", ".."}:
        raise ValueError("run_name_ must be a single directory name")
    if not config.get("output_dir"):
        config.output_dir = str(base / "runs")
    if config.get("resume") not in {None, "auto", "none", False}:
        path_resume = Path(str(config.resume)).expanduser()
        config.resume = str(path_resume if path_resume.is_absolute() else (base / path_resume).resolve())
    return config
