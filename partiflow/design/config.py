"""Compose packaged design defaults and invocation-relative user input."""

from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

from partiflow.paths import config_dir, input_path, model_root, resource_root


def _override_key(value):
    return value.lstrip("+~").split("=", 1)[0]


def _compose_node(node, directory, overrides):
    from hydra import compose, initialize_config_dir
    from hydra.core.config_store import ConfigStore
    from omegaconf import OmegaConf

    store = ConfigStore.instance()
    name = "_partiflow_design_" + uuid4().hex
    store.store(name=name, node=OmegaConf.to_container(node, resolve=False))
    try:
        with initialize_config_dir(version_base=None, config_dir=str(directory)):
            return compose(config_name=name, overrides=overrides)
    finally:
        store.repo.pop(name + ".yaml", None)


def _compose(args):
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf, open_dict

    packaged = config_dir().resolve()
    source = OmegaConf.load(packaged / "design.yaml")
    groups = {
        key for entry in source.defaults if OmegaConf.is_dict(entry) for key in entry
    }
    allowed = (set(source) - {"defaults"}) | groups
    with initialize_config_dir(version_base=None, config_dir=str(packaged)):
        cfg = compose(config_name="design")

    directory = packaged
    searchpath = ["file://" + str(packaged)]
    if getattr(args, "config", None):
        path = input_path(args.config)
        import yaml

        value = yaml.safe_load(path.read_text())
        if not isinstance(value, dict):
            raise ValueError("Design configuration must be a mapping")
        source = OmegaConf.create(value)
        directory = path.parent
        configured_searchpath = OmegaConf.select(source, "hydra.searchpath", default=[])
        if configured_searchpath:
            searchpath = list(configured_searchpath) + searchpath
        overlay = _compose_node(
            source,
            directory,
            ["hydra.searchpath=" + json.dumps(searchpath)],
        )
        with open_dict(cfg):
            cfg = OmegaConf.merge(cfg, overlay)

    overrides = list(getattr(args, "override", None) or ())
    # Compose CLI group choices after the file, then apply its value overrides.
    with open_dict(cfg):
        for override in overrides:
            key = _override_key(override)
            if key in groups:
                cfg.pop(key, None)
        cfg.defaults = ["_self_", *({group: None} for group in sorted(groups))]
    cfg = _compose_node(
        cfg,
        directory,
        ["hydra.searchpath=" + json.dumps(searchpath), *overrides],
    )
    unknown = set(cfg) - allowed
    if unknown:
        raise ValueError(f"Unknown design configuration settings: {sorted(unknown)}")
    return cfg


def _validate_search(cfg):
    from omegaconf import OmegaConf

    search = cfg.get("search")
    if not OmegaConf.is_dict(search):
        raise ValueError("search must be a mapping")
    unknown = set(search) - {
        "algorithm",
        "selection_scope",
        "max_batch_size",
        "best_of_n",
    }
    if unknown:
        raise ValueError(f"Unknown search settings: {sorted(unknown)}")
    algorithm = search.get("algorithm")
    if algorithm not in (None, "best_of_n"):
        raise ValueError("search.algorithm must be null or best_of_n")
    if search.get("selection_scope") != "task_length_plan":
        raise ValueError("search.selection_scope must be task_length_plan")
    best_of_n = search.get("best_of_n")
    if (
        not OmegaConf.is_dict(best_of_n)
        or set(best_of_n) != {"replicas"}
        or type(best_of_n.replicas) is not int
        or best_of_n.replicas < 1
    ):
        raise ValueError("best_of_n accepts only positive replicas")
    maximum = search.get("max_batch_size")
    if maximum is not None and (type(maximum) is not int or maximum < 1):
        raise ValueError("search.max_batch_size must be a positive integer or null")
    if algorithm is None and cfg.get("reward") is not None:
        raise ValueError("Raw sampling requires null reward")
    if algorithm == "best_of_n" and cfg.get("reward") is None:
        raise ValueError("Best-of-N requires AF2 reward")
    if cfg.get("reward") is not None:
        from partiflow.workflows.reward import _config

        _config(OmegaConf.to_container(cfg.reward, resolve=True))


def load(args):
    from omegaconf import OmegaConf, open_dict

    cfg = _compose(args)
    overrides = list(getattr(args, "override", None) or ())
    aliases = {
        "ckpt": "checkpoint.model_ckpt",
        "seed": "seed",
        "batch_size": "datamodule.batch_size",
        "num_workers": "datamodule.num_workers",
        "compile": "compile.enabled",
        "run_name": "output.run_name",
    }
    for name, key in aliases.items():
        value = getattr(args, name, None)
        if value is not None:
            if any(_override_key(item) == key for item in overrides):
                raise ValueError(f"CLI and --override both set {key}")
            OmegaConf.update(cfg, key, value, merge=False)
    _validate_search(cfg)

    task_file = input_path(args.task_file)
    tasks = OmegaConf.to_container(OmegaConf.load(task_file), resolve=True)
    if not isinstance(tasks, dict) or not tasks:
        raise ValueError("Task YAML must contain named task mappings")
    selected = list(getattr(args, "task", None) or tasks)
    unknown = set(selected) - tasks.keys()
    if unknown:
        raise ValueError(f"Unknown tasks: {sorted(unknown)}")
    tasks = {name: tasks[name] for name in selected}
    bundled = task_file.is_relative_to(resource_root() / "configs")
    for name, task in tasks.items():
        if not name or any(
            c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
            for c in name
        ):
            raise ValueError(f"Invalid task name: {name}")
        if not isinstance(task, dict):
            raise ValueError(f"Task {name} must be a mapping")
        if task.get("pdb_path"):
            path = Path(task["pdb_path"]).expanduser()
            task["pdb_path"] = str(
                (
                    resource_root() / path
                    if bundled and not path.is_absolute()
                    else input_path(path)
                ).resolve()
            )
        if task.get("sample_num") is None:
            task["sample_num"] = cfg.sample_num
        if (
            isinstance(task.get("sample_num"), bool)
            or not isinstance(task.get("sample_num"), int)
            or task["sample_num"] < 1
        ):
            raise ValueError(f"Task {name} requires positive sample_num")
        for field in (
            "binder_secondary_class",
            "binder_topology",
            "condition_sampler",
            "omit_aa",
        ):
            if cfg.get(field) is not None:
                task[field] = (
                    OmegaConf.to_container(cfg[field], resolve=True)
                    if OmegaConf.is_config(cfg[field])
                    else cfg[field]
                )
    assets = model_root(getattr(args, "model_dir", None))
    with open_dict(cfg):
        cfg.tasks = tasks
        cfg.task_names = selected
        cfg._model_dir = str(assets)
    for field in ("model_ckpt", "autoencoder_ckpt"):
        value = str(cfg.checkpoint[field])
        path = Path(value).expanduser()
        cfg.checkpoint[field] = str(
            assets / path if value.startswith("./ckpts/") else input_path(path)
        )
    if getattr(args, "out", None):
        out = input_path(args.out)
        cfg.output.root_dir = str(out.parent)
        cfg.output.run_name = out.name
    else:
        cfg.output.root_dir = str(input_path(cfg.output.root_dir))
    return cfg
