import json
import os

from pathlib import Path

import hydra
import lightning as L
from datetime import timedelta
import torch
import wandb
import loralib as lora
from dotenv import load_dotenv
from lightning.pytorch.loggers import CSVLogger, WandbLogger
from lightning.pytorch.utilities import rank_zero_only
from loguru import logger
from omegaconf import OmegaConf

from partiflow.model import PartiFlow
from partiflow.vae.autoencoder import AutoEncoder
from .cli import resolve_resume
from .compilation import compile_forward_by_mode
from .config import default_work_root
from partiflow.utils.ema_callback import EMA, EmaModelCheckpoint
from partiflow.utils.checkpoint_utils import load_state_dict_partial
from partiflow.utils.lora_utils import replace_lora_layers
from partiflow.utils.seed_callback import SeedCallback
from partiflow.utils.training_analysis_utils import (
    GradAndWeightAnalysisCallback,
    LogEpochTimeCallback,
    LogSetpTimeCallback,
    SkipNanGradCallback,
    SkipLargeGradients,
)


@rank_zero_only
def log_info(msg):
    logger.info(msg)


@rank_zero_only
def create_dir(ckpt_path_store, parents=True, exist_ok=True):
    Path(ckpt_path_store).mkdir(parents=parents, exist_ok=exist_ok)


def get_run_dirs(cfg_exp):
    """
    Get root directory for run and directory to store checkpoints.
    """
    run_name = cfg_exp.run_name_
    log_info(f"Job name: {run_name}")
    root_run = str(Path(cfg_exp.output_dir) / run_name)
    log_info(f"Root run: {root_run}")

    ckpt_path_store = os.path.join(
        root_run, "checkpoints"
    )  # Checkpoints in ./store/run_id/checkpoints/<ckpt-file>
    log_info(f"Checkpoints directory: {ckpt_path_store}")
    return run_name, root_run, ckpt_path_store


def initialize_callbacks(cfg_exp):
    """
    Initializes general training callbacks.
    """
    callbacks = [SeedCallback()]

    # Gradient and weight stats thoughout training, possibly skip updates with nan in grad
    if cfg_exp.opt.grad_and_weight_analysis:
        callbacks.append(GradAndWeightAnalysisCallback())
    if cfg_exp.opt.skip_nan_grad:
        callbacks.append(SkipNanGradCallback())

    skip_large = cfg_exp.opt.get("skip_large_grad_updates", {})
    if skip_large.get("use", False):
        callbacks.append(
            SkipLargeGradients(
                moving_avg_size=skip_large.get("moving_avg_size", 100),
                factor_threshold=skip_large.get("factor_threshold", 10),
                min_opt_steps=skip_large.get("min_opt_steps", 2000),
            )
        )
    callbacks.append(LogEpochTimeCallback())
    callbacks.append(LogSetpTimeCallback())

    log_info(f"Using EMA with decay {cfg_exp.ema.decay}")
    callbacks.append(EMA(**cfg_exp.ema))
    return callbacks


def get_training_precision(cfg_exp, is_cluster_run):
    """
    Gets and sets correct training precision.
    """
    precision = "32"
    if not cfg_exp.force_precision_f32 and cfg_exp.hardware.accelerator != "cpu":
        log_info("Using mixed precision")
        torch.set_float32_matmul_precision("medium")
        if is_cluster_run:
            precision = "bf16-mixed"
        else:
            precision = "16"
    else:
        torch.set_float32_matmul_precision("high")
    return precision


def load_data_module(cfg_exp, is_cluster_run):
    """
    Loads data config file and creates corresponding datamodule.
    """
    num_cpus = cfg_exp.hardware.ncpus_per_task_train_
    log_info(
        f"Number of CPUs per task used (will be used for number dataloader number of workers): {num_cpus}"
    )
    cfg_data = cfg_exp.dataset

    configured_num_workers = cfg_data.datamodule.get("num_workers", num_cpus)
    local_devices = max(1, int(cfg_exp.hardware.get("ngpus_per_node_", 1)))
    per_rank_cpu_cap = max(1, num_cpus // local_devices)
    cfg_data.datamodule.num_workers = min(configured_num_workers, per_rank_cpu_cap)
    log_info(
        "Using dataloader num_workers="
        f"{cfg_data.datamodule.num_workers} "
        f"(config={configured_num_workers}, cpu_cap={num_cpus}, "
        f"local_devices={local_devices}, per_rank_cap={per_rank_cpu_cap})"
    )
    log_info(f"Data config {cfg_data}")

    datamodule = hydra.utils.instantiate(cfg_data.datamodule)
    return cfg_data, datamodule


def get_model_n_ckpt_resume(cfg_exp, ckpt_path_store, *, stage="fm", root_run=None):
    resume = resolve_resume(cfg_exp, Path(ckpt_path_store))
    pretrain = cfg_exp.get("pretrain_ckpt_path")
    if resume is not None and pretrain is not None:
        raise ValueError(
            "Choose resume or pretrain_ckpt_path; warm-start only loads model weights"
        )
    # Initialize model weights deterministically on fresh and resumed launches.
    # SeedCallback applies the historical rank/step seed during a resume. This
    # does not promise the same random-number trajectory as an uninterrupted run.
    L.seed_everything(int(cfg_exp.seed), workers=True)
    store_dir = str(Path(root_run or Path(ckpt_path_store).parent) / "validation")
    if stage == "vae":
        model = AutoEncoder(cfg_exp, store_dir=store_dir)
    else:
        model = PartiFlow(cfg_exp, store_dir=store_dir)
    if stage == "fm" and cfg_exp.get("lora") and cfg_exp.lora.get("r"):
        replace_lora_layers(
            model, cfg_exp.lora.r, cfg_exp.lora.lora_alpha, cfg_exp.lora.lora_dropout
        )
        lora.mark_only_lora_as_trainable(model, bias=cfg_exp.lora.train_bias)
    if pretrain is not None:
        checkpoint = torch.load(pretrain, map_location="cpu", weights_only=False)
        load_state_dict_partial(
            model,
            checkpoint,
            strict=False,
            log_prefix=f"{stage} warm-start from {pretrain}",
        )
    log_info(f"Resume checkpoint: {resume}; warm-start weights: {pretrain}")
    return model, resume


def apply_trainable_parameter_filter(cfg_exp, model):
    training_cfg = cfg_exp.get("training", None)
    if training_cfg is None:
        return
    patterns = training_cfg.get("trainable_parameter_patterns", None)
    if patterns is None:
        return
    if isinstance(patterns, str):
        patterns = [patterns]
    patterns = [str(pattern) for pattern in patterns if str(pattern).strip()]
    if len(patterns) == 0:
        return

    matched_names = []
    for name, param in model.named_parameters():
        trainable = any(pattern in name for pattern in patterns)
        param.requires_grad = bool(trainable)
        if trainable:
            matched_names.append(name)

    if not matched_names:
        raise ValueError(
            "training.trainable_parameter_patterns did not match any parameters: "
            f"{patterns}"
        )

    total_params = sum(param.numel() for param in model.parameters())
    trainable_params = sum(
        param.numel() for param in model.parameters() if param.requires_grad
    )
    log_info(
        "Trainable parameter filter enabled: "
        f"{trainable_params}/{total_params} parameters trainable; "
        f"patterns={patterns}; matched={matched_names[:20]}"
    )


def maybe_compile_model(cfg_exp, model):
    compile_cfg = cfg_exp.get("compile", None)
    if compile_cfg is None or not bool(compile_cfg.get("enabled", False)):
        return model

    compile_kwargs = dict(
        mode=str(compile_cfg.get("mode", "default")),
        dynamic=bool(compile_cfg.get("dynamic", True)),
        fullgraph=bool(compile_cfg.get("fullgraph", False)),
    )
    log_info(
        "Compiling denoiser trunk forward with separate training and evaluation "
        f"entry points using torch.compile({compile_kwargs})."
    )
    compile_forward_by_mode(model.nn, **compile_kwargs)
    return model


def setup_inductor_cache(cfg_exp) -> None:
    compile_cfg = cfg_exp.get("compile", None)
    if compile_cfg is None or not bool(compile_cfg.get("enabled", False)):
        return

    env_cache_dir = os.environ.get("TORCHINDUCTOR_CACHE_DIR", "").strip()
    if env_cache_dir:
        Path(env_cache_dir).mkdir(parents=True, exist_ok=True)
        log_info(f"Using TORCHINDUCTOR_CACHE_DIR from environment: {env_cache_dir}")
        return

    cache_dir = compile_cfg.get("inductor_cache_dir", ".train_inductor_cache")
    if cache_dir is None or str(cache_dir).strip() == "":
        return

    cache_path = Path(str(cache_dir))
    if not cache_path.is_absolute():
        cache_path = default_work_root() / cache_path
    cache_path.mkdir(parents=True, exist_ok=True)
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(cache_path)
    log_info(f"Set TORCHINDUCTOR_CACHE_DIR={cache_path}")


def setup_ckpt(cfg_exp, ckpt_path_store):
    """
    Created checkpointing callbacks and creates directory to store checkpoints.
    """

    def interval_kwargs(
        step_key: str, epoch_key: str, *, default_epoch: int | None = None
    ):
        step_value = cfg_exp.log.get(step_key, None)
        epoch_value = cfg_exp.log.get(epoch_key, None)
        if step_value is not None and epoch_value is not None:
            raise ValueError(
                f"Only one of log.{step_key} and log.{epoch_key} can be set."
            )
        if step_value is not None:
            return {"every_n_train_steps": int(step_value)}
        if epoch_value is not None:
            return {"every_n_epochs": int(epoch_value)}
        if default_epoch is not None:
            return {"every_n_epochs": int(default_epoch)}
        return {}

    args_ckpt = {
        "dirpath": ckpt_path_store,
        "save_last": False,
        "save_weights_only": False,
        "filename": "chk_{epoch:04d}_{step:07d}",
        "monitor": "train_loss",
        "save_top_k": 10000,
        "mode": "min",
        **interval_kwargs(
            "checkpoint_every_n_steps", "checkpoint_every_n_epochs", default_epoch=1
        ),
    }
    checkpoint_callbacks = [EmaModelCheckpoint(**args_ckpt)]

    if bool(cfg_exp.log.get("save_last", True)):
        args_ckpt_last = {
            "dirpath": ckpt_path_store,
            "save_weights_only": False,
            "filename": "ignore",
            "save_last": True,
            **interval_kwargs(
                "last_ckpt_every_n_steps", "last_ckpt_every_n_epochs", default_epoch=1
            ),
        }
        checkpoint_callbacks.append(EmaModelCheckpoint(**args_ckpt_last))

    best_valloss_top_k = int(cfg_exp.log.get("best_valloss_top_k", 10))
    if best_valloss_top_k > 0:
        args_ckpt_best_valloss = {
            "dirpath": ckpt_path_store,
            "filename": "chk_{epoch:04d}_valloss_{validation_loss/loss_epoch:.2f}",
            "monitor": "validation_loss/loss_epoch",
            "mode": "min",
            "save_top_k": best_valloss_top_k,
            "save_weights_only": False,
            "auto_insert_metric_name": False,
        }
        checkpoint_callbacks.append(EmaModelCheckpoint(**args_ckpt_best_valloss))

    create_dir(ckpt_path_store, parents=True, exist_ok=True)
    return checkpoint_callbacks


@rank_zero_only
def store_n_log_configs(cfg_exp, cfg_data, run_name, ckpt_path_store, wandb_logger):
    """
    Stores config files locally and logs them to wandb run.
    """

    def store_n_log_config(cfg, cfg_path, wandb_logger):
        with open(cfg_path, "w") as f:
            cfg_aux = OmegaConf.to_container(cfg, resolve=True)
            json.dump(cfg_aux, f, indent=4, sort_keys=True)

        if wandb_logger is not None:
            artifact = wandb.Artifact(f"config_files_{run_name}", type="config")
            artifact.add_file(cfg_path)
            wandb_logger.experiment.log_artifact(artifact)

    cfg_exp_file = os.path.join(ckpt_path_store, f"exp_config_{run_name}.json")
    cfg_data_file = os.path.join(ckpt_path_store, f"data_config_{run_name}.json")

    store_n_log_config(cfg_exp, cfg_exp_file, wandb_logger)
    store_n_log_config(cfg_data, cfg_data_file, wandb_logger)


def train(cfg_exp, *, stage="fm") -> None:
    load_dotenv()
    is_cluster_run = bool(cfg_exp.get("is_cluster_run", False))
    nolog = bool(cfg_exp.get("nolog", False))
    if cfg_exp.get("single", False):
        cfg_exp.hardware.ngpus_per_node_ = 1
        cfg_exp.hardware.nnodes_ = 1
    run_name, root_run, ckpt_path_store = get_run_dirs(cfg_exp)
    Path(root_run).mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg_exp, str(Path(root_run) / "effective_config.yaml"), resolve=True)
    preparation = cfg_exp.get("data_preparation", {})
    if preparation.get("enabled", False):
        from .prepare_data import prepare

        prepare(
            Path(preparation.root),
            Path(preparation.archive) if preparation.get("archive") else None,
            subset=bool(preparation.get("subset", True)),
            extract=bool(preparation.get("extract", False)),
        )
    callbacks = initialize_callbacks(cfg_exp)
    cfg_data, datamodule = load_data_module(cfg_exp, is_cluster_run)
    setup_inductor_cache(cfg_exp)
    model, resume_ckpt_path = get_model_n_ckpt_resume(
        cfg_exp, ckpt_path_store, stage=stage, root_run=root_run
    )
    apply_trainable_parameter_filter(cfg_exp, model)
    if stage == "fm":
        model = maybe_compile_model(cfg_exp, model)

    # A resolved configuration is emitted before Trainer construction and remains
    # available even when the model or distributed runtime later fails.
    OmegaConf.save(cfg_exp, str(Path(root_run) / "effective_config.yaml"), resolve=True)
    log_info(f"Effective configuration: {Path(root_run) / 'effective_config.yaml'}")
    loggers = []
    if not nolog:
        loggers.append(CSVLogger(save_dir=root_run, name="metrics", version=""))
    wandb_logger = None
    if cfg_exp.log.log_wandb and not nolog:
        wandb_logger = WandbLogger(
            project=cfg_exp.log.wandb_project,
            id=run_name,
            offline=cfg_exp.log.get("offline", True),
            entity=cfg_exp.log.get("wandb_entity"),
        )
        loggers.insert(0, wandb_logger)
    if cfg_exp.log.checkpoint and not nolog:
        callbacks.extend(setup_ckpt(cfg_exp, ckpt_path_store))
        store_n_log_configs(cfg_exp, cfg_data, run_name, ckpt_path_store, wandb_logger)
    strategy = cfg_exp.opt.dist_strategy
    if isinstance(strategy, str) and strategy.lower() == "ddp":
        if int(cfg_exp.hardware.ngpus_per_node_) * int(cfg_exp.hardware.nnodes_) == 1:
            strategy = "auto"
        else:
            kwargs = {}
            if cfg_exp.opt.get("ddp_timeout_hours"):
                kwargs["timeout"] = timedelta(hours=cfg_exp.opt.ddp_timeout_hours)
            if cfg_exp.get("lora") and cfg_exp.lora.get("r"):
                kwargs["find_unused_parameters"] = True
            strategy = L.pytorch.strategies.DDPStrategy(**kwargs)
    val_interval = cfg_exp.opt.val_check_interval
    trainer = L.Trainer(
        max_epochs=cfg_exp.opt.max_epochs,
        max_steps=int(cfg_exp.opt.get("max_steps", -1)),
        accelerator=cfg_exp.hardware.accelerator,
        devices=cfg_exp.hardware.ngpus_per_node_,
        num_nodes=cfg_exp.hardware.nnodes_,
        callbacks=callbacks,
        logger=loggers or False,
        enable_checkpointing=bool(cfg_exp.log.checkpoint and not nolog),
        log_every_n_steps=cfg_exp.log.log_every_n_steps,
        default_root_dir=root_run,
        check_val_every_n_epoch=1 if isinstance(val_interval, float) else None,
        val_check_interval=val_interval,
        strategy=strategy,
        enable_progress_bar=bool(cfg_exp.get("enable_progress_bar", True)),
        limit_train_batches=cfg_exp.get("limit_train_batches", 1.0),
        limit_val_batches=cfg_exp.get("limit_val_batches", 100),
        accumulate_grad_batches=cfg_exp.opt.accumulate_grad_batches,
        num_sanity_val_steps=cfg_exp.get("num_sanity_val_steps", 0),
        precision=get_training_precision(cfg_exp, is_cluster_run),
        gradient_clip_algorithm="norm",
        gradient_clip_val=1.0,
    )
    from .loops import install_epoch_boundary_loop

    install_epoch_boundary_loop(trainer)
    if cfg_exp.get("eval_first_only", False):
        trainer.validate(model, datamodule, ckpt_path=resume_ckpt_path)
        return
    trainer.fit(model, datamodule, ckpt_path=resume_ckpt_path)
