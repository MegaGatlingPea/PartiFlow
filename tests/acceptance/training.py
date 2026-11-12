"""Public CLI training gates and explicitly instrumented continuation checks.

Run this file with ``prepare --output-root DIR --resource-root DIR`` first.
``execute --case NAME`` launches ``python -m partiflow train`` as a subprocess.
The public cases use the production datamodule. Instrumented cases use a Hydra
datamodule subclass which only adds a Lightning callback; they are reported
separately. Runtime files, configurations, logs and checkpoints stay in DIR.

This checks software execution and state restoration. It does not test model
convergence or promise matching random trajectories after a restart.
"""

from __future__ import annotations

import argparse
from collections import Counter
import copy
import csv
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any


class AcceptanceError(AssertionError):
    """A required training gate did not hold."""


def require(condition, message):
    if not condition:
        raise AcceptanceError(message)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def load_checkpoint(path):
    import torch

    return torch.load(path, map_location="cpu", weights_only=False)


def cpu_tree(value):
    import torch

    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu_tree(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(cpu_tree(item) for item in value)
    return copy.deepcopy(value)


def compare_tree(actual, expected, path="state", *, close=False):
    """Compare every leaf, including optimizer groups and every moment tensor."""
    import torch

    counts = Counter()
    if isinstance(expected, torch.Tensor):
        require(isinstance(actual, torch.Tensor), f"{path}: expected a tensor")
        require(actual.dtype == expected.dtype, f"{path}: tensor dtype differs")
        require(actual.shape == expected.shape, f"{path}: tensor shape differs")
        actual = actual.detach().cpu()
        expected = expected.detach().cpu()
        if close and (expected.is_floating_point() or expected.is_complex()):
            torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-7, msg=path)
        else:
            require(torch.equal(actual, expected), f"{path}: tensor values differ")
        counts.update(tensors=1, elements=expected.numel())
    elif isinstance(expected, dict):
        require(isinstance(actual, dict), f"{path}: expected a mapping")
        require(actual.keys() == expected.keys(), f"{path}: mapping keys differ")
        for key, value in expected.items():
            counts.update(compare_tree(actual[key], value, f"{path}.{key}", close=close))
    elif isinstance(expected, (tuple, list)):
        require(isinstance(actual, (tuple, list)), f"{path}: expected a sequence")
        require(len(actual) == len(expected), f"{path}: sequence lengths differ")
        for index, value in enumerate(expected):
            counts.update(compare_tree(actual[index], value, f"{path}[{index}]", close=close))
    else:
        require(actual == expected, f"{path}: {actual!r} != {expected!r}")
        counts.update(scalars=1)
    return dict(counts)


def finite_tree(value, path="state"):
    import torch

    counts = Counter()
    if isinstance(value, torch.Tensor):
        require(bool(torch.isfinite(value).all()), f"{path}: non-finite tensor")
        counts.update(tensors=1, elements=value.numel())
    elif isinstance(value, dict):
        for key, item in value.items():
            counts.update(finite_tree(item, f"{path}.{key}"))
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            counts.update(finite_tree(item, f"{path}[{index}]"))
    elif isinstance(value, float):
        require(math.isfinite(value), f"{path}: non-finite scalar")
    return dict(counts)


def moment_inventory(raw, *, expected_step=None):
    """Inspect all Adam states, rather than a representative parameter."""
    result = Counter()
    require(bool(raw.get("param_groups")), "optimizer has no parameter groups")
    identifiers = [key for group in raw["param_groups"] for key in group["params"]]
    require(len(identifiers) == len(set(identifiers)), "optimizer repeats parameters")
    require(set(raw["state"]) == set(identifiers), "optimizer state is incomplete")
    for identifier in identifiers:
        state = raw["state"][identifier]
        require({"step", "exp_avg", "exp_avg_sq"} <= state.keys(), "missing Adam moments")
        finite_tree(state, f"optimizer.state[{identifier}]")
        if expected_step is not None:
            require(float(state["step"]) == expected_step, f"Adam step differs for {identifier}")
        require(state["exp_avg"].shape == state["exp_avg_sq"].shape, "Adam moment shapes differ")
        for name, value in state.items():
            if name != "step":
                result[name] += value.numel()
        result["parameter_states"] += 1
    result["parameter_groups"] = len(raw["param_groups"])
    return dict(result)


def expected_epoch(checkpoint):
    loop = checkpoint.get("loops", {}).get("fit_loop", {})
    progress = loop.get("epoch_progress", {}).get("current", {})
    completed = (
        loop.get("epoch_loop.batch_progress", {}).get("is_last_batch") is True
        and progress.get("ready") == progress.get("processed")
        and progress.get("ready") == progress.get("completed", -2) + 1
    )
    return int(checkpoint["epoch"]) + int(completed)


def runtime_state(trainer, model):
    for optimizer in trainer.optimizers:
        if hasattr(optimizer, "join"):
            optimizer.join()
    return cpu_tree({
        "model": model.state_dict(),
        "optimizers": [optimizer.state_dict() for optimizer in trainer.optimizers],
        "schedulers": [item.scheduler.state_dict() for item in trainer.lr_scheduler_configs],
        "global_step": int(trainer.global_step),
        "epoch": int(trainer.current_epoch),
        "samples": getattr(model, "nsamples_processed", None),
    })


def restored_state_check(actual, checkpoint_path, *, before_update=False):
    path = Path(checkpoint_path)
    checkpoint = load_checkpoint(path)
    paired = checkpoint if path.name.endswith("-EMA.ckpt") else load_checkpoint(
        path.with_name(path.stem + "-EMA.ckpt")
    )
    for key in ("epoch", "global_step", "nsamples_processed"):
        require(checkpoint.get(key) == paired.get(key), f"paired {key} differs")
    counts = {}
    for name, wanted in (
        ("model", checkpoint["state_dict"]),
        ("optimizers", paired["optimizer_states"]),
        ("schedulers", checkpoint.get("lr_schedulers", [])),
    ):
        counts[name] = compare_tree(actual[name], wanted, name)
    require(actual["global_step"] == checkpoint["global_step"], "global step was not restored")
    require(actual["epoch"] == expected_epoch(checkpoint), "resumed epoch is incorrect")
    if not before_update:
        require(actual["samples"] == checkpoint.get("nsamples_processed"), "sample counter was not restored")
    for state in actual["optimizers"]:
        require({"opt", "ema", "current_step", "decay", "every_n_steps"} <= state.keys(), "EMA state is incomplete")
        require(state["current_step"] == actual["global_step"], "EMA counter was not restored")
        moment_inventory(state["opt"], expected_step=actual["global_step"])
    return {"compared": counts, "global_step": actual["global_step"], "epoch": actual["epoch"]}


def reference_adam_update(trainer, model):
    """Recompute one update on CPU from actual, unclipped runtime gradients.

    Lightning invokes on_before_optimizer_step before its configured norm clip.
    This reference applies that clip and the ordinary Adam/AdamW equations. It
    does not invoke the model, backward, or the live optimizer a second time.
    """
    import torch

    require(len(trainer.optimizers) == 1, "reference expects one optimizer")
    wrapped = trainer.optimizers[0]
    optimizer = wrapped.optimizer
    require(type(optimizer) in (torch.optim.Adam, torch.optim.AdamW), "unsupported optimizer reference")
    require(str(trainer.gradient_clip_algorithm).lower().split(".")[-1] == "norm", "reference expects norm clipping")
    require(float(trainer.accumulate_grad_batches) == 1, "reference expects one update per batch")
    names = {id(value): name for name, value in model.named_parameters()}
    active = [p for group in optimizer.param_groups for p in group["params"] if p.grad is not None]
    require(bool(active), "no parameter received a gradient")
    for parameter in active:
        require(not parameter.grad.is_sparse, "sparse gradients are unsupported")
        require(bool(torch.isfinite(parameter.grad).all()), "non-finite actual gradient")
    norm = torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(p.grad.detach(), 2) for p in active]), 2)
    norm_value = float(norm)
    require(math.isfinite(norm_value) and norm_value > 0, "actual gradient norm is invalid")
    clip = min(1.0, float(trainer.gradient_clip_val) / (norm_value + 1e-6))
    values = {}
    for group in optimizer.param_groups:
        for option in ("capturable", "differentiable", "fused"):
            require(not group.get(option), f"reference does not support {option}")
        beta1, beta2 = group["betas"]
        lr = float(group["lr"])
        for parameter in group["params"]:
            if parameter.grad is None:
                continue
            before = parameter.detach().cpu().clone()
            gradient = parameter.grad.detach().cpu().clone().mul_(clip)
            if group.get("maximize", False):
                gradient.neg_()
            state = optimizer.state[parameter]
            step = float(state.get("step", 0)) + 1
            first = cpu_tree(state.get("exp_avg", torch.zeros_like(before)))
            second = cpu_tree(state.get("exp_avg_sq", torch.zeros_like(before)))
            decay = float(group["weight_decay"])
            if isinstance(optimizer, torch.optim.AdamW):
                updated = before * (1 - lr * decay)
            else:
                updated = before.clone()
                gradient.add_(before, alpha=decay)
            first.lerp_(gradient, 1 - beta1)
            second.mul_(beta2).addcmul_(gradient, gradient, value=1 - beta2)
            moments = {"exp_avg": first, "exp_avg_sq": second}
            denominator_value = second
            if group.get("amsgrad", False):
                maximum = cpu_tree(state.get("max_exp_avg_sq", torch.zeros_like(before)))
                torch.maximum(maximum, second, out=maximum)
                moments["max_exp_avg_sq"] = maximum
                denominator_value = maximum
            denominator = denominator_value.sqrt().div_(math.sqrt(1 - beta2 ** step)).add_(group["eps"])
            updated.addcdiv_(first, denominator, value=-lr / (1 - beta1 ** step))
            values[names[id(parameter)]] = {"parameter": updated, "moments": moments, "step": step}
    return {"values": values, "gradient_norm": norm_value, "clip_coefficient": clip}


def check_update(trainer, model, before, reference):
    import torch

    after = runtime_state(trainer, model)
    require(after["global_step"] == before["global_step"] + 1, "next optimizer step did not advance")
    finite = finite_tree(after)
    changed = Counter()
    names = {id(value): name for name, value in model.named_parameters()}
    for optimizer, old, new in zip(trainer.optimizers, before["optimizers"], after["optimizers"]):
        require(new["current_step"] == old["current_step"] + 1, "EMA count did not advance")
        require(new["decay"] == old["decay"] and new["every_n_steps"] == old["every_n_steps"], "EMA settings changed")
        for live_group, saved_group in zip(optimizer.param_groups, new["opt"]["param_groups"]):
            for parameter, identifier in zip(live_group["params"], saved_group["params"]):
                name = names[id(parameter)]
                if name not in reference["values"]:
                    compare_tree(after["model"][name], before["model"][name], name)
                    if identifier in old["opt"]["state"]:
                        compare_tree(new["opt"]["state"][identifier], old["opt"]["state"][identifier], f"inactive.{name}")
                    continue
                wanted = reference["values"][name]
                compare_tree(after["model"][name], wanted["parameter"], f"reference.{name}", close=True)
                state = new["opt"]["state"][identifier]
                require(float(state["step"]) == wanted["step"], f"Adam count did not advance: {name}")
                changed["parameters_checked"] += 1
                changed["parameters_changed"] += int(not torch.equal(after["model"][name], before["model"][name]))
                for key, value in wanted["moments"].items():
                    compare_tree(state[key], value, f"reference.{name}.{key}", close=True)
                    previous = old["opt"]["state"].get(identifier, {}).get(key)
                    changed[f"{key}_checked"] += 1
                    changed[f"{key}_changed"] += int(previous is None or not torch.equal(state[key], previous))
        old_ema = old["ema"]
        parameters = [p for group in optimizer.param_groups for p in group["params"]]
        if not old_ema:
            old_ema = tuple(before["model"][names[id(p)]] for p in parameters)
        expected_ema = tuple(value.clone() for value in old_ema)
        if old["current_step"] % old["every_n_steps"] == 0:
            for target, parameter in zip(expected_ema, parameters):
                target.mul_(old["decay"]).add_(parameter.detach().cpu(), alpha=1 - old["decay"])
        compare_tree(new["ema"], expected_ema, "reference.ema", close=True)
        moment_inventory(new["opt"], expected_step=after["global_step"])
    for key in ("parameters_changed", "exp_avg_changed", "exp_avg_sq_changed"):
        require(changed[key] > 0, f"no actual {key}")
    return {"finite": finite, "changes": dict(changed), "reference_matches": True,
            "gradient_norm": reference["gradient_norm"], "clip_coefficient": reference["clip_coefficient"],
            "global_step": after["global_step"], "ema_values_checked": True}


class TrainingStateTrace:
    """Mixin for the explicitly instrumented callback; no training overrides."""

    def __init__(self, directory, stage):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        require(not (self.directory / "state_checks.json").exists(), "state trace already exists")
        self.stage = stage
        self.before = None
        self.reference = None
        self.frozen = None
        self.result = {"kind": "instrumented", "completed": False, "stage": stage,
                       "training_losses": [], "validation_losses": [], "updates": 0}

    def _save(self):
        write_json(self.directory / "state_checks.json", self.result)

    def on_train_start(self, trainer, model):
        state = runtime_state(trainer, model)
        finite_tree(state)
        require(all("_orig_mod" not in key for key in state["model"]), "compiled checkpoint key leaked")
        trainable = [(name, value) for name, value in model.named_parameters() if value.requires_grad]
        require(bool(trainable), "model has no trainable parameters")
        self.result["trainable_tensors"] = len(trainable)
        self.result["trainable_parameters"] = sum(value.numel() for _, value in trainable)
        self.result["start"] = {key: state[key] for key in ("global_step", "epoch", "samples")}
        if trainer.ckpt_path:
            self.result["restored_on_train_start"] = restored_state_check(state, trainer.ckpt_path)
        if self.stage == "fm":
            frozen_parameters = list(model.autoencoder.parameters())
            require(bool(frozen_parameters) and all(not p.requires_grad for p in frozen_parameters), "FM VAE parameters are not frozen")
            self.frozen = cpu_tree(model.autoencoder.state_dict())
            self.result["frozen_vae_tensors"] = len(self.frozen)
        self._save()

    def on_before_optimizer_step(self, trainer, model, optimizer):
        if self.before is not None or "first_update" in self.result:
            return
        self.before = runtime_state(trainer, model)
        if trainer.ckpt_path:
            self.result["restored_before_optimizer_step"] = restored_state_check(
                self.before, trainer.ckpt_path, before_update=True
            )
        self.reference = reference_adam_update(trainer, model)
        self._save()

    def on_train_batch_end(self, trainer, model, outputs, batch, batch_idx):
        value = outputs["loss"] if isinstance(outputs, dict) else outputs
        loss = float(value.detach()) if hasattr(value, "detach") else float(value)
        require(math.isfinite(loss), "non-finite training loss")
        self.result["training_losses"].append(loss)
        self.result["updates"] += 1
        if self.before is not None:
            self.result["first_update"] = check_update(trainer, model, self.before, self.reference)
            self.before = None
            self.reference = None
        self._save()

    def on_validation_batch_end(self, trainer, model, outputs, batch, batch_idx, dataloader_idx=0):
        if trainer.sanity_checking:
            return
        # Both production models retain each real validation loss until epoch end.
        collected = getattr(model, "validation_output_data" if self.stage == "fm" else "validation_output", None)
        if collected:
            loss = float(collected[-1])
        else:
            value = outputs["loss"] if isinstance(outputs, dict) else outputs
            require(value is not None, "validation loss is unavailable")
            loss = float(value)
        require(math.isfinite(loss), "non-finite validation loss")
        self.result["validation_losses"].append(loss)

    def on_train_end(self, trainer, model):
        require("first_update" in self.result, "no actual optimizer update was checked")
        final = runtime_state(trainer, model)
        finite_tree(final)
        self.result["end"] = {key: final[key] for key in ("global_step", "epoch", "samples")}
        if self.frozen is not None:
            self.result["frozen_vae_comparison"] = compare_tree(model.autoencoder.state_dict(), self.frozen, "frozen_vae")
        compiled = getattr(getattr(model, "nn", None), "forward", None)
        entries = getattr(compiled, "compiled_forwards", ())
        self.result["compiled_entry_points"] = len(entries)
        if entries:
            from torch._inductor import metrics

            self.result["inductor_generated_kernels"] = int(metrics.generated_kernel_count)
        self.result["completed"] = True
        self._save()


def state_trace_callback(directory, stage):
    import lightning as L

    class StateTrace(TrainingStateTrace, L.Callback):
        pass

    return StateTrace(directory, stage)


def instrumented_datamodule(trace_directory, acceptance_stage, **kwargs):
    """Hydra-only instrumentation: retain the real dataset and loader methods."""
    from partiflow.datasets.pdb_data import PepLightningDataModule

    class InstrumentedPepDataModule(PepLightningDataModule):
        def setup(self, stage=None):
            super().setup(stage)
            if stage in (None, "fit") and not getattr(self, "_trace_added", False):
                self.trainer.callbacks.append(state_trace_callback(trace_directory, acceptance_stage))
                self._trace_added = True

    return InstrumentedPepDataModule(**kwargs)


def check_dataset(split_csv):
    path = Path(split_csv).resolve()
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    require(Counter(row["split"] for row in rows) == {"train": 64, "val": 16}, "acceptance split must be 64 train / 16 val")
    require(len({row["pdbid"] for row in rows}) == 80, "sample identifiers must be unique")
    selected = []
    for row in rows:
        item = {key: row[key] for key in ("pdbid", "split")}
        for key in ("structure_path", "processed_path"):
            target = Path(row[key])
            target = target if target.is_absolute() else path.parent / target
            target = target.resolve()
            require(target.is_file() and target.stat().st_size > 0, f"missing real input: {target}")
            item[key] = str(target)
        selected.append(item)
    return {"split_csv": str(path), "train_samples": 64, "validation_samples": 16, "samples": selected}


def case_matrix():
    cases = []
    for stage, mode in (("vae", "eager"), ("fm", "eager"), ("fm", "compiled")):
        name = f"public_{stage}_{mode}_epoch"
        cases.append({"name": name, "stage": stage, "mode": mode, "kind": "public", "source": None})
    for kind in ("public", "instrumented"):
        for stage, source_mode, target_mode in (("vae", "eager", "eager"), ("fm", "eager", "compiled"), ("fm", "compiled", "eager")):
            for weights in ("normal", "ema"):
                cases.append({"name": f"{kind}_{stage}_{source_mode}_{weights}_to_{target_mode}",
                              "stage": stage, "mode": target_mode, "kind": kind,
                              "source": f"public_{stage}_{source_mode}_epoch", "weights": weights})
    return cases


def external_root(path, checkout):
    root = Path(path).expanduser().resolve()
    require(root != Path(root.anchor), "output root must name a dedicated directory")
    require(not root.is_relative_to(Path(checkout).resolve()), "runtime output root must be outside the checkout")
    return root


def prepare(args):
    checkout = Path(args.checkout).expanduser().resolve()
    sys.path.insert(0, str(checkout))
    from omegaconf import OmegaConf
    from partiflow.training.config import load_config

    resources = Path(args.resource_root).expanduser().resolve()
    root = external_root(args.output_root, checkout)
    require(not (root / "training_manifest.json").exists(), "training manifest already exists")
    dataset = check_dataset(args.split_csv or resources / "train_data/pepinter/indexes/acceptance.csv")
    base = {}
    for stage, filename in (("vae", "training_vae_acceptance.yaml"), ("fm", "training_fm_acceptance.yaml")):
        base[stage] = OmegaConf.to_container(load_config(checkout / "configs" / filename, stage=stage, model_directory=resources), resolve=True)
    cases = case_matrix()
    for case in cases:
        config = copy.deepcopy(base[case["stage"]])
        config.update(output_dir=str(root / "runs"), run_name_=case["name"], resume="none",
                      enable_progress_bar=False, force_precision_f32=True, nolog=False,
                      limit_train_batches=1.0, limit_val_batches=1.0, num_sanity_val_steps=0)
        config["hardware"].update(accelerator=args.accelerator, ngpus_per_node_=1, nnodes_=1, ncpus_per_task_train_=1)
        config["opt"].update(max_epochs=1, max_steps=-1, accumulate_grad_batches=1,
                             val_check_interval=1.0, grad_and_weight_analysis=False, skip_nan_grad=False)
        config["log"].update(checkpoint=True, log_wandb=False, log_every_n_steps=1,
                             checkpoint_every_n_steps=None, checkpoint_every_n_epochs=1,
                             last_ckpt_every_n_steps=None, last_ckpt_every_n_epochs=1,
                             save_last=True, best_valloss_top_k=0)
        config["data_preparation"]["enabled"] = False
        config["data_preparation"]["root"] = str(resources / "train_data/pepinter")
        module = config["dataset"]["datamodule"]
        module.update(split_csv=dataset["split_csv"], processed_dir=str(resources / "train_data/pepinter/processed"),
                      batch_size=1, num_workers=0, pin_memory=False)
        require(module["_target_"] == "partiflow.datasets.pdb_data.PepLightningDataModule", "unexpected production datamodule")
        if case["stage"] == "fm":
            config["autoencoder_ckpt_path"] = str(resources / "ckpts/autoencoder.ckpt")
            require(Path(config["autoencoder_ckpt_path"]).is_file(), "FM autoencoder checkpoint is missing")
            config["opt"].update(use_scheduler=True, warmup_steps=2, total_steps=130, min_lr=1e-6)
            config["compile"].update(enabled=case["mode"] == "compiled", mode="default", dynamic=True,
                                      fullgraph=True, inductor_cache_dir=str(root / "inductor_cache"))
        if case["source"]:
            filename = "last-EMA.ckpt" if case["weights"] == "ema" else "last.ckpt"
            config["resume"] = str(root / "runs" / case["source"] / "checkpoints" / filename)
            config["opt"].update(max_epochs=2, max_steps=65)
        if case["kind"] == "instrumented":
            module.update(_target_="acceptance.training.instrumented_datamodule",
                          trace_directory=str(root / "traces" / case["name"]), acceptance_stage=case["stage"])
        case["config"] = str(root / "configs" / (case["name"] + ".yaml"))
        case["run_directory"] = str(root / "runs" / case["name"])
        Path(case["config"]).parent.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(OmegaConf.create(config), case["config"], resolve=True)
    manifest = {"version": 1, "output_root": str(root), "checkout": str(checkout),
                "resource_root": str(resources), "dataset": dataset, "cases": cases}
    write_json(root / "training_manifest.json", manifest)
    return {"manifest": str(root / "training_manifest.json"), "cases": [case["name"] for case in cases]}


def read_manifest(root):
    root = Path(root).expanduser().resolve()
    manifest = json.loads((root / "training_manifest.json").read_text())
    require(manifest["version"] == 1 and manifest["output_root"] == str(root), "manifest root/version differs")
    external_root(root, manifest["checkout"])
    return manifest


def find_case(manifest, name):
    matches = [case for case in manifest["cases"] if case["name"] == name]
    require(len(matches) == 1, f"unknown case: {name}")
    return matches[0]


def cli_command(case, python):
    return [str(python), "-m", "partiflow", "train", case["stage"], "--config", case["config"]]


def audit_checkpoint_pair(normal_path, stage, *, step, epoch):
    import torch

    normal_path = Path(normal_path)
    averaged_path = normal_path.with_name(normal_path.stem + "-EMA.ckpt")
    ordinary, averaged = load_checkpoint(normal_path), load_checkpoint(averaged_path)
    for item in (ordinary, averaged):
        require(item["global_step"] == step and item["epoch"] == epoch, "checkpoint step/epoch differs")
        require(item["nsamples_processed"] == step, "checkpoint sample counter differs")
        require(all("_orig_mod" not in key for key in item["state_dict"]), "compiled checkpoint key leaked")
        finite_tree(item["state_dict"], "checkpoint.model")
    require(ordinary["state_dict"].keys() == averaged["state_dict"].keys(), "paired model keys differ")
    compare_tree(ordinary.get("lr_schedulers", []), averaged.get("lr_schedulers", []), "paired.schedulers")
    require(len(ordinary["optimizer_states"]) == len(averaged["optimizer_states"]) == 1, "expected one optimizer")
    wrapped = averaged["optimizer_states"][0]
    require({"opt", "ema", "current_step", "decay", "every_n_steps"} <= wrapped.keys(), "EMA checkpoint has incomplete state")
    paired_comparison = compare_tree(ordinary["optimizer_states"][0], wrapped["opt"], "paired.optimizer")
    moments = moment_inventory(wrapped["opt"], expected_step=step)
    finite_tree(wrapped, "checkpoint.ema_optimizer")
    require(wrapped["current_step"] == step, "EMA checkpoint count differs")
    weights = [(key, value) for key, value in averaged["state_dict"].items() if stage != "fm" or not key.startswith("autoencoder.")]
    require(len(weights) == len(wrapped["ema"]), "EMA parameter count differs")
    for (name, value), ema in zip(weights, wrapped["ema"]):
        compare_tree(ema, value, f"checkpoint.ema.{name}")
    require(sum(value.numel() for _, value in weights) > 1_000_000, "checkpoint is not the real training model")
    require(any(not torch.equal(ordinary["state_dict"][name], value) for name, value in weights), "ordinary and EMA model weights never differ")
    frozen_count = 0
    if stage == "fm":
        for name, value in ordinary["state_dict"].items():
            if name.startswith("autoencoder."):
                compare_tree(averaged["state_dict"][name], value, f"paired.frozen.{name}")
                frozen_count += 1
        require(frozen_count > 0 and len(ordinary.get("lr_schedulers", [])) == 1, "FM frozen VAE/scheduler is missing")
    return {"step": step, "epoch": epoch, "moments": moments, "paired_optimizer": paired_comparison,
            "model_tensors": len(ordinary["state_dict"]), "ema_tensors": len(weights), "frozen_vae_tensors": frozen_count}


def csv_loss_values(path, fields):
    with Path(path).open(newline="") as handle:
        reader = csv.DictReader(handle)
        field = next((name for name in fields if name in (reader.fieldnames or [])), None)
        require(field is not None, f"missing CSV loss columns: {fields}")
        values = [float(row[field]) for row in reader if row.get(field, "") != ""]
    require(values and all(math.isfinite(value) for value in values), f"invalid CSV losses: {field}")
    return values


def audit_trace(root, case):
    require(case["kind"] == "instrumented", "pre-update tracing is available only for instrumented cases")
    trace = json.loads((Path(root) / "traces" / case["name"] / "state_checks.json").read_text())
    require(trace["kind"] == "instrumented" and trace["completed"] is True, "instrumented run did not complete")
    require(trace["updates"] == len(trace["training_losses"]) == 1, "instrumented restart must execute one update")
    require(not trace["validation_losses"], "one-step restart unexpectedly validated")
    for phase in ("restored_on_train_start", "restored_before_optimizer_step"):
        require(trace[phase]["global_step"] == 64 and trace[phase]["epoch"] == 1, "pre-update state was not checked at the correct boundary")
        for component in ("model", "optimizers"):
            require(trace[phase]["compared"][component].get("tensors", 0) > 0, f"no {component} tensors compared")
    require(trace["first_update"]["reference_matches"] is True and trace["first_update"]["ema_values_checked"] is True, "next update reference was not checked")
    require(trace["end"]["global_step"] == 65 and trace["end"]["samples"] == 65, "instrumented restart did not advance")
    require(all(math.isfinite(value) for value in trace["training_losses"]), "non-finite traced losses")
    if case["stage"] == "fm":
        require(trace.get("frozen_vae_comparison", {}).get("tensors", 0) > 0, "frozen FM VAE tensors were not compared")
    if case["mode"] == "compiled":
        require(trace["compiled_entry_points"] == 2, "training/evaluation compile entries are missing")
    return trace


def audit_case(manifest, case):
    root = Path(manifest["output_root"])
    record = json.loads((root / "commands" / (case["name"] + ".json")).read_text())
    require(record["returncode"] == 0 and record["status"] == "completed", "public CLI process did not succeed")
    require(record["command"][1:] == cli_command(case, record["command"][0])[1:], "executed public CLI command differs")
    run = Path(case["run_directory"])
    source = case["source"]
    result = {"case": case["name"], "kind": case["kind"], "stage": case["stage"], "mode": case["mode"]}
    result["checkpoints"] = audit_checkpoint_pair(run / "checkpoints/last.ckpt", case["stage"],
                                                   step=65 if source else 64, epoch=1 if source else 0)
    if case["stage"] == "fm":
        checkpoint = load_checkpoint(run / "checkpoints/last.ckpt")
        initial = load_checkpoint(Path(manifest["resource_root"]) / "ckpts/autoencoder.ckpt")
        frozen = {name[len("autoencoder."):]: value for name, value in checkpoint["state_dict"].items()
                  if name.startswith("autoencoder.")}
        result["frozen_vae_from_resource"] = compare_tree(frozen, initial["state_dict"], "frozen_vae_from_resource")
        del checkpoint, initial, frozen
    train_losses = csv_loss_values(run / "metrics/metrics.csv", ("train_loss", "train/loss_step"))
    require(len(train_losses) == (1 if source else 64), "real training loss count differs")
    result["training_losses"] = len(train_losses)
    result["training_loss_range"] = [min(train_losses), max(train_losses)]
    if not source:
        validation_losses = csv_loss_values(run / "metrics/metrics.csv", ("validation_loss/loss_step",))
        require(len(validation_losses) == 16, "real validation loss count differs")
        csv_loss_values(run / "metrics/metrics.csv", ("validation_loss/loss_epoch",))
        checkpoint = load_checkpoint(run / "checkpoints/last.ckpt")
        progress = checkpoint["loops"]["fit_loop"]["epoch_loop.val_loop.batch_progress"]
        require(progress["total"]["completed"] == 16, "checkpoint validation batch count differs")
        result["validation_losses"] = len(validation_losses)
    else:
        result["validation_losses"] = 0
    log = (root / "logs" / (case["name"] + ".log")).read_text(errors="replace")
    if case["mode"] == "compiled":
        require("torchinductor" in log.lower(), "no default Inductor execution was logged")
        require("Compiling denoiser trunk forward" in log, "runner did not enable compilation")
        require("won't convert" not in log.lower() and "falling back to eager" not in log.lower(), "compiler fallback was logged")
    if case["kind"] == "instrumented":
        result["state_checks"] = audit_trace(root, case)
    else:
        result["pre_update_state_checked"] = False
    result["passed"] = True
    write_json(root / "reports" / (case["name"] + ".json"), result)
    return result


def execute_case(manifest, case, *, python=sys.executable, timeout=None):
    root = Path(manifest["output_root"])
    if case["source"]:
        prerequisite = root / "reports" / (case["source"] + ".json")
        require(prerequisite.is_file() and json.loads(prerequisite.read_text()).get("passed") is True,
                "audit the source epoch before executing its restart")
    run = Path(case["run_directory"])
    require(not run.exists(), "run directory already exists; choose a new output root")
    command = cli_command(case, python)
    environment = os.environ.copy()
    for name in ("TORCHDYNAMO_DISABLE", "TORCH_COMPILE_DISABLE", "TORCHDYNAMO_SUPPRESS_ERRORS"):
        require(environment.get(name, "0").lower() not in ("1", "true"), f"{name} disables real compilation")
    require(not environment.get("TORCHDYNAMO_BACKEND"), "external compiler backend override is unsupported")
    checkout = Path(manifest["checkout"])
    environment["PYTHONPATH"] = os.pathsep.join(filter(None, (str(checkout), str(checkout / "tests"), environment.get("PYTHONPATH", ""))))
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["TORCHINDUCTOR_CACHE_DIR"] = str(root / "inductor_cache")
    environment.setdefault("OMP_NUM_THREADS", "1")
    if case["mode"] == "compiled":
        environment["TORCH_LOGS"] = "inductor"
    path = root / "commands" / (case["name"] + ".json")
    require(not path.exists(), "case already has a command record")
    record = {"case": case["name"], "kind": case["kind"], "command": command,
              "cwd": str(checkout), "status": "running", "returncode": None,
              "cuda_visible_devices": environment.get("CUDA_VISIBLE_DEVICES")}
    write_json(path, record)
    log = root / "logs" / (case["name"] + ".log")
    log.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    try:
        with log.open("x") as handle:
            completed = subprocess.run(command, cwd=checkout, env=environment, stdout=handle,
                                       stderr=subprocess.STDOUT, timeout=timeout, check=False)
        record.update(returncode=completed.returncode, status="completed")
    except BaseException as error:
        record.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        record["elapsed_seconds"] = time.monotonic() - started
        write_json(path, record)
    require(completed.returncode == 0, f"public CLI failed; inspect {log}")
    return audit_case(manifest, case)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    command = commands.add_parser("prepare", help="write immutable external configurations; do not train")
    command.add_argument("--output-root", required=True)
    command.add_argument("--resource-root", required=True)
    command.add_argument("--checkout", default=str(Path(__file__).resolve().parents[2]))
    command.add_argument("--split-csv")
    command.add_argument("--accelerator", choices=("gpu", "cpu"), default="gpu")
    for action in ("execute", "audit", "audit-states", "resume-pairs"):
        command = commands.add_parser(action)
        command.add_argument("--output-root", required=True)
        if action != "resume-pairs":
            command.add_argument("--case", required=True)
        else:
            command.add_argument("--kind", choices=("public", "instrumented", "all"), default="all")
        if action == "execute":
            command.add_argument("--python", default=sys.executable)
            command.add_argument("--timeout", type=float)
    args = parser.parse_args(argv)
    if args.action == "prepare":
        result = prepare(args)
    else:
        manifest = read_manifest(args.output_root)
        if args.action == "resume-pairs":
            result = [{**case, "command": cli_command(case, sys.executable)} for case in manifest["cases"]
                      if case["source"] and args.kind in ("all", case["kind"])]
        else:
            case = find_case(manifest, args.case)
            if args.action == "execute":
                result = execute_case(manifest, case, python=args.python, timeout=args.timeout)
            elif args.action == "audit":
                result = audit_case(manifest, case)
            else:
                result = audit_trace(manifest["output_root"], case)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
