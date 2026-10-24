"""Original generator with deferred evaluation and public result publication."""

from __future__ import annotations
import copy
import json
import os
import shutil
from collections import defaultdict
from pathlib import Path


def load_for_check(model_path, autoencoder_path, device="cpu"):
    import torch
    from omegaconf import OmegaConf
    from partiflow.model import PartiFlow

    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    params = checkpoint.get("hyper_parameters", {})
    if "cfg_exp" not in params:
        raise ValueError("FM checkpoint is missing cfg_exp")
    cfg = copy.deepcopy(params["cfg_exp"])
    if not OmegaConf.is_config(cfg):
        cfg = OmegaConf.create(cfg)
    model = PartiFlow(
        cfg, autoencoder_ckpt_path=str(autoencoder_path), create_val_tmp=False
    )
    state = {
        key.replace("._orig_mod.", "."): value
        for key, value in checkpoint["state_dict"].items()
    }
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def _model(cfg):
    import torch

    model = load_for_check(cfg.checkpoint.model_ckpt, cfg.checkpoint.autoencoder_ckpt)
    nn_ag = None
    args = cfg.generation.args
    if float(args.get("ag_ratio", 0)) > 0 and float(args.get("guidance_w", 1)) != 1:
        path = args.get("ag_ckpt_path")
        if not path:
            raise ValueError("Autoguidance requires ag_ckpt_path")
        guidance = load_for_check(path, cfg.checkpoint.autoencoder_ckpt)
        nn_ag = guidance.nn
        for parameter in nn_ag.parameters():
            parameter.requires_grad = False
    model.configure_inference(
        cfg.generation,
        nn_ag,
        search_cfg=cfg.search,
        receptor_source=cfg.output.receptor_source,
    )
    if cfg.compile.enabled:
        settings = dict(
            mode=str(cfg.compile.mode),
            dynamic=bool(cfg.compile.dynamic),
            fullgraph=bool(cfg.compile.fullgraph),
        )
        model.nn = torch.compile(model.nn, **settings)
        if model.nn_ag is not None:
            model.nn_ag = torch.compile(model.nn_ag, **settings)
    return model


def _requests(cfg):
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    from .tasks import validate_task, validate_task_postprocess_config

    tasks = OmegaConf.to_container(cfg.tasks, resolve=True)
    for name, task in tasks.items():
        validate_task(task, name)
        validate_task_postprocess_config(task, name, inspect_reference_chemistry=True)
    settings = OmegaConf.to_container(cfg.datamodule, resolve=True)
    settings["raw_tasks"] = tasks
    datamodule = instantiate(settings)
    requests = datamodule._process_tasks()
    if not requests:
        raise ValueError("Selected tasks expand to no requests")
    return datamodule, requests, tasks


def _prepare(cfg, requests, tasks):
    from omegaconf import OmegaConf

    root = Path(cfg.output.root_dir) / cfg.output.run_name
    for name in tasks:
        path = root / name
        if path.exists() and (not path.is_dir() or any(path.iterdir())):
            raise FileExistsError(
                f"Design task directory must be absent or empty: {path}"
            )
    replicas = int(cfg.search.best_of_n.replicas) if cfg.search.algorithm else 1
    for name, task in tasks.items():
        path = root / name
        (path / ".state").mkdir(parents=True, exist_ok=True)
        (path / "logs").mkdir(exist_ok=True)
        count = sum(r["task_name"] == name for r in requests)
        resolved = OmegaConf.to_container(cfg, resolve=True)
        resolved["tasks"] = {name: task}
        resolved["task_names"] = [name]
        resolved["_runtime"] = {
            "planned_base_samples": count,
            "replicas_per_base": replicas,
            "planned_candidates": count * replicas,
        }
        OmegaConf.save(OmegaConf.create(resolved), path / "resolved_config.yaml")
    return root, replicas


def _read_candidates(root, tasks):
    from partiflow.common.records import Candidate

    candidates = []
    failures = []
    for task in tasks:
        for file in sorted((root / task / ".state").glob("generation_rank*.jsonl")):
            for line in file.read_text().splitlines():
                record = json.loads(line)
                error = record.pop("error", "")
                if error:
                    record["error"] = error
                    failures.append(record)
                    continue
                if isinstance(record.get("topology_entries"), str):
                    record["topology_entries"] = json.loads(record["topology_entries"])
                record["pdb_path"] = Path(record["pdb_path"])
                candidates.append(Candidate(**record))
    ids = [c.candidate_id for c in candidates]
    if len(ids) != len(set(ids)):
        raise RuntimeError("Duplicate global candidate identities")
    return sorted(candidates, key=lambda c: c.candidate_ordinal), failures


def _publish(cfg, root, requests, tasks, candidates, failures):
    from omegaconf import OmegaConf
    from partiflow.common.records import Evaluation
    from partiflow.common.results import BASE_FIELDS, TAIL_FIELDS, write_results
    from .constraints import populate, public_identity, serialize_connections

    raw = []
    for candidate in candidates:
        try:
            atoms = populate(candidate)
            candidate.context_chains = tuple(
                dict.fromkeys(
                    a["chain"] for a in atoms if a["chain"] != candidate.binder_chain
                )
            )
            raw.append(candidate)
        except ValueError as exc:
            failures.append(
                {
                    "task_name": candidate.task_name,
                    "base_sample_id": candidate.base_sample_id,
                    "candidate_id": candidate.candidate_id,
                    "error": f"constraints: {exc}",
                }
            )
    reward = (
        OmegaConf.to_container(cfg.get("reward"), resolve=True)
        if cfg.get("reward") is not None
        else None
    )
    if reward is None:
        evaluations = [
            Evaluation(c.candidate_id, pdb_path=c.pdb_path, sequence=c.sequence)
            for c in raw
        ]
    else:
        from partiflow.workflows.reward import evaluate_candidates, close
        from partiflow.paths import model_root

        evaluations = []
        try:
            for task in tasks:
                group = [candidate for candidate in raw if candidate.task_name == task]
                if not group:
                    continue
                current = evaluate_candidates(
                    group,
                    reward,
                    model_root(cfg.get("_model_dir")),
                    root / task / ".state" / "reward",
                )
                ids = [item.candidate_id for item in current]
                if len(ids) != len(group) or set(ids) != {
                    candidate.candidate_id for candidate in group
                }:
                    raise RuntimeError(
                        "Reward returned inconsistent candidate identities"
                    )
                evaluations.extend(current)
        finally:
            close()
    evaluated = {item.candidate_id: item for item in evaluations}
    valid = []
    for candidate in raw:
        item = evaluated[candidate.candidate_id]
        if item.error:
            failures.append(
                {
                    "task_name": candidate.task_name,
                    "base_sample_id": candidate.base_sample_id,
                    "candidate_id": candidate.candidate_id,
                    "error": f"reward: {item.error}",
                }
            )
            continue
        if item.pdb_path and Path(item.pdb_path) != candidate.pdb_path:
            original = root / candidate.task_name / "original"
            original.mkdir(exist_ok=True)
            shutil.copy2(
                candidate.pdb_path,
                original
                / f"sample{candidate.base_sample_id}_candidate{candidate.replica_id}.pdb",
            )
            if Path(item.pdb_path).suffix.lower() != ".pdb":
                from partiflow.workflows.structure import export_pdb

                converted = candidate.pdb_path.with_name(
                    candidate.pdb_path.stem + "_fold.pdb"
                )
                export_pdb(item.pdb_path, converted)
                candidate.pdb_path = converted
            else:
                candidate.pdb_path = Path(item.pdb_path)
        candidate.sequence = item.sequence or candidate.sequence
        from partiflow.postprocess.records import pdb_atoms

        final_atoms = pdb_atoms(candidate.pdb_path)
        candidate.length = len(
            {
                (a["res_id"], a["ins_code"])
                for a in final_atoms
                if a["chain"] == candidate.binder_chain and a["atom"] == "CA"
            }
        )
        valid.append(candidate)
    ranks = {}
    if reward is not None:
        groups = defaultdict(list)
        for c in valid:
            groups[(c.task_name, c.length)].append(c)
        for group in groups.values():
            for rank, c in enumerate(
                sorted(
                    group,
                    key=lambda c: (
                        -evaluated[c.candidate_id].score,
                        c.candidate_ordinal,
                    ),
                ),
                start=1,
            ):
                ranks[c.candidate_id] = rank
    from partiflow.postprocess import process_candidates

    settings = OmegaConf.to_container(cfg.postprocess_runtime, resolve=True)
    processed = process_candidates(valid, settings, enabled_only=True)
    by_task = defaultdict(list)
    states = defaultdict(list)
    for candidate, result in zip(valid, processed):
        rank = ranks.get(candidate.candidate_id, "")
        name = (
            f"sample{candidate.base_sample_id}.pdb"
            if reward is None
            else f"sample{candidate.base_sample_id}_{candidate.binder_chain}_rank{rank}.pdb"
        )
        directory = root / candidate.task_name / ".state" / "publication"
        directory.mkdir(parents=True, exist_ok=True)
        destination = directory / name
        if destination.exists():
            raise RuntimeError(f"Duplicate public candidate filename: {name}")
        candidate.pdb_path = Path(result.get("pdb_path") or candidate.pdb_path)
        error = str(result.get("error") or "")
        try:
            identity = public_identity(candidate)
            serialize_connections(candidate)
        except ValueError as exc:
            failures.append(
                {
                    "task_name": candidate.task_name,
                    "base_sample_id": candidate.base_sample_id,
                    "candidate_id": candidate.candidate_id,
                    "error": f"final structure: {exc}",
                }
            )
            continue
        temporary = destination.with_suffix(".tmp")
        shutil.copyfile(candidate.pdb_path, temporary)
        os.replace(temporary, destination)
        row = dict(
            identity,
            filename=name,
            base_sample_id=candidate.base_sample_id,
            design_modality="protein",
            reward_method="af2multimer" if reward else "",
            global_rank=rank,
            contig_string=candidate.contig_string,
            tip_atoms=candidate.tip_atoms,
            postprocess_status={
                "success": "succeeded",
                "failed": "failed",
                "skipped": "disabled",
            }.get(result["status"], "not_run"),
            error=error,
        )
        metrics = evaluated[candidate.candidate_id].metrics
        if set(metrics).intersection(BASE_FIELDS + TAIL_FIELDS):
            raise RuntimeError("Reward metric collides with a public result field")
        row.update(metrics)
        by_task[candidate.task_name].append(row)
        state = dict(
            candidate.metadata,
            filename=name,
            base_sample_id=candidate.base_sample_id,
            sample_id=candidate.candidate_id,
            binder_chain=candidate.binder_chain,
            topology_entries=candidate.topology_entries,
        )
        states[candidate.task_name].append(state)
    for task in tasks:
        count = sum(request["task_name"] == task for request in requests)
        replicas = int(cfg.search.best_of_n.replicas) if cfg.search.algorithm else 1
        rows = by_task[task]
        successful = {int(row["base_sample_id"]) for row in rows}
        task_failures = [
            failure for failure in failures if failure["task_name"] == task
        ]
        for base in range(1, count + 1):
            if base in successful:
                continue
            errors = [
                str(item["error"])
                for item in task_failures
                if int(item["base_sample_id"]) == base
            ]
            if not errors:
                raise RuntimeError(f"Missing candidate records for {task} base {base}")
            rows.append(
                {
                    "base_sample_id": base,
                    "reward_method": "af2multimer" if reward else "",
                    "postprocess_status": "not_run",
                    "error": "; ".join(dict.fromkeys(errors)),
                }
            )
        rows.sort(
            key=lambda row: (
                int(row["base_sample_id"]),
                int(row.get("global_rank") or 0),
            )
        )
        state = {
            "task": tasks[task],
            "generation": {
                "planned_base_samples": count,
                "replicas_per_base": replicas,
                "planned_candidates": count * replicas,
            },
            "samples": states[task],
            "failures": task_failures,
        }
        directory = root / task / ".state" / "publication"
        directory.mkdir(parents=True, exist_ok=True)
        state_path = root / task / ".state" / "design.json"
        staged_state = directory / "design.json"
        staged_state.write_text(json.dumps(state, allow_nan=False, indent=2))
        write_results(directory / "result.csv", rows)
        promoted = []
        state_published = False
        try:
            for row in rows:
                if not row.get("filename"):
                    continue
                destination = root / task / row["filename"]
                if destination.exists():
                    raise FileExistsError(
                        f"Public structure already exists: {destination}"
                    )
                os.replace(directory / row["filename"], destination)
                promoted.append(destination)
            os.replace(staged_state, state_path)
            state_published = True
            os.replace(directory / "result.csv", root / task / "result.csv")
        except BaseException:
            for destination in promoted:
                destination.unlink(missing_ok=True)
            if state_published:
                state_path.unlink(missing_ok=True)
            raise
        shutil.rmtree(directory, ignore_errors=True)
    return sum(bool(row.get("error")) for rows in by_task.values() for row in rows)


def run(args):
    from omegaconf import OmegaConf
    from .config import load

    cfg = load(args)
    if getattr(args, "print_config", False):
        print(OmegaConf.to_yaml(cfg, resolve=True))
        return 0
    import gc
    import lightning as L
    import torch
    from loguru import logger
    from .output import CandidateCollector

    L.seed_everything(int(cfg.seed), workers=True)
    datamodule, requests, tasks = _requests(cfg)
    rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", 0)))
    if rank == 0:
        root, replicas = _prepare(cfg, requests, tasks)
    else:
        root = Path(cfg.output.root_dir) / cfg.output.run_name
        replicas = int(cfg.search.best_of_n.replicas) if cfg.search.algorithm else 1
    L.seed_everything(int(cfg.seed) + rank, workers=True)
    model = _model(cfg)
    sinks = []

    class RuntimeSetup(L.Callback):
        def on_predict_start(self, trainer, pl_module):
            L.seed_everything(int(cfg.seed) + trainer.global_rank, workers=True)
            for task in tasks:
                sinks.append(
                    logger.add(
                        root
                        / task
                        / "logs"
                        / f"generation_rank{trainer.global_rank}.log"
                    )
                )

    strategy = cfg.trainer.strategy
    if str(strategy).lower() == "ddp":
        from datetime import timedelta

        strategy = L.pytorch.strategies.DDPStrategy(timeout=timedelta(hours=24))
    trainer = L.Trainer(
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.ngpus_per_node,
        num_nodes=cfg.trainer.num_nodes,
        strategy=strategy,
        precision=cfg.trainer.precision,
        default_root_dir=str(root),
        callbacks=[
            RuntimeSetup(),
            CandidateCollector(
                root,
                requests,
                replicas=replicas,
                receptor_source=cfg.output.receptor_source,
            ),
        ],
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=True,
    )
    try:
        trainer.predict(model, datamodule=datamodule, return_predictions=False)
        distributed = (
            torch.distributed.is_available() and torch.distributed.is_initialized()
        )
        if distributed:
            torch.distributed.barrier()
        model.cpu()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        errors = 0
        if trainer.is_global_zero:
            candidates, failures = _read_candidates(root, tasks)
            if len(candidates) + len(failures) != len(requests) * replicas:
                raise RuntimeError(
                    "Generated candidate count differs from the planned budget"
                )
            errors = _publish(cfg, root, requests, tasks, candidates, failures)
        if distributed:
            result = torch.tensor([errors], device=trainer.strategy.root_device)
            torch.distributed.broadcast(result, src=0)
            errors = int(result.item())
        return 2 if errors else 0
    finally:
        for sink in sinks:
            logger.remove(sink)


def run_smoke(*args, **kwargs):
    from .smoke import run_smoke as execute

    return execute(*args, **kwargs)
