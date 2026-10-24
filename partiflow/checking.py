"""Read-only resource checks and explicit real CPU model-load checks."""

from __future__ import annotations
import importlib.metadata
import json
import os
from pathlib import Path
from partiflow.download import MODEL_TARGETS, resource_status
from partiflow.paths import config_dir, model_root, resource_root


def check_task(path):
    import yaml

    path = Path(path).expanduser().resolve()
    tasks = yaml.safe_load(path.read_text())
    if not isinstance(tasks, dict) or not tasks:
        raise ValueError("Task YAML needs nonempty named task mappings")
    valid = {"denovo", "motif", "docking", "inv_folding", "packing"}
    results = []
    bundled = path.is_relative_to(config_dir())
    for name, task in tasks.items():
        if not isinstance(task, dict) or task.get("task_type") not in valid:
            raise ValueError(f"{name}: task_type must be one of {sorted(valid)}")
        if not task.get("pdb_path"):
            raise ValueError(f"{name}: pdb_path is required")
        source = Path(task["pdb_path"]).expanduser()
        source = (
            (resource_root() / source if bundled else Path.cwd() / source)
            if not source.is_absolute()
            else source
        )
        if not source.is_file():
            raise FileNotFoundError(f"{name}: missing structure {source}")
        count = task.get("sample_num")
        if count is not None and (type(count) is not int or count < 1):
            raise ValueError(f"{name}: sample_num must be a positive integer")
        results.append(
            {
                "task": str(name),
                "status": "present",
                "task_type": task["task_type"],
                "pdb_path": str(source.resolve()),
            }
        )
    return {
        "status": "complete",
        "scope": "basic_schema_and_input_paths",
        "tasks": results,
    }


def _load_model(name, root):
    if name.startswith("partiflow"):
        from partiflow.design import load_for_check

        model = load_for_check(
            root / "ckpts" / f"{name}.ckpt",
            root / "ckpts" / "autoencoder.ckpt",
            device="cpu",
        )
        if model is None:
            raise RuntimeError("Strict model loader returned no model")
        del model
        return {"target": name, "status": "loaded", "device": "cpu", "strict": True}
    if name == "autoencoder":
        from partiflow.vae.autoencoder import AutoEncoder

        model = AutoEncoder.load_from_checkpoint(
            root / "ckpts" / "autoencoder.ckpt",
            map_location="cpu",
            strict=True,
            create_val_tmp=False,
        )
        model.eval()
        del model
        return {"target": name, "status": "loaded", "device": "cpu", "strict": True}
    if name in {"proteinmpnn", "solublempnn"}:
        from partiflow.workflows.sequence import MPNNBackend

        backend = MPNNBackend(device="cpu")
        try:
            flavor = "vanilla" if name == "proteinmpnn" else "soluble"
            backend._model(
                {
                    "checkpoint": str(root / "proteinmpnn" / flavor / "v_48_020.pt"),
                    "ca_only": False,
                    "backbone_noise": 0.0,
                }
            )
        finally:
            backend.close()
        return {"target": name, "status": "loaded", "device": "cpu", "strict": True}
    if name == "af2":
        os.environ["JAX_PLATFORMS"] = "cpu"
        from partiflow.workflows.af2 import AF2Backend

        backend = AF2Backend(
            root, {"models": [1, 2], "samples": 1, "recycles": 1}, device="cpu"
        )
        try:
            backend._get_model("complex")
            backend._get_model("monomer")
        finally:
            backend.close()
        return {
            "target": name,
            "status": "loaded",
            "device": "cpu",
            "model_numbers": [1, 2],
            "protocols": ["complex", "monomer"],
        }
    raise ValueError(f"No model loader for {name}")


def run(args):
    selected = (
        list(MODEL_TARGETS)
        if getattr(args, "all", False)
        else list(getattr(args, "model", None) or ["partiflow", "autoencoder"])
    )
    unknown = set(selected) - set(MODEL_TARGETS)
    if unknown:
        raise ValueError(f"Unknown models: {sorted(unknown)}")
    root = model_root(getattr(args, "model_dir", None))
    package = Path(__file__).parent
    required = [
        config_dir() / "design.yaml",
        config_dir() / "validate.yaml",
        config_dir() / "preset.yaml",
        package / "utils" / "relax_assets" / "topology.xml",
        package / "postprocess" / "rosetta_ncaa" / "support_matrix.csv",
        package / "_vendor" / "proteinmpnn" / "protein_mpnn_utils.py",
        package / "_vendor" / "af2" / "colabdesign" / "__init__.py",
    ]
    resources = [{"path": str(path), "present": path.is_file()} for path in required]
    report = {
        "model_root": str(root),
        "resources": resources,
        "models": [],
        "mode": "paths",
    }
    failed = any(not item["present"] for item in resources)
    if getattr(args, "task_file", None):
        try:
            report["task_check"] = check_task(args.task_file)
        except (OSError, ValueError) as exc:
            report["task_check"] = {
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
            }
            failed = True
    for name in dict.fromkeys(selected):
        status = resource_status(name, root)
        report["models"].append(status)
        failed |= status["status"] != "present"
    if getattr(args, "data_dir", None) or getattr(args, "config", None):
        report["pepinter"] = resource_status(
            "pepinter",
            root,
            data_dir=getattr(args, "data_dir", None),
            config=getattr(args, "config", None),
        )
        failed |= report["pepinter"]["status"] != "present"
    if getattr(args, "load", False):
        report["mode"] = "load"
        report["loads"] = []
        for status in report["models"]:
            if status["status"] != "present":
                report["loads"].append(
                    {
                        "target": status["target"],
                        "status": "unavailable",
                        "error": "required resource files are missing",
                    }
                )
                continue
            try:
                report["loads"].append(_load_model(status["target"], root))
            except Exception as exc:
                report["loads"].append(
                    {
                        "target": status["target"],
                        "status": "failed",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                failed = True
    if getattr(args, "smoke", False):
        report["mode"] = "smoke"
        if failed:
            report["smoke"] = {
                "status": "unavailable",
                "error": "required resource/input checks failed",
            }
        else:
            try:
                from partiflow.design import run_smoke

                result = run_smoke(
                    task_file=getattr(args, "task_file", None),
                    model_dir=root,
                    models=selected,
                )
                complete = (type(result) is int and result == 0) or (
                    isinstance(result, dict)
                    and result.get("status") in {"success", "complete"}
                )
                if not complete:
                    raise RuntimeError(
                        f"Smoke execution did not report completion: {result}"
                    )
                report["smoke"] = {"status": "complete", "result": result}
            except Exception as exc:
                report["smoke"] = {
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                }
                failed = True
    report["status"] = "failed" if failed else "complete"
    print(json.dumps(report, indent=2, default=str))
    return int(failed)
