"""Small real diagnostic runs; formal acceptance uses full generation schedules."""

from __future__ import annotations
import tempfile
from pathlib import Path
from types import SimpleNamespace


def run_smoke(task_file=None, model_dir=None, models=None, out=None):
    import yaml
    from partiflow.paths import model_root, resource_root
    from .runtime import run, _requests
    from .config import load

    selected = list(models or ["partiflow", "autoencoder"])
    directory = (
        Path(out).expanduser().resolve()
        if out
        else Path(tempfile.mkdtemp(prefix="partiflow-smoke-"))
    )
    directory.mkdir(parents=True, exist_ok=True)
    source = (
        Path(task_file).expanduser().resolve()
        if task_file
        else resource_root() / "configs/tasks/example.yaml"
    )
    args = SimpleNamespace(
        task_file=source,
        task=None,
        override=[],
        model_dir=model_dir,
        print_config=False,
        config=None,
    )
    cfg = load(args)
    name = next(iter(cfg.tasks)) if task_file else "klk2_freegen"
    from omegaconf import OmegaConf

    task = OmegaConf.to_container(cfg.tasks[name], resolve=True)
    task["sample_num"] = 1
    if not task_file:
        task["lens_cfg"] = dict(nres=[12], min_len=None, max_len=None, step_len=None)
    path = directory / "task.yaml"
    path.write_text(yaml.safe_dump({"smoke": task}))
    args.task_file = path
    args.batch_size = 1
    args.num_workers = 0
    args.compile = False
    args.seed = 2026
    args.override = ["generation.args.nsteps=20", "trainer.precision=32-true"]
    root = model_root(model_dir)
    results = {}
    structure = None

    for model in [value for value in selected if value.startswith("partiflow")]:
        args.ckpt = root / "ckpts" / f"{model}.ckpt"
        args.out = directory / model
        status = run(args)
        if status:
            raise RuntimeError(
                f"{model} diagnostic generation failed with status {status}"
            )
        from partiflow.common.results import read_results

        _, rows = read_results(args.out / "smoke/result.csv")
        if not rows or any(row["error"] or not row["filename"] for row in rows):
            raise RuntimeError(f"{model} diagnostic produced incomplete results")
        structure = args.out / "smoke" / rows[0]["filename"]
        results[model] = {"status": "complete", "samples": len(rows), "steps": 20}
    if "autoencoder" in selected:
        import torch
        from partiflow.vae.autoencoder import AutoEncoder

        vae = AutoEncoder.load_from_checkpoint(
            root / "ckpts/autoencoder.ckpt",
            map_location="cpu",
            strict=True,
            create_val_tmp=False,
        ).eval()
        configuration = load(args)
        dm, _, _ = _requests(configuration)
        dm.setup("predict")
        batch = next(iter(dm.predict_dataloader()))
        with torch.no_grad():
            _, decoded = vae.predict_step(batch, 0)
        for key in ("coors_nm", "seq_logits", "z_latent"):
            if not torch.isfinite(decoded[key]).all():
                raise RuntimeError(f"VAE diagnostic has non-finite {key}")
        results["autoencoder"] = {
            "status": "complete",
            "residues": int(decoded["residue_mask"].sum()),
        }
        del vae
    workflow_models = [
        model for model in selected if model in {"af2", "proteinmpnn", "solublempnn"}
    ]
    if workflow_models:
        from partiflow.workflows.smoke import run_structure_smoke

        if structure is None or not task_file:
            structure = resource_root() / "pdbs/design/mdm2/mdm2_motif.pdb"
        workflow = run_structure_smoke(
            structure, "L", root, workflow_models, directory / "workflow"
        )
        if workflow.get("status") != "complete":
            raise RuntimeError("Structure-model diagnostic did not complete")
        results.update(workflow["models"])
    if set(results) != set(selected):
        raise RuntimeError(
            f"Unsupported smoke targets: {sorted(set(selected) - set(results))}"
        )
    return {
        "status": "complete",
        "directory": str(directory),
        "models": results,
        "diagnostic": True,
    }
