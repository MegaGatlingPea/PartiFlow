"""Compare native original and current PartiFlow forward paths on CPU.

Example (all generated files must be outside either source checkout)::

    CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
      python tests/acceptance/core.py --original-root /path/to/PartiFlow-v2 \
      --model-root /path/to/resources --fixture-config /outside/resolved_config.yaml \
      --output-root /outside/core

The original ``src`` package is imported directly, without source rewriting or
substitute modules. Both implementations load the same checkpoint with strict
keys, receive cloned tensors and reset CPU RNG states, and execute their native
400-step solver. A delegating observer saves every native integration step.
This is a forward computation gate; it does not check training restoration or
the physical quality of generated structures.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import copy
import importlib
import inspect
import json
import os
from pathlib import Path
import random
import sys
import time
import traceback


class AcceptanceError(AssertionError):
    """A required core comparison failed."""


def require(condition, message):
    if not condition:
        raise AcceptanceError(message)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def external_root(root, *checkouts):
    root = Path(root).expanduser().resolve()
    for checkout in checkouts:
        checkout = Path(checkout).expanduser().resolve()
        require(root != checkout and checkout not in root.parents and root not in checkout.parents,
                f"Runtime directory must be outside source checkout: {root}")
    return root


def cpu_tree(value):
    import torch

    if isinstance(value, torch.Tensor):
        require(value.device.type == "cpu", "Non-CPU tensor reached the CPU gate")
        return value.detach().clone()
    if isinstance(value, Mapping):
        return {key: cpu_tree(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(cpu_tree(item) for item in value)
    return copy.deepcopy(value)


def seed_cpu(seed):
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.random.default_generator.manual_seed(seed)


def rng_state():
    import numpy as np
    import torch

    numpy_state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": (numpy_state[0], torch.from_numpy(numpy_state[1].copy()), *numpy_state[2:]),
        "torch_cpu": torch.get_rng_state().clone(),
    }


class Comparisons:
    """Record every leaf, including failed shapes, non-finite values and keys."""

    def __init__(self, path, *, rtol=1e-5, atol=1e-6):
        self.path = Path(path)
        self.rtol, self.atol = rtol, atol
        self.count = self.tensor_count = self.failures = 0
        self.max_abs_delta = 0.0
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("")

    def record(self, label, actual, expected):
        import torch

        row = {"label": label, "rtol": self.rtol, "atol": self.atol}
        try:
            if isinstance(expected, torch.Tensor):
                self.tensor_count += 1
                require(isinstance(actual, torch.Tensor), "Expected a tensor")
                row.update(shape=list(expected.shape), dtype=str(expected.dtype), elements=expected.numel())
                require(actual.device.type == expected.device.type == "cpu", "Expected CPU tensors")
                require(actual.shape == expected.shape, f"Shape differs: {tuple(actual.shape)}")
                require(actual.dtype == expected.dtype, f"Dtype differs: {actual.dtype}")
                require(bool(torch.isfinite(actual).all()) and bool(torch.isfinite(expected).all()),
                        "Non-finite tensor")
                delta = (actual.to(torch.float64) - expected.to(torch.float64)).abs()
                maximum = float(delta.max()) if delta.numel() else 0.0
                row["max_abs_delta"] = maximum
                self.max_abs_delta = max(self.max_abs_delta, maximum)
                if expected.is_floating_point():
                    torch.testing.assert_close(actual, expected, rtol=self.rtol, atol=self.atol)
                else:
                    require(torch.equal(actual, expected), "Exact tensor values differ")
            else:
                require(actual == expected, f"Values differ: {actual!r} != {expected!r}")
            row["result"] = "PASS"
        except (AssertionError, TypeError, RuntimeError) as exc:
            row.update(result="FAIL", error=str(exc))
            self.failures += 1
        self.count += 1
        with self.path.open("a") as stream:
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")

    def tree(self, label, actual, expected):
        if isinstance(expected, Mapping):
            if not isinstance(actual, Mapping) or actual.keys() != expected.keys():
                self.record(label + ".keys", sorted(actual) if isinstance(actual, Mapping) else None,
                            sorted(expected))
                return
            for key in expected:
                self.tree(f"{label}.{key}", actual[key], expected[key])
        elif isinstance(expected, (list, tuple)):
            if not isinstance(actual, (list, tuple)) or len(actual) != len(expected):
                self.record(label + ".length", len(actual) if isinstance(actual, (list, tuple)) else None,
                            len(expected))
                return
            for index, wanted in enumerate(expected):
                self.tree(f"{label}[{index}]", actual[index], wanted)
        else:
            self.record(label, actual, expected)


class StepObserver:
    """Observe a native method by calling it once, without changing its inputs."""

    def __init__(self, native_step, batch, directory, *, reference=None, comparisons=None, progress=None):
        self.native_step = native_step
        self.batch = batch
        self.directory = Path(directory)
        self.reference = Path(reference) if reference is not None else None
        self.comparisons = comparisons
        self.progress = progress
        self.steps = self.self_conditioned_steps = 0
        self.directory.mkdir(parents=True, exist_ok=True)

    def __call__(self, *args, **kwargs):
        import torch

        require(not args, "Native solver changed its integration call signature")
        state = cpu_tree(kwargs)
        state["self_conditioning"] = cpu_tree(self.batch.get("x_sc"))
        self.self_conditioned_steps += int(state["self_conditioning"] is not None)
        result = self.native_step(**kwargs)
        state["updated"] = cpu_tree(result)
        name = f"{self.steps:04d}.pt"
        torch.save(state, self.directory / name)
        if self.reference is not None:
            expected = torch.load(self.reference / name, map_location="cpu", weights_only=False)
            self.comparisons.tree(f"solver.step[{self.steps}]", state, expected)
        self.steps += 1
        if self.progress is not None and self.steps % 25 == 0:
            self.progress(self.steps)
        return result


def load_native_modules(original_root, checkout):
    original_root = Path(original_root).resolve()
    checkout = Path(checkout).resolve()
    sys.path.insert(0, str(original_root))
    sys.path.insert(0, str(checkout))
    old = importlib.import_module("src.partiflow")
    new = importlib.import_module("partiflow.model")
    old_flow = importlib.import_module("src.flow.product_space_flow_matcher")
    new_flow = importlib.import_module("partiflow.flow.product_space_flow_matcher")
    for module, root in ((old, original_root), (old_flow, original_root), (new, checkout), (new_flow, checkout)):
        require(root in Path(module.__file__).resolve().parents, f"Imported a different source tree: {module.__file__}")
    return {"original": (old, old_flow), "current": (new, new_flow)}


def fixture(config_path, seed):
    from omegaconf import OmegaConf
    from partiflow.design.runtime import _requests

    seed_cpu(seed)
    cfg = OmegaConf.load(config_path)
    require(int(cfg.generation.args.nsteps) == 400, "Core acceptance requires the full 400-step schedule")
    require(bool(cfg.generation.args.self_cond), "Core acceptance requires self conditioning")
    require(float(cfg.generation.args.get("guidance_w", 1)) == 1 and
            float(cfg.generation.args.get("ag_ratio", 0)) == 0, "Core gate expects the base model")
    require(int(cfg.generation.get("n_recycle", 0)) == 0, "Core gate expects the pretrained zero-recycle protocol")
    require(str(cfg.trainer.precision) == "32-true", "Core acceptance requires FP32")
    require(not bool(cfg.compile.enabled), "Core acceptance expects eager forward")
    cfg.datamodule.batch_size = 1
    cfg.datamodule.num_workers = 0
    cfg.datamodule.pin_memory = False
    dm, requests, _ = _requests(cfg)
    require(len(requests) == 1, "Fixture must contain exactly one request")
    dm.setup("predict")
    raw = next(iter(dm.predict_dataloader()))
    batch = cpu_tree(dict(raw.items()))
    valid = batch["mask_dict"]["coords"][..., 0, 0].bool()
    counts = {
        "valid_residues": int(valid.sum()),
        "peptide_residues": int(((batch["is_peptide"] == 1) & valid).sum()),
        "receptor_residues": int(((batch["is_peptide"] == 0) & valid).sum()),
    }
    require(counts == {"valid_residues": 241, "peptide_residues": 12, "receptor_residues": 229},
            f"Expected the 12-peptide/229-receptor fixture: {counts}")
    require(batch["coords_nm"].dtype.__str__() == "torch.float32", "Fixture coordinates must be FP32")
    return cfg, batch, counts


def run(args):
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "Set CUDA_VISIBLE_DEVICES= before starting")
    require(1 <= args.threads <= 4, "This CPU gate permits at most four threads")
    import torch
    from omegaconf import OmegaConf

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(args.threads)
    checkout = Path(__file__).resolve().parents[2]
    original_root = Path(args.original_root).expanduser().resolve()
    root = external_root(args.output_root, checkout, original_root)
    root.mkdir(parents=True, exist_ok=True)
    require(not (root / "summary.json").exists(), "Choose a fresh output directory")
    started = time.monotonic()

    def event(name, **values):
        value = {"event": name, "elapsed_seconds": round(time.monotonic() - started, 4), **values}
        print(json.dumps(value, sort_keys=True, allow_nan=False), flush=True)
        with (root / "events.jsonl").open("a") as stream:
            stream.write(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")

    comparisons = Comparisons(root / "comparisons.jsonl")
    model_root = Path(args.model_root).expanduser().resolve()
    model_path = model_root / "ckpts/partiflow.ckpt"
    ae_path = model_root / "ckpts/autoencoder.ckpt"
    command = {"argv": sys.argv, "python": sys.executable, "cwd": str(Path.cwd()),
               "environment": {key: os.environ.get(key) for key in
                               ("CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")}}
    write_json(root / "command.json", command)
    try:
        modules = load_native_modules(original_root, checkout)
        cfg, batch, counts = fixture(args.fixture_config, args.seed)
        torch.save(batch, root / "fixture.pt")
        OmegaConf.save(cfg, root / "fixture_config.yaml")
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
        cfg_exp = checkpoint["hyper_parameters"]["cfg_exp"]
        state = {key.replace("._orig_mod.", "."): value for key, value in checkpoint["state_dict"].items()}
        require(len(state) == len(checkpoint["state_dict"]), "Checkpoint key normalization collided")
        parameters = {"seed": args.seed, "stage_seeds": {}, "device": "cpu", "torch": torch.__version__,
                      "threads": torch.get_num_threads(), "interop_threads": torch.get_num_interop_threads(),
                      "rtol": comparisons.rtol, "atol": comparisons.atol, "steps": 400, "fixture": counts,
                      "original_root": str(original_root), "checkout": str(checkout),
                      "fixture_config": str(Path(args.fixture_config).resolve()),
                      "model_paths": {"flow": str(model_path), "autoencoder": str(ae_path)},
                      "sampling": OmegaConf.to_container(cfg.generation, resolve=True),
                      "implementations": {}}
        event("START", **{key: parameters[key] for key in ("device", "threads", "interop_threads", "fixture", "steps", "rtol", "atol")})
        for implementation, (module, flow_module) in modules.items():
            directory = root / implementation
            directory.mkdir(exist_ok=True)
            seed_cpu(args.seed)
            model_cfg = copy.deepcopy(cfg_exp)
            if not OmegaConf.is_config(model_cfg):
                model_cfg = OmegaConf.create(model_cfg)
            model = module.PartiFlow(model_cfg, autoencoder_ckpt_path=str(ae_path), create_val_tmp=False)
            result = model.load_state_dict(state, strict=True)
            require(not result.missing_keys and not result.unexpected_keys, "Strict checkpoint load failed")
            model = model.cpu().float().eval()
            actual_state = model.state_dict()
            require(actual_state.keys() == state.keys(), "Checkpoint parameter keys differ")
            for key, wanted in state.items():
                require(torch.equal(actual_state[key], wanted), f"Loaded checkpoint value differs: {key}")
            require(all(not p.requires_grad for p in model.autoencoder.parameters()), "VAE must stay frozen")
            sources = {"model": inspect.getfile(type(model)), "autoencoder": inspect.getfile(type(model.autoencoder)),
                       "encoder": inspect.getfile(type(model.autoencoder.encoder)),
                       "decoder": inspect.getfile(type(model.autoencoder.decoder)),
                       "trunk": inspect.getfile(type(model.nn)), "flow": inspect.getfile(type(model.fm)),
                       "base_flow": inspect.getfile(type(model.fm.base_flow_matchers["bb_ca"])),
                       "full_simulation": inspect.getfile(model.fm.full_simulation),
                       "simulation_step": inspect.getfile(model.fm.simulation_step),
                       "masks": inspect.getfile(flow_module.compute_masks_unified)}
            selected_root = original_root if implementation == "original" else checkout
            for source in sources.values():
                require(selected_root in Path(source).resolve().parents, f"Unexpected native source: {source}")
            parameters["implementations"][implementation] = {
                "source_paths": sources, "strict_checkpoint_tensors": len(state),
                "strict_checkpoint_elements": sum(value.numel() for value in state.values()),
                "motif_constraint_mode": model.motif_constraint_mode,
            }
            outputs = {}
            random_states = {}

            def stage(name, offset, callback):
                parameters["stage_seeds"][name] = args.seed + offset
                seed_cpu(args.seed + offset)
                before = rng_state()
                with torch.no_grad():
                    value = callback()
                random_states[name] = {"before": before, "after": rng_state()}
                outputs[name] = cpu_tree(value)
                return value

            encoded = stage("vae.encode", 1, lambda: model.autoencoder.encode(cpu_tree(batch)))
            valid = batch["mask_dict"]["coords"][..., 0, 0].bool()
            stage("vae.decode.encoded", 2, lambda: model.autoencoder.decode(
                encoded["z_latent"], batch["coords_nm"][:, :, 1, :], valid,
                batch["is_peptide"], extra_features=cpu_tree(batch)))
            clean = stage("conditioning", 3, lambda: model.add_clean_samples(cpu_tree(batch)))
            # Persist all conditioning tensors without repeating the large input tensors in comparisons.
            outputs["conditioning"] = cpu_tree({key: clean[key] for key in ("x_1", "gt_cond_latents", "is_fix")})
            for mode in ("flex", "rigid", "rigid_receptor"):
                stage(f"masks.{mode}", 4, lambda mode=mode: flow_module.compute_masks_unified(cpu_tree(clean), mode))
            motif_batch = cpu_tree(clean)
            motif_positions = (batch["is_peptide"] == 1).nonzero(as_tuple=False)[:2]
            for row, column in motif_positions:
                motif_batch["ca_cond_mask"][row, column] = True
                motif_batch["latent_cond_mask"][row, column] = True
            stage("masks.rigid_with_peptide_motif", 4,
                  lambda: flow_module.compute_masks_unified(motif_batch, "rigid"))
            sampling_batch = cpu_tree(clean)
            sampling_state = model.fm.prepare_sampling_state(sampling_batch, motif_constraint_mode=model.motif_constraint_mode)
            outputs["sampling_state"] = cpu_tree(sampling_state)
            initial = stage("solver.initial_noise", 5, lambda: model.fm.initialize_sampling_state(sampling_batch, sampling_state))
            trunk_batch = cpu_tree(sampling_batch)
            trunk_batch.update(x_t=cpu_tree(initial), t={mode: torch.full((1,), 0.5) for mode in model.fm.data_modes}, mask=valid)
            raw = stage("trunk.raw", 6, lambda: model.call_nn(cpu_tree(trunk_batch), n_recycle=0))
            predicted_clean = stage("trunk.clean_prediction", 6, lambda: model.fm.nn_out_to_clean_sample_prediction(
                cpu_tree(trunk_batch), cpu_tree(raw)))
            trunk_batch["x_sc"] = cpu_tree(predicted_clean)
            stage("trunk.self_conditioned", 7, lambda: model.call_nn(cpu_tree(trunk_batch), n_recycle=0))
            stage("solver.schedule", 8, lambda: model.fm.sample_schedule(400, cfg.generation.model))
            solver_batch = cpu_tree(clean)
            native_step = model.fm.simulation_step
            observer = StepObserver(native_step, solver_batch, directory / "steps",
                                    reference=root / "original/steps" if implementation == "current" else None,
                                    comparisons=comparisons,
                                    progress=lambda count: event("SOLVER_PROGRESS", implementation=implementation, steps=count))
            # The observer delegates to this exact bound method, then clones its returned tensors.
            model.fm.simulation_step = observer
            try:
                final, extra = stage("solver.final", 9, lambda: model.fm.full_simulation(
                    batch=solver_batch,
                    predict_for_sampling=lambda data, mode: model.predict_for_sampling(data, mode, n_recycle=0),
                    nsteps=400, self_cond=True, sampling_model_args=cfg.generation.model,
                    device=torch.device("cpu"), save_trajectory_every=0, guidance_w=1.0, ag_ratio=0.0,
                    motif_constraint_mode=model.motif_constraint_mode))
            finally:
                del model.fm.simulation_step
            require(observer.steps == 400, f"Native solver executed {observer.steps} steps")
            require(observer.self_conditioned_steps == 399, "Native solver did not self-condition all later steps")
            outputs["solver.final"] = {"sample": cpu_tree(final),
                                       "masks": cpu_tree({key: extra[key] for key in
                                                          ("mask", "is_peptide", "is_receptor", "cond_mask", "noise_mask")})}
            for fixed in (False, True):
                stage(f"vae.decode.final.fixed_receptor={fixed}", 10, lambda fixed=fixed: model.autoencoder.decode(
                    final["local_latents"], final["bb_ca"], extra["mask"], batch["is_peptide"],
                    extra_features=cpu_tree(clean), force_fix_receptor=fixed))
            torch.save(outputs, directory / "outputs.pt")
            torch.save(random_states, directory / "rng_states.pt")
            if implementation == "current":
                comparisons.tree("forward", outputs, torch.load(root / "original/outputs.pt", map_location="cpu", weights_only=False))
                comparisons.tree("rng", random_states, torch.load(root / "original/rng_states.pt", map_location="cpu", weights_only=False))
            parameters["implementations"][implementation].update(
                native_solver_steps=observer.steps, self_conditioned_steps=observer.self_conditioned_steps)
            write_json(root / "parameters.json", parameters)
            event("IMPLEMENTATION_COMPLETE", implementation=implementation, native_solver_steps=observer.steps)
            del observer, native_step, model, actual_state, outputs, random_states
        require(comparisons.failures == 0, f"{comparisons.failures} comparison leaves failed")
        summary = {"result": "PASS", "comparisons": comparisons.count, "tensor_comparisons": comparisons.tensor_count,
                   "failed_comparisons": comparisons.failures, "max_abs_delta": comparisons.max_abs_delta,
                   "native_solver_steps_per_implementation": 400, "rtol": comparisons.rtol, "atol": comparisons.atol,
                   "elapsed_seconds": round(time.monotonic() - started, 4), "output_root": str(root)}
        write_json(root / "summary.json", summary)
        event("SUMMARY", **summary)
        return 0
    except Exception as exc:
        summary = {"result": "FAIL", "error": str(exc), "comparisons": comparisons.count,
                   "failed_comparisons": comparisons.failures, "traceback": traceback.format_exc()}
        write_json(root / "summary.json", summary)
        event("SUMMARY", **summary)
        return 1


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--original-root", required=True)
    parser.add_argument("--model-root", required=True)
    parser.add_argument("--fixture-config", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--threads", type=int, default=4)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
