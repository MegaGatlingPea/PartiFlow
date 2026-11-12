"""Small CPU and subprocess-contract checks for the training acceptance helper."""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


sys.path.insert(0, str(Path(__file__).resolve().parent))
from acceptance import training as harness


HAS_TORCH = importlib.util.find_spec("torch") is not None
HAS_LIGHTNING = HAS_TORCH and importlib.util.find_spec("lightning") is not None


class TrainingHarnessContractTests(unittest.TestCase):
    def test_matrix_separates_public_and_instrumented_cross_mode_restarts(self):
        cases = harness.case_matrix()
        self.assertEqual(len(cases), 15)
        self.assertEqual(len({item["name"] for item in cases}), 15)
        self.assertEqual(sum(item["source"] is None for item in cases), 3)
        for kind in ("public", "instrumented"):
            fm = [item for item in cases if item["kind"] == kind and item["stage"] == "fm" and item["source"]]
            self.assertEqual(len(fm), 4)
            self.assertEqual({(item["source"], item["mode"], item["weights"]) for item in fm}, {
                (f"public_fm_{source}_epoch", target, weights)
                for source, target in (("eager", "compiled"), ("compiled", "eager"))
                for weights in ("normal", "ema")
            })

    def test_runtime_output_is_rejected_inside_checkout(self):
        with tempfile.TemporaryDirectory() as directory:
            checkout = Path(directory) / "checkout"
            for root in (checkout, checkout / "runs", Path("/")):
                with self.subTest(root=root), self.assertRaises(harness.AcceptanceError):
                    harness.external_root(root, checkout)

    def test_dataset_requires_exact_counts_unique_ids_and_every_real_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            split = root / "split.csv"
            structure, processed = root / "real.pdb", root / "real.pt"
            structure.write_text("ATOM\n")
            processed.write_bytes(b"processed")
            rows = [f"sample{i},{'train' if i < 64 else 'val'},real.pdb,real.pt\n" for i in range(80)]
            header = "pdbid,split,structure_path,processed_path\n"
            split.write_text(header + "".join(rows))
            self.assertEqual(harness.check_dataset(split)["validation_samples"], 16)
            split.write_text(header + "".join(rows[:-1]))
            with self.assertRaises(harness.AcceptanceError):
                harness.check_dataset(split)
            split.write_text(header + "".join(rows[:-1] + [rows[-2]]))
            with self.assertRaises(harness.AcceptanceError):
                harness.check_dataset(split)
            split.write_text(header + "".join(rows))
            processed.unlink()
            with self.assertRaises(harness.AcceptanceError):
                harness.check_dataset(split)

    def test_instrumented_hydra_datamodule_only_adds_one_callback(self):
        class Base:
            def __init__(self, **kwargs):
                self.arguments = kwargs
                self.trainer = SimpleNamespace(callbacks=[])
                self.calls = []

            def setup(self, stage=None):
                self.calls.append(stage)

        module = SimpleNamespace(PepLightningDataModule=Base)
        sentinel = object()
        with patch.dict(sys.modules, {"partiflow.datasets.pdb_data": module}), patch.object(
            harness, "state_trace_callback", return_value=sentinel
        ) as factory:
            data = harness.instrumented_datamodule("/outside/trace", "fm", split_csv="actual.csv")
            data.setup("fit")
            data.setup("fit")
        self.assertEqual(data.arguments, {"split_csv": "actual.csv"})
        self.assertEqual(data.calls, ["fit", "fit"])
        self.assertEqual(data.trainer.callbacks, [sentinel])
        factory.assert_called_once_with("/outside/trace", "fm")

    def test_execute_invokes_public_cli_subprocess_with_separate_pythonpath(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            case = {"name": "public_fm_eager_epoch", "stage": "fm", "mode": "eager", "kind": "public",
                    "source": None, "config": str(root / "config.yaml"), "run_directory": str(root / "runs/case")}
            manifest = {"output_root": str(root), "checkout": str(root / "checkout")}
            with patch.object(harness.subprocess, "run", return_value=SimpleNamespace(returncode=0)) as run, patch.object(
                harness, "audit_case", return_value={"passed": True}
            ):
                result = harness.execute_case(manifest, case, python="/environment/bin/python")
            self.assertEqual(result, {"passed": True})
            args, kwargs = run.call_args
            self.assertEqual(args[0], ["/environment/bin/python", "-m", "partiflow", "train", "fm", "--config", case["config"]])
            self.assertEqual(kwargs["env"]["PYTHONPATH"].split(":" )[:2], [str(root / "checkout"), str(root / "checkout/tests")])
            self.assertFalse(kwargs["check"])
            record = json.loads((root / "commands" / (case["name"] + ".json")).read_text())
            self.assertEqual((record["status"], record["returncode"]), ("completed", 0))

    def test_failed_cli_is_never_reported_as_a_pass(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            case = {"name": "failed", "stage": "vae", "mode": "eager", "kind": "public", "source": None,
                    "config": "config.yaml", "run_directory": str(root / "runs/failed")}
            with patch.object(harness.subprocess, "run", return_value=SimpleNamespace(returncode=7)), patch.object(
                harness, "audit_case"
            ) as audit:
                with self.assertRaises(harness.AcceptanceError):
                    harness.execute_case({"output_root": str(root), "checkout": str(root / "checkout")}, case)
            audit.assert_not_called()


@unittest.skipUnless(HAS_TORCH, "PyTorch is required")
class TrainingHarnessStateTests(unittest.TestCase):
    def _pair(self):
        import torch

        parameter = torch.full((1_000_001,), 0.8)
        averaged_parameter = torch.full_like(parameter, 0.7)
        raw = {"state": {0: {"step": torch.tensor(64.0), "exp_avg": torch.full_like(parameter, 0.2),
                             "exp_avg_sq": torch.full_like(parameter, 0.04)}},
               "param_groups": [{"params": [0], "lr": 0.01, "betas": (0.9, 0.999), "eps": 1e-8}]}
        ordinary = {"state_dict": {"weight": parameter}, "optimizer_states": [raw], "epoch": 0,
                    "global_step": 64, "nsamples_processed": 64, "lr_schedulers": []}
        averaged = copy.deepcopy(ordinary)
        averaged["state_dict"] = {"weight": averaged_parameter}
        averaged["optimizer_states"] = [{"opt": copy.deepcopy(raw), "ema": (averaged_parameter.clone(),),
                                           "current_step": 64, "decay": 0.999, "every_n_steps": 1}]
        return ordinary, averaged

    def test_pair_audits_all_moments_and_ema_values(self):
        ordinary, averaged = self._pair()
        with patch.object(harness, "load_checkpoint", side_effect=[ordinary, averaged]):
            result = harness.audit_checkpoint_pair("last.ckpt", "vae", step=64, epoch=0)
        self.assertEqual(result["moments"]["exp_avg_sq"], 1_000_001)
        for mutation in ("last_moment", "last_ema", "group_option", "counter", "nonfinite"):
            broken = copy.deepcopy(averaged)
            state = broken["optimizer_states"][0]
            if mutation == "last_moment":
                state["opt"]["state"][0]["exp_avg_sq"][-1] += 1
            elif mutation == "last_ema":
                state["ema"][0][-1] += 1
            elif mutation == "group_option":
                state["opt"]["param_groups"][0]["eps"] = 0.1
            elif mutation == "counter":
                state["current_step"] = 63
            else:
                state["opt"]["state"][0]["exp_avg"][-1] = float("nan")
            with self.subTest(mutation=mutation), patch.object(
                harness, "load_checkpoint", side_effect=[ordinary, broken]
            ), self.assertRaises(harness.AcceptanceError):
                harness.audit_checkpoint_pair("last.ckpt", "vae", step=64, epoch=0)

    def test_preupdate_comparison_checks_model_groups_moments_ema_scheduler_and_counters(self):
        import torch

        ordinary, averaged = self._pair()
        ordinary["loops"] = {"fit_loop": {"epoch_loop.batch_progress": {"is_last_batch": True},
                                              "epoch_progress": {"current": {"ready": 1, "processed": 1, "completed": 0}}}}
        ordinary["lr_schedulers"] = [{"last_epoch": 64, "_last_lr": [0.001]}]
        actual = {"model": copy.deepcopy(ordinary["state_dict"]), "optimizers": copy.deepcopy(averaged["optimizer_states"]),
                  "schedulers": copy.deepcopy(ordinary["lr_schedulers"]), "epoch": 1, "global_step": 64, "samples": 64}
        with patch.object(harness, "load_checkpoint", side_effect=[ordinary, averaged]):
            harness.restored_state_check(actual, "last.ckpt")
        for component in ("model", "groups", "moments", "ema", "scheduler", "epoch", "global_step", "samples"):
            broken = copy.deepcopy(actual)
            if component == "model":
                broken["model"]["weight"][-1] += 1
            elif component == "groups":
                broken["optimizers"][0]["opt"]["param_groups"][0]["lr"] = 0.5
            elif component == "moments":
                broken["optimizers"][0]["opt"]["state"][0]["exp_avg_sq"][-1] += 1
            elif component == "ema":
                broken["optimizers"][0]["ema"][0][-1] += 1
            elif component == "scheduler":
                broken["schedulers"][0]["last_epoch"] = 0
            else:
                broken[component] = -1
            with self.subTest(component=component), patch.object(
                harness, "load_checkpoint", side_effect=[ordinary, averaged]
            ), self.assertRaises(harness.AcceptanceError):
                harness.restored_state_check(broken, "last.ckpt")

    def test_finite_scan_rejects_last_tensor_and_partial_optimizer(self):
        import torch

        with self.assertRaises(harness.AcceptanceError):
            harness.finite_tree({"first": torch.ones(2), "last": torch.tensor([float("inf")])})
        with self.assertRaises(harness.AcceptanceError):
            harness.moment_inventory({"state": {}, "param_groups": [{"params": [0]}]})


@unittest.skipUnless(HAS_LIGHTNING, "Lightning is required")
class TrainingHarnessRuntimeTests(unittest.TestCase):
    def _model(self, stage):
        import lightning as L
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        class Model(L.LightningModule):
            def __init__(self):
                super().__init__()
                self.layer = torch.nn.Sequential(torch.nn.Linear(2, 3), torch.nn.Tanh(), torch.nn.Linear(3, 1))
                self.autoencoder = torch.nn.Linear(2, 2)
                self.autoencoder.requires_grad_(False)
                self.nsamples_processed = 0
                self.validation_output = []
                self.validation_output_data = []

            def training_step(self, batch, batch_idx):
                self.nsamples_processed += len(batch[0])
                loss = (self.layer(batch[0]) - batch[1]).square().mean()
                if self.logger:
                    self.log("train_loss", loss, on_step=True, on_epoch=False)
                    self.log("train/loss", loss, on_step=True, on_epoch=True)
                return loss

            def validation_step(self, batch, batch_idx):
                loss = (self.layer(batch[0]) - batch[1]).square().mean()
                self.validation_output.append(float(loss))
                self.validation_output_data.append(float(loss))
                if self.logger:
                    self.log("validation_loss/loss", loss, on_step=True, on_epoch=True)
                return loss

            def configure_optimizers(self):
                optimizer_type = torch.optim.Adam if stage == "fm" else torch.optim.AdamW
                optimizer = optimizer_type(self.layer.parameters(), lr=0.03, betas=(0.7, 0.8),
                                           eps=1e-6, weight_decay=0.02, amsgrad=True)
                scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.8)
                return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}

            def train_dataloader(self):
                return DataLoader(TensorDataset(torch.arange(8.0).reshape(4, 2) / 4, torch.tensor([[0.2], [0.5], [-0.1], [0.9]])), batch_size=1)

            def val_dataloader(self):
                return DataLoader(TensorDataset(torch.tensor([[0.3, 0.4], [0.6, 0.8]]), torch.tensor([[0.2], [0.7]])), batch_size=1)

            def on_save_checkpoint(self, checkpoint):
                checkpoint["nsamples_processed"] = self.nsamples_processed

            def on_load_checkpoint(self, checkpoint):
                self.nsamples_processed = checkpoint["nsamples_processed"]

        return Model()

    def _trainer(self, root, *, steps=-1, callbacks=(), logger=False):
        import lightning as L
        from partiflow.training.loops import install_epoch_boundary_loop
        from partiflow.utils.ema_callback import EMA, EmaModelCheckpoint

        trainer = L.Trainer(accelerator="cpu", devices=1, logger=logger, enable_progress_bar=False,
                            enable_model_summary=False, max_epochs=1 if steps < 0 else 2, max_steps=steps,
                            num_sanity_val_steps=0, gradient_clip_algorithm="norm", gradient_clip_val=1.0, log_every_n_steps=1,
                            callbacks=[EMA(decay=0.5), *callbacks, EmaModelCheckpoint(
                                dirpath=root, save_last=True, save_top_k=-1, every_n_epochs=1)])
        install_epoch_boundary_loop(trainer)
        return trainer

    def test_real_lightning_csv_and_saved_validation_counter_match_auditor(self):
        from lightning.pytorch.loggers import CSVLogger

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trainer = self._trainer(root / "checkpoints", logger=CSVLogger(str(root), name="metrics", version=""))
            trainer.fit(self._model("vae"))
            metrics = root / "metrics/metrics.csv"
            self.assertEqual(len(harness.csv_loss_values(metrics, ("train_loss", "train/loss_step"))), 4)
            self.assertEqual(len(harness.csv_loss_values(metrics, ("validation_loss/loss_step",))), 2)
            self.assertEqual(len(harness.csv_loss_values(metrics, ("validation_loss/loss_epoch",))), 1)
            checkpoint = harness.load_checkpoint(root / "checkpoints/last.ckpt")
            self.assertEqual(checkpoint["loops"]["fit_loop"]["epoch_loop.val_loop.batch_progress"]["total"]["completed"], 2)
            self.assertEqual(harness.expected_epoch(checkpoint), 1)

    def test_ordinary_and_direct_ema_restart_compare_every_state_before_real_update(self):
        import lightning as L

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for stage in ("vae", "fm"):
                L.seed_everything(29)
                initial = self._trainer(root / stage / "initial")
                initial.fit(self._model(stage))
                for weights in ("last.ckpt", "last-EMA.ckpt"):
                    with self.subTest(stage=stage, weights=weights):
                        trace = harness.state_trace_callback(root / stage / weights, stage)
                        resumed = self._trainer(root / stage / (weights + "-run"), steps=5, callbacks=[trace])
                        resumed.fit(self._model(stage), ckpt_path=str(root / stage / "initial" / weights))
                        result = trace.result
                        self.assertTrue(result["completed"])
                        self.assertEqual(result["restored_before_optimizer_step"]["global_step"], 4)
                        self.assertEqual(result["restored_before_optimizer_step"]["epoch"], 1)
                        self.assertTrue(result["first_update"]["reference_matches"])
                        self.assertEqual(result["first_update"]["changes"]["parameters_checked"], 4)
                        self.assertEqual(result["first_update"]["changes"]["max_exp_avg_sq_checked"], 4)
                        self.assertEqual((result["end"]["global_step"], result["end"]["samples"]), (5, 5))
                        if stage == "fm":
                            self.assertEqual(result["frozen_vae_comparison"]["tensors"], 2)


if __name__ == "__main__":
    unittest.main()
