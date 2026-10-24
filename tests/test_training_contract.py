"""CPU contracts for configuration and paired optimizer/EMA continuation."""

from __future__ import annotations
from argparse import Namespace
import contextlib
import io
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

REPOSITORY = Path(__file__).resolve().parents[1]
HAS_HYDRA = importlib.util.find_spec("hydra") is not None
HAS_LIGHTNING = importlib.util.find_spec("lightning") is not None


class ResumeSelectionTests(unittest.TestCase):
    def test_auto_resume_selects_only_normal_checkpoint_versions(self):
        from partiflow.training.cli import resolve_resume

        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            for filename in (
                "last.ckpt",
                "last-v2.ckpt",
                "last-v3-EMA.ckpt",
                "not-last-v9.ckpt",
            ):
                (directory / filename).touch()
            self.assertEqual(
                resolve_resume({"resume": "auto"}, directory),
                str(directory / "last-v2.ckpt"),
            )
            self.assertIsNone(resolve_resume({"resume": "none"}, directory))

    def test_explicit_missing_resume_fails(self):
        from partiflow.training.cli import resolve_resume

        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                resolve_resume(
                    {"resume": str(Path(directory) / "missing.ckpt")}, Path(directory)
                )


@unittest.skipUnless(HAS_HYDRA, "Hydra is required")
class TrainingConfigurationTests(unittest.TestCase):
    def test_both_stages_compose_from_another_directory(self):
        import os
        from partiflow.training.config import load_config

        original = Path.cwd()
        try:
            with tempfile.TemporaryDirectory() as directory:
                os.chdir(directory)
                for stage, name in (("fm", "training"), ("vae", "training_ae")):
                    config = load_config(
                        REPOSITORY / "configs" / (name + ".yaml"), stage=stage
                    )
                    self.assertTrue(
                        Path(config.dataset.datamodule.split_csv).is_absolute()
                    )
                    self.assertEqual(config.dataset.datamodule.train_sampler, None)
                    self.assertTrue(
                        config.dataset.datamodule._target_.startswith("partiflow.")
                    )
                    self.assertEqual(
                        config.network.name,
                        "transformer_ae_large"
                        if stage == "vae"
                        else "transformer_fm_large",
                    )
        finally:
            os.chdir(original)

    def test_acceptance_configs_and_overrides(self):
        from partiflow.training.config import load_config

        for stage in ("fm", "vae"):
            accepted = load_config(
                REPOSITORY / "configs" / f"training_{stage}_acceptance.yaml",
                stage=stage,
            )
            self.assertEqual(accepted.opt.max_steps, -1)
            self.assertEqual(accepted.opt.max_epochs, 1)
            self.assertIsInstance(accepted.limit_train_batches, float)
            self.assertIsInstance(accepted.limit_val_batches, float)
            smoke = load_config(
                REPOSITORY / "configs" / f"training_{stage}_smoke.yaml", stage=stage
            )
            self.assertEqual(smoke.opt.max_steps, 2)
            self.assertIsInstance(smoke.limit_train_batches, int)
            config = load_config(
                REPOSITORY / "configs" / f"training_{stage}_acceptance.yaml",
                ["hardware.accelerator=cpu", "opt.max_steps=3"],
                stage=stage,
            )
            self.assertEqual(config.opt.max_steps, 3)
            self.assertEqual(config.hardware.accelerator, "cpu")
            self.assertTrue(
                config.dataset.datamodule.split_csv.endswith("acceptance.csv")
            )

    def test_external_legacy_groups_relocate(self):
        from partiflow.training.config import load_config

        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "custom.yaml"
            config_path.write_text(
                "defaults:\n  - network: transformer_ae_large\n  - _self_\ndataset:\n  datamodule:\n    _target_: src.datasets.pdb_data.PepLightningDataModule\n    split_csv: split.csv\n"
            )
            config = load_config(config_path, stage="vae")
            self.assertEqual(config.network.name, "transformer_ae_large")
            self.assertEqual(
                config.dataset.datamodule._target_,
                "partiflow.datasets.pdb_data.PepLightningDataModule",
            )
            self.assertEqual(
                config.dataset.datamodule.split_csv, str(Path(directory) / "split.csv")
            )

    def test_installed_resource_layout_keeps_assets_and_outputs_separate(self):
        import json
        import os
        import shutil
        import subprocess
        import sys

        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            package = directory / "site" / "partiflow"
            package.mkdir(parents=True)
            shutil.copy2(REPOSITORY / "partiflow/__init__.py", package / "__init__.py")
            shutil.copy2(REPOSITORY / "partiflow/paths.py", package / "paths.py")
            shutil.copytree(
                REPOSITORY / "partiflow/training",
                package / "training",
                ignore=shutil.ignore_patterns("__pycache__"),
            )
            shutil.copytree(REPOSITORY / "configs", package / "_resources/configs")
            invocation = directory / "invocation"
            invocation.mkdir()
            assets = directory / "model-assets"
            external = directory / "external" / "training.yaml"
            external.parent.mkdir()
            external.write_text(
                "dataset:\n  datamodule:\n    split_csv: data/split.csv\noutput_dir: output\nautoencoder_ckpt_path: custom.ckpt\n"
            )
            script = """import json
from pathlib import Path
from partiflow.paths import checkout_root, config_dir
from partiflow.training.config import load_config
bundled = load_config(config_dir() / 'training.yaml')
external = load_config(Path(__import__('sys').argv[1]))
print(json.dumps({'checkout': str(checkout_root()), 'config': str(config_dir()),
 'data': bundled.dataset.datamodule.split_csv, 'output': bundled.output_dir,
 'asset': bundled.autoencoder_ckpt_path, 'external_data': external.dataset.datamodule.split_csv,
 'external_output': external.output_dir, 'external_asset': external.autoencoder_ckpt_path}))
"""
            environment = dict(
                os.environ,
                PYTHONPATH=str(directory / "site"),
                PARTIFLOW_MODEL_DIR=str(assets),
            )
            completed = subprocess.run(
                [sys.executable, "-c", script, str(external)],
                cwd=invocation,
                env=environment,
                text=True,
                capture_output=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            result = json.loads(completed.stdout)
            self.assertEqual(result["checkout"], "None")
            self.assertEqual(result["config"], str(package / "_resources/configs"))
            self.assertEqual(
                result["data"],
                str(invocation / "train_data/pepinter/indexes/split.csv"),
            )
            self.assertEqual(result["output"], str(invocation / "runs"))
            self.assertEqual(result["asset"], str(assets / "ckpts/autoencoder.ckpt"))
            self.assertEqual(
                result["external_data"], str(external.parent / "data/split.csv")
            )
            self.assertEqual(result["external_output"], str(external.parent / "output"))
            self.assertEqual(result["external_asset"], str(assets / "custom.ckpt"))

    def test_public_show_config(self):
        from partiflow.training import run

        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertEqual(
                run(
                    Namespace(
                        stage="vae",
                        config=REPOSITORY / "configs/training_ae.yaml",
                        override=[],
                        print_config=True,
                    )
                ),
                0,
            )
        self.assertIn(
            "partiflow.datasets.pdb_data.PepLightningDataModule", output.getvalue()
        )


@unittest.skipUnless(HAS_LIGHTNING, "Lightning is required")
class TrainingResumeTests(unittest.TestCase):
    def _model(self):
        import lightning as L
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        class TinyModel(L.LightningModule):
            def __init__(self, cfg=None, store_dir=None):
                super().__init__()
                self.layer = torch.nn.Linear(1, 1)
                self.sample_count = 0

            def training_step(self, batch, batch_idx):
                x, y = batch
                self.sample_count += len(x)
                loss = (self.layer(x) - y).square().mean()
                self.log("train_loss", loss)
                return loss

            def configure_optimizers(self):
                optimizer = torch.optim.AdamW(self.parameters(), lr=0.05)
                warmup = torch.optim.lr_scheduler.LinearLR(
                    optimizer, start_factor=0.01, total_iters=2
                )
                cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=6)
                scheduler = torch.optim.lr_scheduler.SequentialLR(
                    optimizer, [warmup, cosine], milestones=[2]
                )
                return {
                    "optimizer": optimizer,
                    "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
                }

            def train_dataloader(self):
                return DataLoader(
                    TensorDataset(torch.arange(4.0).reshape(4, 1), torch.ones(4, 1)),
                    batch_size=2,
                )

            def on_save_checkpoint(self, checkpoint):
                checkpoint["sample_count"] = self.sample_count

            def on_load_checkpoint(self, checkpoint):
                self.sample_count = checkpoint["sample_count"]

        return TinyModel

    def _trainer(self, directory, steps, callbacks=()):
        import lightning as L
        from partiflow.utils.ema_callback import EMA, EmaModelCheckpoint

        return L.Trainer(
            accelerator="cpu",
            devices=1,
            max_epochs=3,
            max_steps=steps,
            logger=False,
            enable_progress_bar=False,
            enable_model_summary=False,
            callbacks=[
                EMA(decay=0.5),
                *callbacks,
                EmaModelCheckpoint(
                    dirpath=str(directory),
                    filename="step{step}",
                    every_n_train_steps=1,
                    save_top_k=-1,
                    save_last=True,
                ),
            ],
        )

    def test_normal_pair_and_direct_ema_restore_full_training_state(self):
        import lightning as L
        import torch
        from partiflow.utils.ema_callback import EMAOptimizer

        TinyModel = self._model()
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            L.seed_everything(19)
            trainer = self._trainer(directory, 2)
            trainer.fit(TinyModel())
            trainer.checkpoint_callback._save_checkpoint(
                trainer, str(directory / "last.ckpt")
            )
            regular = torch.load(directory / "last.ckpt", weights_only=False)
            averaged = torch.load(directory / "last-EMA.ckpt", weights_only=False)
            self.assertIn("param_groups", regular["optimizer_states"][0])
            self.assertNotIn("ema", regular["optimizer_states"][0])
            self.assertEqual(averaged["optimizer_states"][0]["current_step"], 2)
            self.assertEqual(regular["lr_schedulers"][0]["last_epoch"], 2)
            self.assertTrue(
                any(
                    not torch.equal(
                        regular["state_dict"][key], averaged["state_dict"][key]
                    )
                    for key in regular["state_dict"]
                )
            )
            for filename in ("last.ckpt", "last-EMA.ckpt"):
                captured = {}

                class Capture(L.Callback):
                    def on_train_start(self, trainer, model):
                        optimizer = trainer.optimizers[0]
                        captured.update(
                            step=trainer.global_step,
                            epoch=trainer.current_epoch,
                            ema_step=optimizer.current_step,
                            scheduler_epoch=trainer.lr_scheduler_configs[
                                0
                            ].scheduler.last_epoch,
                            samples=model.sample_count,
                            ema=tuple(t.clone() for t in optimizer.ema_params),
                            weights={
                                key: value.clone()
                                for key, value in model.state_dict().items()
                            },
                            learning_rate=optimizer.param_groups[0]["lr"],
                            moment=next(iter(optimizer.optimizer.state.values()))[
                                "exp_avg"
                            ].clone(),
                            optimizer_step=next(
                                iter(optimizer.optimizer.state.values())
                            )["step"].item(),
                        )

                resumed = TinyModel()
                resumed_trainer = self._trainer(
                    directory / filename.replace(".ckpt", "_resumed"), 3, [Capture()]
                )
                resumed_trainer.fit(resumed, ckpt_path=str(directory / filename))
                self.assertEqual(captured["step"], 2)
                self.assertEqual(captured["epoch"], regular["epoch"])
                self.assertEqual(captured["ema_step"], 2)
                self.assertEqual(captured["scheduler_epoch"], 2)
                self.assertEqual(captured["samples"], 4)
                self.assertEqual(captured["optimizer_step"], 2)
                self.assertEqual(
                    captured["learning_rate"],
                    regular["optimizer_states"][0]["param_groups"][0]["lr"],
                )
                torch.testing.assert_close(
                    captured["moment"],
                    next(iter(regular["optimizer_states"][0]["state"].values()))[
                        "exp_avg"
                    ],
                )
                restored_weights = (
                    averaged["state_dict"]
                    if filename.endswith("-EMA.ckpt")
                    else regular["state_dict"]
                )
                for key, tensor in captured["weights"].items():
                    torch.testing.assert_close(tensor, restored_weights[key])
                self.assertEqual(resumed.sample_count, 6)
                self.assertIsInstance(resumed_trainer.optimizers[0], EMAOptimizer)
                self.assertEqual(resumed_trainer.optimizers[0].current_step, 3)
                self.assertEqual(
                    resumed_trainer.lr_scheduler_configs[0].scheduler.last_epoch, 3
                )
                self.assertEqual(resumed_trainer.global_step, 3)
                for left, right in zip(
                    captured["ema"], averaged["optimizer_states"][0]["ema"]
                ):
                    torch.testing.assert_close(left, right)

    def test_missing_pair_fails_explicitly(self):
        from lightning.pytorch.utilities.exceptions import MisconfigurationException

        TinyModel = self._model()
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            self._trainer(directory, 1).fit(TinyModel())
            (directory / "last-EMA.ckpt").unlink()
            with self.assertRaisesRegex(MisconfigurationException, "associated EMA"):
                self._trainer(directory / "resumed", 2).fit(
                    TinyModel(), ckpt_path=str(directory / "last.ckpt")
                )

    def test_warm_start_loads_only_model_weights(self):
        import torch
        from omegaconf import OmegaConf
        from partiflow.training.runner import get_model_n_ckpt_resume

        TinyModel = self._model()
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            original = TinyModel()
            self._trainer(directory / "initial", 1).fit(original)
            config = OmegaConf.create(
                {
                    "seed": 2026,
                    "resume": "none",
                    "pretrain_ckpt_path": str(directory / "initial/last.ckpt"),
                }
            )
            with patch("partiflow.training.runner.AutoEncoder", TinyModel):
                warm, resume = get_model_n_ckpt_resume(
                    config, str(directory / "fresh"), stage="vae"
                )
            self.assertIsNone(resume)
            self.assertEqual(warm.sample_count, 0)
            for key, value in original.state_dict().items():
                torch.testing.assert_close(value, warm.state_dict()[key])
            optimizer_config = warm.configure_optimizers()
            self.assertEqual(len(optimizer_config["optimizer"].state), 0)
            self.assertEqual(
                optimizer_config["lr_scheduler"]["scheduler"].last_epoch, 0
            )


if __name__ == "__main__":
    unittest.main()
