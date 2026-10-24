"""CPU checks for completed and partial epoch continuation."""

from __future__ import annotations

import copy
import importlib.util
from pathlib import Path
import tempfile
import unittest


HAS_LIGHTNING = importlib.util.find_spec("lightning") is not None


@unittest.skipUnless(HAS_LIGHTNING, "Lightning is required")
class TrainingEpochResumeTests(unittest.TestCase):
    def _model(self):
        import lightning as L
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        class TinyModel(L.LightningModule):
            def __init__(self):
                super().__init__()
                self.layer = torch.nn.Linear(2, 1)
                self.training_batches = 0
                self.validation_batches = 0

            def training_step(self, batch, batch_idx):
                self.training_batches += 1
                x, y = batch
                return (self.layer(x) - y).square().mean()

            def validation_step(self, batch, batch_idx):
                self.validation_batches += 1
                x, y = batch
                return (self.layer(x) - y).square().mean()

            def configure_optimizers(self):
                optimizer = torch.optim.Adam(
                    self.parameters(), lr=0.05, betas=(0.7, 0.8), eps=1e-6
                )
                scheduler = torch.optim.lr_scheduler.StepLR(
                    optimizer, step_size=1, gamma=0.8
                )
                return {
                    "optimizer": optimizer,
                    "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
                }

            def train_dataloader(self):
                x = torch.arange(1.0, 9.0).reshape(4, 2) / 4
                y = torch.tensor([[0.2], [0.8], [-0.4], [0.5]])
                return DataLoader(TensorDataset(x, y), batch_size=1)

            def val_dataloader(self):
                x = torch.tensor([[0.3, 0.6], [0.9, 1.2]])
                y = torch.tensor([[0.1], [0.7]])
                return DataLoader(TensorDataset(x, y), batch_size=1)

            def on_save_checkpoint(self, checkpoint):
                checkpoint["training_batches"] = self.training_batches
                checkpoint["validation_batches"] = self.validation_batches

            def on_load_checkpoint(self, checkpoint):
                self.training_batches = checkpoint["training_batches"]
                self.validation_batches = checkpoint["validation_batches"]

        return TinyModel

    def _trainer(
        self,
        directory,
        *,
        max_epochs,
        max_steps,
        callbacks=(),
        adapted=True,
        step_checkpoint=False,
    ):
        import lightning as L
        from partiflow.training.loops import install_epoch_boundary_loop
        from partiflow.utils.ema_callback import EMA, EmaModelCheckpoint

        checkpoint = EmaModelCheckpoint(
            dirpath=str(directory),
            filename="step{step}" if step_checkpoint else "epoch{epoch}-step{step}",
            auto_insert_metric_name=False,
            save_top_k=-1,
            save_last=not step_checkpoint,
            **(
                {"every_n_train_steps": 2} if step_checkpoint else {"every_n_epochs": 1}
            ),
        )
        trainer = L.Trainer(
            accelerator="cpu",
            devices=1,
            max_epochs=max_epochs,
            max_steps=max_steps,
            logger=False,
            enable_progress_bar=False,
            enable_model_summary=False,
            num_sanity_val_steps=0,
            callbacks=[EMA(decay=0.5), *callbacks, checkpoint],
        )
        if adapted:
            install_epoch_boundary_loop(trainer)
        return trainer

    def _state(self, trainer, model):
        optimizer = trainer.optimizers[0]
        optimizer.join()
        return copy.deepcopy(
            {
                "weights": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": trainer.lr_scheduler_configs[0].scheduler.state_dict(),
                "training_batches": model.training_batches,
                "validation_batches": model.validation_batches,
                "step": trainer.global_step,
                "epoch": trainer.current_epoch,
                "loop": trainer.fit_loop.state_dict(),
                "restarting": trainer.fit_loop.restarting,
                "epoch_restarting": trainer.fit_loop.epoch_loop.restarting,
            }
        )

    def _capture(self):
        import lightning as L

        state = self._state

        class Capture(L.Callback):
            def __init__(self):
                self.start = None
                self.before_updates = []
                self.batch_starts = []
                self.batches = []

            def on_train_start(self, trainer, model):
                self.start = state(trainer, model)

            def on_train_batch_start(self, trainer, model, batch, batch_idx):
                self.batch_starts.append(
                    (trainer.current_epoch, trainer.global_step, batch_idx)
                )
                self.batches.append(copy.deepcopy(batch))

            def on_before_optimizer_step(self, trainer, model, optimizer):
                self.before_updates.append(state(trainer, model))

        return Capture()

    def _assert_equal(self, actual, expected, path="state"):
        import torch

        if isinstance(expected, torch.Tensor):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0, msg=path)
        elif isinstance(expected, dict):
            self.assertEqual(actual.keys(), expected.keys(), path)
            for key in expected:
                self._assert_equal(actual[key], expected[key], f"{path}.{key}")
        elif isinstance(expected, (tuple, list)):
            self.assertEqual(len(actual), len(expected), path)
            for index, (left, right) in enumerate(zip(actual, expected)):
                self._assert_equal(left, right, f"{path}[{index}]")
        else:
            self.assertEqual(actual, expected, path)

    def _assert_restored(self, captured, checkpoint, averaged):
        expected = {
            "weights": checkpoint["state_dict"],
            "optimizer": averaged["optimizer_states"][0],
            "scheduler": checkpoint["lr_schedulers"][0],
        }
        for snapshot in (captured.start, *captured.before_updates):
            for key, value in expected.items():
                self._assert_equal(snapshot[key], value, key)
            self.assertEqual(snapshot["step"], checkpoint["global_step"])
        self.assertEqual(
            captured.start["training_batches"], checkpoint["training_batches"]
        )
        self.assertEqual(
            captured.start["validation_batches"], checkpoint["validation_batches"]
        )

    def _assert_one_update(self, captured, final, checkpoint, averaged):
        import torch

        self.assertEqual(len(captured.before_updates), 1)
        self.assertEqual(len(captured.batches), 1)
        expected_model = self._model()()
        expected_model.load_state_dict(checkpoint["state_dict"])
        configured = expected_model.configure_optimizers()
        optimizer = configured["optimizer"]
        scheduler = configured["lr_scheduler"]["scheduler"]
        optimizer.load_state_dict(copy.deepcopy(averaged["optimizer_states"][0]["opt"]))
        scheduler.load_state_dict(copy.deepcopy(checkpoint["lr_schedulers"][0]))
        x, y = captured.batches[0]
        (expected_model.layer(x) - y).square().mean().backward()
        optimizer.step()
        scheduler.step()
        self._assert_equal(
            final["weights"], expected_model.state_dict(), "updated weights"
        )
        self._assert_equal(
            final["optimizer"]["opt"], optimizer.state_dict(), "updated Adam"
        )
        self._assert_equal(
            final["scheduler"], scheduler.state_dict(), "updated scheduler"
        )
        saved_optimizer = averaged["optimizer_states"][0]
        decay = saved_optimizer["decay"]
        expected_ema = tuple(
            old.mul(decay).add(new.detach(), alpha=1 - decay)
            for old, new in zip(saved_optimizer["ema"], expected_model.parameters())
        )
        self._assert_equal(final["optimizer"]["ema"], expected_ema, "updated EMA")
        self.assertEqual(
            final["optimizer"]["current_step"], checkpoint["global_step"] + 1
        )
        self.assertEqual(final["step"], checkpoint["global_step"] + 1)
        self.assertEqual(final["training_batches"], checkpoint["training_batches"] + 1)
        self.assertTrue(
            any(
                not torch.equal(value, checkpoint["state_dict"][name])
                for name, value in final["weights"].items()
            )
        )

    def _assert_epoch_counters(self, loop, count):
        expected = {
            key: count for key in ("ready", "started", "processed", "completed")
        }
        self._assert_equal(
            loop["epoch_progress"], {"total": expected, "current": expected}
        )

    def test_completed_epoch_normal_and_direct_ema_resume(self):
        import lightning as L
        import torch
        from partiflow.utils.ema_callback import EMAOptimizer

        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            L.seed_everything(19)
            initial = self._trainer(directory / "initial", max_epochs=1, max_steps=-1)
            initial.fit(self._model()())
            ordinary = torch.load(directory / "initial/last.ckpt", weights_only=False)
            averaged = torch.load(
                directory / "initial/last-EMA.ckpt", weights_only=False
            )
            self.assertEqual((ordinary["epoch"], ordinary["global_step"]), (0, 4))
            self.assertEqual(
                (ordinary["training_batches"], ordinary["validation_batches"]), (4, 2)
            )
            loop = ordinary["loops"]["fit_loop"]
            for scope in ("current", "total"):
                self._assert_equal(
                    loop["epoch_progress"][scope],
                    {"ready": 1, "started": 1, "processed": 1, "completed": 0},
                )
                self._assert_equal(
                    loop["epoch_loop.batch_progress"][scope],
                    {"ready": 4, "started": 4, "processed": 4, "completed": 4},
                )
                self._assert_equal(
                    loop["epoch_loop.val_loop.batch_progress"][scope],
                    {"ready": 2, "started": 2, "processed": 2, "completed": 2},
                )
            self.assertTrue(loop["epoch_loop.batch_progress"]["is_last_batch"])
            self._assert_equal(
                ordinary["optimizer_states"][0], averaged["optimizer_states"][0]["opt"]
            )
            self.assertTrue(
                any(
                    not torch.equal(value, averaged["state_dict"][name])
                    for name, value in ordinary["state_dict"].items()
                )
            )
            for filename, checkpoint in (
                ("last.ckpt", ordinary),
                ("last-EMA.ckpt", averaged),
            ):
                with self.subTest(checkpoint=filename):
                    captured = self._capture()
                    resume_name = (
                        "resumed-ema"
                        if filename.endswith("-EMA.ckpt")
                        else "resumed-normal"
                    )
                    trainer = self._trainer(
                        directory / resume_name,
                        max_epochs=2,
                        max_steps=5,
                        callbacks=[captured],
                    )
                    model = self._model()()
                    trainer.fit(model, ckpt_path=str(directory / "initial" / filename))
                    self._assert_restored(captured, checkpoint, averaged)
                    self.assertEqual(
                        (captured.start["epoch"], captured.start["step"]), (1, 4)
                    )
                    self.assertFalse(captured.start["restarting"])
                    self.assertFalse(captured.start["epoch_restarting"])
                    self.assertEqual(captured.batch_starts, [(1, 4, 0)])
                    self.assertIsInstance(trainer.optimizers[0], EMAOptimizer)
                    final = self._state(trainer, model)
                    self._assert_one_update(captured, final, checkpoint, averaged)
                    self.assertEqual((final["epoch"], final["step"]), (2, 5))
                    self._assert_epoch_counters(captured.start["loop"], 1)
                    self._assert_epoch_counters(final["loop"], 2)

    def test_partial_epoch_keeps_native_resume_and_next_update(self):
        import lightning as L
        import torch

        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            L.seed_everything(29)
            initial = self._trainer(
                directory / "initial", max_epochs=1, max_steps=-1, step_checkpoint=True
            )
            initial.fit(self._model()())
            path = directory / "initial/step2.ckpt"
            checkpoint = torch.load(path, weights_only=False)
            averaged = torch.load(
                directory / "initial/step2-EMA.ckpt", weights_only=False
            )
            self.assertEqual((checkpoint["epoch"], checkpoint["global_step"]), (0, 2))
            self.assertEqual(
                (checkpoint["training_batches"], checkpoint["validation_batches"]),
                (2, 0),
            )
            loop = checkpoint["loops"]["fit_loop"]
            self.assertFalse(loop["epoch_loop.batch_progress"]["is_last_batch"])
            for scope in ("current", "total"):
                self._assert_equal(
                    loop["epoch_progress"][scope],
                    {"ready": 1, "started": 1, "processed": 0, "completed": 0},
                )
                self._assert_equal(
                    loop["epoch_loop.batch_progress"][scope],
                    {"ready": 2, "started": 2, "processed": 2, "completed": 1},
                )
            results = []
            for adapted in (False, True):
                captured = self._capture()
                trainer = self._trainer(
                    directory / f"resumed-{adapted}",
                    max_epochs=2,
                    max_steps=3,
                    callbacks=[captured],
                    adapted=adapted,
                    step_checkpoint=True,
                )
                model = self._model()()
                trainer.fit(model, ckpt_path=str(path))
                self._assert_restored(captured, checkpoint, averaged)
                self.assertEqual(
                    (captured.start["epoch"], captured.start["step"]), (0, 2)
                )
                self.assertTrue(captured.start["restarting"])
                self.assertTrue(captured.start["epoch_restarting"])
                self.assertEqual(captured.batch_starts, [(0, 2, 2)])
                final = self._state(trainer, model)
                self._assert_one_update(captured, final, checkpoint, averaged)
                self.assertEqual((final["epoch"], final["step"]), (1, 3))
                self._assert_epoch_counters(final["loop"], 1)
                results.append((captured.start, final))
            self._assert_equal(
                results[1], results[0], "adapted and native continuation"
            )


if __name__ == "__main__":
    unittest.main()
