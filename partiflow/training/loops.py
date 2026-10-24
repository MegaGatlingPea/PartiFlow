"""Narrow compatibility for completed-epoch resumes with Lightning 2.5.5."""

from lightning.pytorch.loops.fit_loop import _FitLoop
from lightning.pytorch.loops.loop import _Loop


def _aligned(progress, *, completed_epoch=False):
    try:
        for scope in ("current", "total"):
            values = progress[scope]
            ready = values["ready"]
            if not (ready == values["started"] == values["processed"]):
                return False
            if ready != values["completed"] + int(completed_epoch):
                return False
        return True
    except (KeyError, TypeError):
        return False


class EpochBoundaryFitLoop(_FitLoop):
    def load_state_dict(self, state_dict, prefix=""):
        super().load_state_dict(state_dict, prefix)
        epoch = state_dict.get(prefix + "epoch_progress", {})
        batch = state_dict.get(prefix + "epoch_loop.batch_progress", {})
        if (
            self.max_steps >= 0
            and batch.get("is_last_batch") is True
            and _aligned(epoch, completed_epoch=True)
            and _aligned(batch)
        ):
            # Native reset completes this epoch. The 2.5.5 fit-loop setter would
            # incorrectly re-enable restart merely because max_steps is set.
            _Loop.restarting.fset(self, False)


def install_epoch_boundary_loop(trainer):
    previous = trainer.fit_loop
    loop = EpochBoundaryFitLoop(
        trainer, min_epochs=previous.min_epochs, max_epochs=previous.max_epochs
    )
    loop.epoch_loop = previous.epoch_loop
    loop._data_source = previous._data_source
    trainer.fit_loop = loop
