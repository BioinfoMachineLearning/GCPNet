# -------------------------------------------------------------------------------------------------------------------------------------
# Following code curated for GCPNet (https://github.com/BioinfoMachineLearning/GCPNet):
# -------------------------------------------------------------------------------------------------------------------------------------

import os
import pytorch_lightning as pl

from pytorch_lightning.utilities.types import STEP_OUTPUT
from typing import Any

from src.utils.pylogger import get_pylogger
from src.utils.rich_utils import enforce_tags, print_config_tree
from src.utils.utils import (
    close_loggers,
    extras,
    get_metric_value,
    instantiate_callbacks,
    instantiate_loggers,
    log_hyperparameters,
    save_file,
    task_wrapper,
)


class NStepModelCheckpoint(pl.Callback):
    """
    Save a checkpoint every `N` steps, in contrast to Lightning's default
    callback that checkpoints based on a metric such as e.g., validation loss.
    """

    def __init__(
        self,
        save_step_frequency: int,
        prefix: str = "N_step_checkpoint",
        use_modelcheckpoint_filename: bool = False
    ):
        """
        Args:
            save_step_frequency: How often to save in terms of number of steps.
            prefix: Prefix to add to the checkpoint names when `use_modelcheckpoint_filename=False`.
            use_modelcheckpoint_filename: Whether to use the `ModelCheckpoint` callback's
                default filename instead of the one from this callback.
        """
        self.save_step_frequency = save_step_frequency
        self.prefix = prefix
        self.use_modelcheckpoint_filename = use_modelcheckpoint_filename

    def on_train_batch_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", outputs: STEP_OUTPUT, batch: Any, batch_idx: int
    ):
        """After each training batch, check whether to save a checkpoint based on a running step counter."""
        epoch = trainer.current_epoch
        global_step = trainer.global_step
        if global_step % self.save_step_frequency == 0:
            if self.use_modelcheckpoint_filename:
                filename = trainer.checkpoint_callback.filename
            else:
                filename = f"{self.prefix}_{epoch=}_{global_step=}.ckpt"
            ckpt_path = os.path.join(trainer.checkpoint_callback.dirpath, filename)
            trainer.save_checkpoint(ckpt_path)
