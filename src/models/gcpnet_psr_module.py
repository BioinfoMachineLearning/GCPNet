# -------------------------------------------------------------------------------------------------------------------------------------
# Following code curated for GCPNet (https://github.com/BioinfoMachineLearning/GCPNet):
# -------------------------------------------------------------------------------------------------------------------------------------

import os
import torch
import torchmetrics

import torch.nn as nn

from pytorch_lightning import LightningModule
from functools import partial
from typing import Any, List, Tuple
from torch_scatter import scatter
from omegaconf import DictConfig
from tqdm import tqdm

from src.datamodules.components.atom3d_dataset import NUM_ATOM_TYPES
from src.models import HALT_FILE_EXTENSION, get_psr_corr_coefs
from src.models.components import centralize, localize
from src.models.components.gcpnet import GCPEmbedding, GCPLayerNorm, ScalarVector

from torchtyping import patch_typeguard
from typeguard import typechecked

patch_typeguard()  # use before @typechecked


class GCPNetPSRLitModule(LightningModule):
    """LightningModule for protein structure ranking (PSR) using GCPNet.

    This LightningModule organizes the PyTorch code into 6 sections:
        - Computations (init)
        - Train loop (training_step)
        - Validation loop (validation_step)
        - Test loop (test_step)
        - Prediction loop (predict_step)
        - Optimizers and LR schedulers (configure_optimizers)
        - End of model training (on_fit_end)
    """

    def __init__(
        self,
        layer_class: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        model_cfg: DictConfig,
        module_cfg: DictConfig,
        layer_cfg: DictConfig,
        path_cfg: DictConfig = None,
        num_atom_types: int = NUM_ATOM_TYPES,
        **kwargs
    ):
        super().__init__()

        # this line allows to access init params with `self.hparams` attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False, ignore=["layer_class"])

        # feature dimensionalities
        edge_input_dims = ScalarVector(model_cfg.e_input_dim, model_cfg.xi_input_dim)
        node_input_dims = ScalarVector(num_atom_types, model_cfg.chi_input_dim)
        self.edge_dims = ScalarVector(model_cfg.e_hidden_dim, model_cfg.xi_hidden_dim)
        self.node_dims = ScalarVector(model_cfg.h_hidden_dim, model_cfg.chi_hidden_dim)

        # PyTorch modules #

        # input embeddings
        self.gcp_embedding = GCPEmbedding(
            edge_input_dims,
            node_input_dims,
            self.edge_dims,
            self.node_dims,
            num_atom_types=num_atom_types,
            cfg=module_cfg
        )

        # message-passing layers
        self.interaction_layers = nn.ModuleList(
            layer_class(
                self.node_dims,
                self.edge_dims,
                cfg=module_cfg,
                layer_cfg=layer_cfg,
                dropout=model_cfg.dropout
            ) for _ in range(model_cfg.num_encoder_layers)
        )

        # predictions
        self.invariant_node_projection = nn.ModuleList([
            GCPLayerNorm(self.node_dims),
            module_cfg.selected_GCP(
                self.node_dims,
                (self.node_dims.scalar, 0),
                nonlinearities=tuple(module_cfg.nonlinearities),
                scalar_gate=module_cfg.scalar_gate,
                vector_gate=module_cfg.vector_gate,
                frame_gate=module_cfg.frame_gate,
                sigma_frame_gate=module_cfg.sigma_frame_gate,
                vector_frame_residual=module_cfg.vector_frame_residual,
                ablate_frame_updates=module_cfg.ablate_frame_updates,
                node_inputs=True
            )
        ])
        self.dense = nn.Sequential(
            nn.Linear(self.node_dims.scalar, self.node_dims.scalar * model_cfg.output_scale_factor),
            nn.ReLU(inplace=True),
            nn.Dropout(model_cfg.dense_dropout),
            nn.Linear(self.node_dims.scalar * model_cfg.output_scale_factor, model_cfg.output_dim)
        )

        # loss function and metrics #
        self.criterion = torch.nn.MSELoss()
        # note: for averaging loss across batches
        self.train_loss = torchmetrics.MeanMetric()
        self.val_loss = torchmetrics.MeanMetric()
        self.test_loss = torchmetrics.MeanMetric()

        # use separate metrics instances for the steps
        # of each phase (e.g., train, val and test)
        # to ensure a proper reduction over the epoch
        self.train_phase, self.val_phase, self.test_phase = "train", "val", "test"
        phases = [self.train_phase, self.val_phase, self.test_phase]

        metrics = {
            "RMSE": partial(torchmetrics.regression.mse.MeanSquaredError, squared=False),
            "GlobalPearsonCorrCoef": partial(torchmetrics.PearsonCorrCoef)
        }

        for phase in phases:
            for k, v in metrics.items():
                if phase != self.test_phase:
                    # create metrics for training and validation only
                    setattr(self, f"{phase}_{k}_metric", v())
        self.metrics = {
            phase: nn.ModuleDict(
                {
                    k: getattr(self, f"{phase}_{k}_metric") for k, _ in metrics.items()
                    if phase != self.test_phase
                }
            ) for phase in phases
        }

        # note: for logging best-so-far validation metrics
        self.val_rmse_best = torchmetrics.MinMetric()
        self.val_global_pearson_corr_coef_best = torchmetrics.MaxMetric()

    def get_ids(self, batch: Any) -> List[int]:
        if type(batch) in [list, tuple]:
            return [id for id in batch[0].id]
        return [id for id in batch.id]

    @staticmethod
    def get_labels(batch: Any) -> Any:
        if type(batch) in [list, tuple]:
            return batch[0].label
        return batch.label

    @typechecked
    def forward(self, batch: Any) -> Tuple[Any, torch.Tensor]:
        # centralize node positions to make them translation-invariant
        _, batch.x = centralize(batch, key="x", batch_index=batch.batch)

        # craft complete local frames corresponding to each edge
        batch.f_ij = localize(batch.x, batch.edge_index, norm_x_diff=self.hparams.module_cfg.norm_x_diff)

        # embed node and edge input features
        (h, chi), (e, xi) = self.gcp_embedding(batch)

        # update graph features using a series of geometric message-passing layers
        for layer in self.interaction_layers:
            (h, chi) = layer(
                (h, chi), (e, xi),
                batch.edge_index, batch.f_ij
            )

        # record final version of each feature in `Batch` object
        batch.h, batch.chi, batch.e, batch.xi = h, chi, e, xi

        # summarize intermediate node representations as final predictions
        out = self.invariant_node_projection[0]((h, chi))  # e.g., GCPLayerNorm()
        out = self.invariant_node_projection[1](
            out,
            batch.edge_index,
            batch.f_ij,
            node_inputs=True
        )  # e.g., GCP((h, chi)) -> h'
        out = scatter(out, batch.batch, dim=0, reduce="mean")
        out = self.dense(out).squeeze()

        return batch, out

    def step(self, batch: Any) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # make a forward pass and score it
        labels = self.get_labels(batch)
        _, preds = self.forward(batch)
        loss = self.criterion(preds, labels)
        return loss, preds, labels

    def on_train_start(self):
        # by default lightning executes validation step sanity checks before training starts,
        # so we need to make sure val_`metric`_best doesn't store any values from these checks
        self.val_loss.reset()
        self.metrics[self.val_phase]["RMSE"].reset()
        self.metrics[self.val_phase]["GlobalPearsonCorrCoef"].reset()
        self.val_rmse_best.reset()
        self.val_global_pearson_corr_coef_best.reset()

    def training_step(self, batch: Any, batch_idx: int):
        loss, preds, labels = self.step(batch)

        # update metrics
        self.train_loss(loss)
        for metric in self.metrics[self.train_phase].keys():
            preds = preds.detach()
            self.metrics[self.train_phase][metric](preds, labels)

        return {"loss": loss, "preds": preds, "labels": labels}

    def training_epoch_end(self, outputs: List[Any]):
        # log metrics
        self.log(f"{self.train_phase}/loss", self.train_loss, prog_bar=False)
        for metric in self.metrics[self.train_phase].keys():
            self.log(
                f"{self.train_phase}/" + metric,
                self.metrics[self.train_phase][metric],
                metric_attribute=self.metrics[self.train_phase][metric],
                prog_bar=True
            )

    def validation_step(self, batch: Any, batch_idx: int):
        loss, preds, labels = self.step(batch)

        # update metrics
        self.val_loss(loss)
        for metric in self.metrics[self.val_phase].keys():
            preds = preds.detach()
            self.metrics[self.val_phase][metric](preds, labels)

        return {"loss": loss, "preds": preds, "labels": labels}

    def validation_epoch_end(self, outputs: List[Any]):
        # update best-so-far validation metrics according to the current epoch's results
        self.val_rmse_best.update(self.metrics[self.val_phase]["RMSE"].compute())
        self.val_global_pearson_corr_coef_best.update(self.metrics[self.val_phase]["GlobalPearsonCorrCoef"].compute())

        # log metrics
        self.log(f"{self.val_phase}/loss", self.val_loss, prog_bar=True)
        for metric in self.metrics[self.val_phase].keys():
            self.log(
                f"{self.val_phase}/" + metric,
                self.metrics[self.val_phase][metric],
                metric_attribute=self.metrics[self.val_phase][metric],
                prog_bar=True
            )

        # log best-so-far metrics as a value through `.compute()` method, instead of as a metric object;
        # otherwise, metric would be reset by Lightning after each epoch
        # note: when logging as a value, set `sync_dist=True` for proper reduction over processes in DDP mode
        self.log(
            f"{self.val_phase}/RMSE_best",
            self.val_rmse_best.compute(),
            prog_bar=True,
            sync_dist=True
        )
        self.log(
            f"{self.val_phase}/GlobalPearsonCorrCoef_best",
            self.val_global_pearson_corr_coef_best.compute(),
            prog_bar=True,
            sync_dist=True
        )

    def test_step(self, batch: Any, batch_idx: int):
        loss, preds, labels = self.step(batch)
        ids = self.get_ids(batch)

        # update loss
        self.test_loss(loss)

        return {"loss": loss, "preds": preds, "labels": labels, "ids": ids}

    def test_epoch_end(self, outputs: List[Any]):
        # log loss
        self.log(f"{self.test_phase}/loss", self.test_loss, prog_bar=False)

        # derive local and global PSR metrics
        corr_coefs = get_psr_corr_coefs()

        # collect all labels, predictions, and target IDs as CPU-bound lists of floats
        labels, preds, ids = [], [], []
        for output in tqdm(outputs):
            labels.extend(list(output["labels"].cpu().numpy()))
            preds.extend(list(output["preds"].cpu().numpy()))
            ids.extend(output["ids"])

        # log local and global PSR metrics
        for name, func in corr_coefs.items():
            func = partial(func, ids=ids)
            value = func(labels, preds)
            self.log(f"{self.test_phase}/{name}", value, prog_bar=False)

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.

        Examples:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        optimizer = self.hparams.optimizer(params=self.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}

    def on_fit_end(self):
        """Lightning calls this upon completion of the user's call to `trainer.fit()` for model training.
        For example, Lightning will call this hook upon exceeding `trainer.max_epochs` in model training.
        """
        if self.trainer.is_global_zero:
            path_cfg = self.hparams.path_cfg
            if path_cfg is not None and path_cfg.grid_search_script_dir is not None:
                # uniquely record when model training is concluded
                grid_search_script_dir = self.hparams.path_cfg.grid_search_script_dir
                run_id = self.logger.experiment.id
                fit_end_indicator_filename = f"{run_id}.{HALT_FILE_EXTENSION}"
                fit_end_indicator_filepath = os.path.join(grid_search_script_dir, fit_end_indicator_filename)
                with open(fit_end_indicator_filepath, "w") as f:
                    f.write("`on_fit_end` has been called.")
        return super().on_fit_end()


if __name__ == "__main__":
    import hydra
    import omegaconf
    import pyrootutils

    root = pyrootutils.setup_root(__file__, pythonpath=True)
    cfg = omegaconf.OmegaConf.load(root / "configs" / "model" / "gcpnet_psr.yaml")
    _ = hydra.utils.instantiate(cfg)
