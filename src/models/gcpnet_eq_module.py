# -------------------------------------------------------------------------------------------------------------------------------------
# Following code curated for GCPNet (https://github.com/BioinfoMachineLearning/GCPNet):
# -------------------------------------------------------------------------------------------------------------------------------------

import os
import tempfile
import torch
import torchmetrics

import numpy as np
import pandas as pd
import torch.nn as nn

from datetime import datetime
from pathlib import Path
from pytorch_lightning import LightningModule
from functools import partial
from typing import Any, Dict, List, Optional, Tuple, Union
from torch_scatter import scatter
from omegaconf import DictConfig

from src import utils
from src.datamodules.components.eq_dataset import ATOM_TYPES, MAX_PLDDT_VALUE
from src.models import HALT_FILE_EXTENSION, Queue, annotate_pdb_with_new_column_values, convert_idx_from_batch_local_to_global, get_grad_norm
from src.models.components import centralize, localize
from src.models.components.gcpnet import GCPEmbedding, GCPLayerNorm, ScalarVector

from torchtyping import patch_typeguard
from typeguard import typechecked

patch_typeguard()  # use before @typechecked


log = utils.get_pylogger(__name__)


class GCPNetEQLitModule(LightningModule):
    """LightningModule for equivariant quality assessment (EQ) of protein structures using GCPNet.

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
        num_atom_types: int = len(ATOM_TYPES),
        **kwargs
    ):
        super().__init__()

        # this line allows to access init params with `self.hparams` attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False, ignore=["layer_class"])

        # feature dimensionalities
        edge_input_dims = ScalarVector(model_cfg.e_input_dim, model_cfg.xi_input_dim)
        node_input_dims = ScalarVector(model_cfg.h_input_dim + num_atom_types, model_cfg.chi_input_dim)
        self.edge_dims = ScalarVector(model_cfg.e_hidden_dim, model_cfg.xi_hidden_dim)
        self.node_dims = ScalarVector(model_cfg.h_hidden_dim, model_cfg.chi_hidden_dim)

        # PyTorch modules #
        # input embeddings
        self.atom_embedding = nn.Embedding(num_atom_types, num_atom_types)
        self.gcp_embedding = GCPEmbedding(
            edge_input_dims,
            node_input_dims,
            self.edge_dims,
            self.node_dims,
            num_atom_types=0,
            nonlinearities=(module_cfg.scalar_nonlinearity, module_cfg.vector_nonlinearity),
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
                ablate_scalars=module_cfg.ablate_scalars,
                ablate_vectors=module_cfg.ablate_vectors,
                enable_e3_equivariance=module_cfg.enable_e3_equivariance,
                node_inputs=True
            )
        ])

        self.dense = nn.Sequential(
            nn.Linear(self.node_dims.scalar, self.node_dims.scalar * model_cfg.output_scale_factor),
            nn.ReLU(inplace=True),
            nn.Dropout(model_cfg.dense_dropout),
            nn.Linear(self.node_dims.scalar * model_cfg.output_scale_factor, model_cfg.output_dim)
        )

        # training #
        if self.hparams.module_cfg.clip_gradients:
            self.gradnorm_queue = Queue()
            self.gradnorm_queue.add(3000)  # add large value that will be flushed

        # loss function and metrics #
        self.criterion = torch.nn.SmoothL1Loss()
        # note: for averaging loss across batches
        self.train_loss = torchmetrics.MeanMetric()
        self.val_loss = torchmetrics.MeanMetric()
        self.test_loss = torchmetrics.MeanMetric()

        # use separate metrics instances for the steps
        # of each phase (e.g., train, val and test)
        # to ensure a proper reduction over the epoch
        self.train_phase, self.val_phase, self.test_phase, self.predict_phase = "train", "val", "test", "predict"
        phases = [self.train_phase, self.val_phase, self.test_phase]

        metrics = {
            "PerResidueMSE": partial(torchmetrics.regression.mse.MeanSquaredError),
            "PerResidueMAE": partial(torchmetrics.regression.mae.MeanAbsoluteError),
            "PerResiduePearsonCorrCoef": partial(torchmetrics.PearsonCorrCoef),
            "PerModelMSE": partial(torchmetrics.regression.mse.MeanSquaredError),
            "PerModelMAE": partial(torchmetrics.regression.mae.MeanAbsoluteError),
            "PerModelPearsonCorrCoef": partial(torchmetrics.PearsonCorrCoef)
        }

        for phase in phases:
            for k, v in metrics.items():
                setattr(self, f"{phase}_{k}_metric", v())
        self.metrics = {
            phase: nn.ModuleDict(
                {k: getattr(self, f"{phase}_{k}_metric") for k, _ in metrics.items()}
            ) for phase in phases
        }

        # note: for logging best-so-far validation metrics
        self.val_per_residue_mse_best = torchmetrics.MinMetric()
        self.val_per_residue_mae_best = torchmetrics.MinMetric()
        self.val_per_residue_pearson_corr_coef_best = torchmetrics.MaxMetric()
        self.val_per_model_mse_best = torchmetrics.MinMetric()
        self.val_per_model_mae_best = torchmetrics.MinMetric()
        self.val_per_model_pearson_corr_coef_best = torchmetrics.MaxMetric()

    @staticmethod
    def get_labels(batch: Any) -> Any:
        if type(batch) in [list, tuple]:
            return batch[0].label
        return batch.label

    @typechecked
    def forward(self, batch: Any) -> Tuple[Any, torch.Tensor]:
        # correct residue-wise graph metadata for batch context
        batch.ca_atom_idx, ca_atom_batch_index = convert_idx_from_batch_local_to_global(
            batch.ca_atom_idx, batch.batch, batch.num_graphs
        )
        batch.atom_residue_idx, _ = convert_idx_from_batch_local_to_global(
            batch.atom_residue_idx, ca_atom_batch_index, batch.num_graphs
        )

        # centralize node positions to make them translation-invariant
        _, batch.x = centralize(
            batch,
            key="x",
            batch_index=batch.batch,
            node_mask=batch.mask
        )

        # craft complete local frames corresponding to each edge
        batch.f_ij = localize(
            batch.x,
            batch.edge_index,
            norm_x_diff=self.hparams.module_cfg.norm_x_diff,
            node_mask=batch.mask
        )

        # embed node and edge input 
        batch.h = torch.cat((batch.h, self.atom_embedding(batch.atom_types)), dim=-1)
        (h, chi), (e, xi) = self.gcp_embedding(batch)

        # update graph features using a series of geometric message-passing layers
        for layer in self.interaction_layers:
            (h, chi) = layer(
                (h, chi),
                (e, xi),
                batch.edge_index,
                batch.f_ij,
                node_mask=batch.mask
            )

        # record final version of each feature in `Batch` object
        batch.h, batch.chi, batch.e, batch.xi = h, chi, e, xi

        # summarize intermediate node representations as final predictions
        out = self.invariant_node_projection[0]((h, chi))  # e.g., GCPLayerNorm()
        out = self.invariant_node_projection[1](
            out,
            batch.edge_index,
            batch.f_ij,
            node_inputs=True,
            node_mask=batch.mask
        )  # e.g., GCP((h, chi)) -> h'
        res_out = scatter(out[batch.mask], batch.atom_residue_idx[batch.mask], dim=0, reduce="mean")  # get batch-wise plDDT for each residue
        res_out = self.dense(res_out).squeeze()

        return batch, res_out

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
        self.metrics[self.val_phase]["PerResidueMSE"].reset()
        self.metrics[self.val_phase]["PerResidueMAE"].reset()
        self.metrics[self.val_phase]["PerResiduePearsonCorrCoef"].reset()
        self.metrics[self.val_phase]["PerModelMSE"].reset()
        self.metrics[self.val_phase]["PerModelMAE"].reset()
        self.metrics[self.val_phase]["PerModelPearsonCorrCoef"].reset()
        self.val_per_residue_mse_best.reset()
        self.val_per_residue_mae_best.reset()
        self.val_per_residue_pearson_corr_coef_best.reset()
        self.val_per_model_mse_best.reset()
        self.val_per_model_mae_best.reset()
        self.val_per_model_pearson_corr_coef_best.reset()

    def training_step(self, batch: Any, batch_idx: int):
        try:
            loss, preds, labels = self.step(batch)
        except RuntimeError as e:
            if "CUDA out of memory" not in str(e):
                raise(e)
            torch.cuda.empty_cache()
            log.info(f"Skipping batch with index {batch_idx} due to OOM error...")
            return

        # update metrics
        self.train_loss(loss.detach())

        # log per-residue metrics
        preds = preds.detach()
        for metric in self.metrics[self.train_phase].keys():
            if "residue" in metric.lower():
                self.metrics[self.train_phase][metric](preds, labels)

        # log per-model metrics
        ca_batch = scatter(batch.batch[batch.mask], batch.atom_residue_idx[batch.mask], dim=0, reduce="mean").long()  # get node-batch indices for Ca atoms
        preds_out = scatter(preds, ca_batch, dim=0, reduce="mean")  # get batch-wise global plDDT
        labels_out = scatter(labels, ca_batch, dim=0, reduce="mean")
        for metric in self.metrics[self.train_phase].keys():
            if "model" in metric.lower():
                self.metrics[self.train_phase][metric](preds_out, labels_out)

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
        try:
            loss, preds, labels = self.step(batch)
        except RuntimeError as e:
            if "CUDA out of memory" not in str(e):
                raise(e)
            torch.cuda.empty_cache()
            log.info(f"Skipping batch with index {batch_idx} due to OOM error...")
            return
        except AssertionError as e:
            if "The neighbor search missed some atoms" in str(e):
                torch.cuda.empty_cache()
                log.info(f"Skipping batch with index {batch_idx} due to missing atoms in ET neighborhood search...")
                return
            else:
                raise(e)

        # update metrics
        self.val_loss(loss.detach())

        # log per-residue metrics
        preds = preds.detach()
        for metric in self.metrics[self.val_phase].keys():
            if "residue" in metric.lower():
                self.metrics[self.val_phase][metric](preds, labels)

        # log per-model metrics
        ca_batch = scatter(batch.batch[batch.mask], batch.atom_residue_idx[batch.mask], dim=0, reduce="mean").long()  # get node-batch indices for Ca atoms
        preds_out = scatter(preds, ca_batch, dim=0, reduce="mean")  # get batch-wise global plDDT
        labels_out = scatter(labels, ca_batch, dim=0, reduce="mean")
        for metric in self.metrics[self.val_phase].keys():
            if "model" in metric.lower():
                self.metrics[self.val_phase][metric](preds_out, labels_out)

        return {"loss": loss, "preds": preds, "labels": labels}

    def validation_epoch_end(self, outputs: List[Any]):
        # update best-so-far validation metrics according to the current epoch's results
        self.val_per_residue_mse_best.update(self.metrics[self.val_phase]["PerResidueMSE"].compute())
        self.val_per_residue_mae_best.update(self.metrics[self.val_phase]["PerResidueMAE"].compute())
        self.val_per_residue_pearson_corr_coef_best.update(self.metrics[self.val_phase]["PerResiduePearsonCorrCoef"].compute())
        self.val_per_model_mse_best.update(self.metrics[self.val_phase]["PerModelMSE"].compute())
        self.val_per_model_mae_best.update(self.metrics[self.val_phase]["PerModelMAE"].compute())
        self.val_per_model_pearson_corr_coef_best.update(self.metrics[self.val_phase]["PerModelPearsonCorrCoef"].compute())

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
            f"{self.val_phase}/PerResidueMSE_best",
            self.val_per_residue_mse_best.compute(),
            prog_bar=True,
            sync_dist=True
        )
        self.log(
            f"{self.val_phase}/PerResidueMAE_best",
            self.val_per_residue_mae_best.compute(),
            prog_bar=True,
            sync_dist=True
        )
        self.log(
            f"{self.val_phase}/PerResiduePearsonCorrCoef_best",
            self.val_per_residue_pearson_corr_coef_best.compute(),
            prog_bar=True,
            sync_dist=True
        )
        self.log(
            f"{self.val_phase}/PerModelMSE_best",
            self.val_per_model_mse_best.compute(),
            prog_bar=True,
            sync_dist=True
        )
        self.log(
            f"{self.val_phase}/PerModelMAE_best",
            self.val_per_model_mae_best.compute(),
            prog_bar=True,
            sync_dist=True
        )
        self.log(
            f"{self.val_phase}/PerModelPearsonCorrCoef_best",
            self.val_per_model_pearson_corr_coef_best.compute(),
            prog_bar=True,
            sync_dist=True
        )

    def test_step(self, batch: Any, batch_idx: int):
        loss, preds, labels = self.step(batch)

        # update loss
        self.test_loss(loss.detach())

        # log per-residue metrics
        preds = preds.detach()
        for metric in self.metrics[self.test_phase].keys():
            if "residue" in metric.lower():
                self.metrics[self.test_phase][metric](preds, labels)

        # log per-model metrics
        ca_batch = scatter(batch.batch[batch.mask], batch.atom_residue_idx[batch.mask], dim=0, reduce="mean").long()  # get node-batch indices for Ca atoms
        preds_out = scatter(preds, ca_batch, dim=0, reduce="mean")  # get batch-wise global plDDT
        labels_out = scatter(labels, ca_batch, dim=0, reduce="mean")
        for metric in self.metrics[self.test_phase].keys():
            if "model" in metric.lower():
                self.metrics[self.test_phase][metric](preds_out, labels_out)

        return {"loss": loss, "preds": preds, "labels": labels}

    def test_epoch_end(self, outputs: List[Any]):
        # log metrics
        self.log(f"{self.test_phase}/loss", self.test_loss, prog_bar=False)
        for metric in self.metrics[self.test_phase].keys():
            self.log(
                f"{self.test_phase}/" + metric,
                self.metrics[self.test_phase][metric],
                metric_attribute=self.metrics[self.test_phase][metric],
                prog_bar=True
            )

    def on_predict_epoch_start(self):
        # configure loss function for inference to report loss values per batch element
        self.criterion = torch.nn.SmoothL1Loss(reduction='none')
        # define where the final predictions should be recorded
        self.predictions_csv_path = os.path.join(
            self.trainer.default_root_dir,
            f"{self.predict_phase}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_rank_{self.global_rank}_predictions.csv",
        )

    @torch.inference_mode()
    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0):
        batch.init_h = batch.h.clone()
        if hasattr(batch, "true_pdb_filepath") and all(batch.true_pdb_filepath):
            # note: currently, we can only score the loss for batches without any missing true (i.e., native) PDB files
            loss, preds, labels = self.step(batch)
        else:
            _, preds = self.forward(batch)
            loss, labels = None, None

        # collect per-model predictions
        batch.ca_batch = scatter(batch.batch[batch.mask], batch.atom_residue_idx[batch.mask], dim=0, reduce="mean").long()  # get node-batch indices for Ca atoms
        global_preds = scatter(preds, batch.ca_batch, dim=0, reduce="mean")  # get batch-wise global plDDT

        if loss is not None:
            # get batch-wise global plDDT loss
            loss = scatter(loss, batch.ca_batch, dim=0, reduce="mean")
            # get initial residue-wise plDDT values from AlphaFold
            batch.initial_res_scores = scatter(batch.init_h[:, -1][batch.mask], batch.atom_residue_idx[batch.mask], dim=0, reduce="mean")

        # collect outputs, and visualize predicted lDDT scores
        step_outputs = self.record_qa_preds(
            batch=batch,
            res_preds=preds,
            global_preds=global_preds,
            loss=loss,
            labels=labels
        )
        return step_outputs
    
    def on_predict_epoch_end(self, outputs: List[Any]):
        prediction_outputs = [
            output for output_ in outputs for output__ in output_ for output in output__
        ]
        # compile predictions collected by the current device (e.g., rank zero)
        predictions_csv_df = pd.DataFrame(prediction_outputs)
        predictions_csv_df.to_csv(self.predictions_csv_path, index=False)
    
    @torch.inference_mode()
    @typechecked
    def record_qa_preds(
        self,
        batch: Any,
        res_preds: torch.Tensor,
        global_preds: torch.Tensor,
        loss: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        plddt_scale_factor: float = MAX_PLDDT_VALUE
    ) -> List[Dict[str, Any]]:
        # create temporary output PDB files for predictions
        batch_metrics = []
        initial_res_scores = batch.initial_res_scores.detach().cpu().numpy()
        pred_res_scores = res_preds.detach().cpu().numpy()
        pred_global_scores = global_preds.detach().cpu().numpy()
        batch_loss = None if loss is None else loss.detach().cpu().numpy()
        batch_labels = None if labels is None else labels.detach().cpu().numpy()
        res_batch_index = batch.ca_batch.detach().cpu().numpy()
        for b_index in range(batch.num_graphs):
            metrics = {}
            temp_pdb_dir = tempfile._get_default_tempdir()
            temp_pdb_code = next(tempfile._get_candidate_names())
            initial_pdb_filepath = batch.decoy_pdb_filepath[b_index]
            prediction_path = str(temp_pdb_dir / Path(f"predicted_{temp_pdb_code}").with_suffix(".pdb"))
            true_path = str(temp_pdb_dir / Path(f"true_{temp_pdb_code}").with_suffix(".pdb"))
            # isolate each individual example within the current batch
            initial_res_scores_ = initial_res_scores[res_batch_index == b_index] * plddt_scale_factor
            pred_res_scores_ = pred_res_scores[res_batch_index == b_index] * plddt_scale_factor
            pred_global_score_ = pred_global_scores[b_index] * plddt_scale_factor
            loss_ = np.nan if batch_loss is None else batch_loss[b_index]
            labels_ = None if batch_labels is None else batch_labels[res_batch_index == b_index] * plddt_scale_factor
            annotate_pdb_with_new_column_values(
                input_pdb_filepath=initial_pdb_filepath,
                output_pdb_filepath=prediction_path,
                column_name="b_factor",
                new_column_values=pred_res_scores_
            )
            if labels_ is not None:
                annotate_pdb_with_new_column_values(
                    input_pdb_filepath=initial_pdb_filepath,
                    output_pdb_filepath=true_path,
                    column_name="b_factor",
                    new_column_values=labels_
                )
                initial_per_res_plddt_ae = (np.abs(initial_res_scores_ - labels_).mean() / plddt_scale_factor)
                pred_per_res_plddt_ae = (np.abs(pred_res_scores_ - labels_).mean() / plddt_scale_factor)
            else:
                true_path = None
                initial_per_res_plddt_ae = None
                pred_per_res_plddt_ae = None
            metrics["input_annotated_pdb_filepath"] = initial_pdb_filepath
            metrics["predicted_annotated_pdb_filepath"] = prediction_path
            metrics["true_annotated_pdb_filepath"] = true_path
            metrics["global_plddt"] = pred_global_score_
            metrics["plddt_loss"] = loss_
            metrics["input_per_residue_plddt_absolute_error"] = initial_per_res_plddt_ae
            metrics["predicted_per_residue_plddt_absolute_error"] = pred_per_res_plddt_ae
            batch_metrics.append(metrics)
        return batch_metrics

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
    
    @typechecked
    def configure_gradient_clipping(
        self,
        optimizer: torch.optim.Optimizer,
        optimizer_idx: int,
        gradient_clip_val: Optional[Union[int, float]] = None,
        gradient_clip_algorithm: Optional[str] = None,
        verbose: bool = False
    ):
        if not self.hparams.module_cfg.clip_gradients:
            return

        # allow gradient norm to be 150% + 2 * stdev of recent gradient history
        max_grad_norm = (
            1.5 * self.gradnorm_queue.mean() + 2 * self.gradnorm_queue.std()
        )

        # get current `grad_norm`
        params = [p for g in optimizer.param_groups for p in g["params"]]
        grad_norm = get_grad_norm(params, device=self.device)

        # note: Lightning will then handle the gradient clipping
        self.clip_gradients(
            optimizer,
            gradient_clip_val=max_grad_norm,
            gradient_clip_algorithm="norm"
        )

        if float(grad_norm) > max_grad_norm:
            self.gradnorm_queue.add(float(max_grad_norm))
        else:
            self.gradnorm_queue.add(float(grad_norm))

        if verbose:
            log.info(f"Current gradient norm: {grad_norm}")

        if float(grad_norm) > max_grad_norm:
            log.info(
                f"Clipped gradient with value {grad_norm:.1f}, since the maximum value currently allowed is {max_grad_norm:.1f}")

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
                os.makedirs(grid_search_script_dir, exist_ok=True)
                with open(fit_end_indicator_filepath, "w") as f:
                    f.write("`on_fit_end` has been called.")
        return super().on_fit_end()


if __name__ == "__main__":
    import hydra
    import omegaconf
    import pyrootutils

    root = pyrootutils.setup_root(__file__, pythonpath=True)
    cfg = omegaconf.OmegaConf.load(root / "configs" / "model" / "gcpnet_eq.yaml")
    _ = hydra.utils.instantiate(cfg)
