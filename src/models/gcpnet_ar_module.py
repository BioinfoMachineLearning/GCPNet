# -------------------------------------------------------------------------------------------------------------------------------------
# Following code curated for GCPNet (https://github.com/BioinfoMachineLearning/GCPNet):
# -------------------------------------------------------------------------------------------------------------------------------------

import copy
import os
import tempfile
import torch
import torchmetrics
import wandb

import pandas as pd
import torch.nn as nn

from collections import defaultdict
from datetime import datetime
from omegaconf import DictConfig
from pathlib import Path
from pytorch_lightning import LightningModule
from torch_geometric.data import Batch
from typing import Any, Dict, List, Optional, Tuple, Union

from src import utils
from src.datamodules.components.eq_dataset import generate_lddt_score
from src.models import HALT_FILE_EXTENSION, Queue, amber_relax, calculate_molprobity_metrics, calculate_tmscore_metrics, get_grad_norm, write_residue_atom_positions_as_pdb
from src.models.components import centralize, decentralize, localize
from src.models.components.gcpnet import GCPEmbedding, ScalarVector

from torchtyping import patch_typeguard
from typeguard import typechecked

patch_typeguard()  # use before @typechecked


log = utils.get_pylogger(__name__)


class GCPNetARLitModule(LightningModule):
    """LightningModule for atomic refinement (AR) of protein structures using GCPNet.

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
        **kwargs
    ):
        super().__init__()

        # this line allows to access init params with `self.hparams` attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False, ignore=["layer_class"])

        # feature dimensionalities
        edge_input_dims = ScalarVector(model_cfg.e_input_dim, model_cfg.xi_input_dim)
        node_input_dims = ScalarVector(model_cfg.h_input_dim, model_cfg.chi_input_dim)
        self.edge_dims = ScalarVector(model_cfg.e_hidden_dim, model_cfg.xi_hidden_dim)
        self.node_dims = ScalarVector(model_cfg.h_hidden_dim, model_cfg.chi_hidden_dim)

        # PyTorch modules #
        # input embeddings
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

        # training #
        if self.hparams.module_cfg.clip_gradients:
            self.gradnorm_queue = Queue()
            self.gradnorm_queue.add(3000)  # add large value that will be flushed

        # loss function #
        self.criterion = torch.nn.MSELoss(reduction="sum")
        # note: for averaging loss across batches
        self.train_loss = torchmetrics.MeanMetric()
        self.val_loss = torchmetrics.MeanMetric()
        self.test_loss = torchmetrics.MeanMetric()
        
        # use separate metrics instances for the steps
        # of each phase (e.g., train, val and test)
        # to ensure a proper reduction over the epoch
        self.train_phase, self.val_phase, self.test_phase, self.predict_phase = "train", "val", "test", "predict"
        refinement_test_metrics = [
            "GDT-TS", "GDT-HA", "RMSD", "lddt_score",
            "improvement_score", "molprobity_score",
            "TM-score", "MaxSub", "clash_score",
            "rotamer_outliers", "ramachandran_outliers"
        ]
        self.all_refinement_test_metrics = [
            metric_name if metric_name in ["improvement_score"] else f"{prefix}_{metric_name}"
            for metric_name in refinement_test_metrics
            for prefix in ["init", "pred", "relaxed_pred"]
        ]

    @staticmethod
    def get_labels(batch) -> Any:
        if type(batch) in [list, tuple]:
            return batch[0].label
        return batch.label

    @typechecked
    def forward(self, batch: Batch) -> Tuple[Any, torch.Tensor]:
        # centralize node positions to make them translation-invariant
        batch_x_input = copy.deepcopy(batch.x)
        x_centroid, batch.x = centralize(batch, key="x", batch_index=batch.batch)

        # craft complete local frames corresponding to each edge
        batch.f_ij = localize(batch.x, batch.edge_index, norm_x_diff=self.hparams.module_cfg.norm_x_diff)

        # embed node and edge input features
        (h, chi), (e, xi) = self.gcp_embedding(batch)

        # update graph features using a series of geometric message-passing layers
        for layer in self.interaction_layers:
            (h, chi), batch.x = layer(
                (h, chi), (e, xi),
                batch.edge_index, batch.f_ij,
                node_pos=batch.x
            )

        # record final version of each feature in `Batch` object
        batch.h, batch.chi, batch.e, batch.xi = h, chi, e, xi

        # decentralize updated node positions to make the position updates translation-equivariant
        batch.x = decentralize(batch, key="x", batch_index=batch.batch, entities_centroid=x_centroid)
        batch_x_shift = (batch.x - batch_x_input)

        # use predicted all-atom positions as an "offset" for initial Ca atom positions
        batch_x_pred = []
        num_residues = batch.ca_x.size(0)
        atom_start_idx, atom_end_idx = 0, 0
        for i in range(num_residues):
            atom_start_idx = atom_end_idx
            atom_end_idx += batch.num_atoms_per_residue[i].item()
            batch_x_pred.append(batch.ca_x[i] + batch_x_shift[atom_start_idx:atom_end_idx, :])
        batch_x_pred = torch.cat(batch_x_pred)

        return batch, batch_x_pred

    def step(self, batch: Union[Batch, List[Batch]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # make a forward pass and score it
        if isinstance(batch, list):
            # assemble individual batch inputs (and outputs)
            labels_list, preds_list = [], []
            for batch_ in batch:
                labels_ = self.get_labels(batch_)[batch_.overlap_true_start_atom_index:batch_.overlap_true_end_atom_index]
                preds_ = self.forward(batch_)[-1][batch_.overlap_true_start_atom_index:batch_.overlap_true_end_atom_index]
                labels_list.append(labels_)
                preds_list.append(preds_)
            labels = torch.cat(labels_list, dim=0)
            preds = torch.cat(preds_list, dim=0)
            num_nodes = len(labels)
        else:
            labels = self.get_labels(batch)
            _, preds = self.forward(batch)
            num_nodes = batch.num_nodes
        loss = torch.sqrt(self.criterion(preds, labels) / num_nodes)  # note: node-normalize manually since `reduction=sum`
        return loss, preds, labels
    
    def on_train_start(self):
        # by default lightning executes validation step sanity checks before training starts,
        # so we need to make sure val_`metric`_best doesn't store any values from these checks
        self.val_loss.reset()

        # ensure directory for storing refinement outputs is defined
        if not getattr(self, "refinement_output_dir", None):
            self.refinement_output_dir = Path(self.trainer.default_root_dir)

    def training_step(self, batch: Batch, batch_idx: int):
        try:
            loss, preds, labels = self.step(batch)
        except RuntimeError as e:
            if "CUDA out of memory" not in str(e):
                raise(e)
            torch.cuda.empty_cache()
            log.info(f"Skipping batch with index {batch_idx} due to OOM error...")
            return

        # skip backpropagation if loss was invalid
        if loss.isnan().any() or loss.isinf().any():
            log.info(f"Loss for batch with index {batch_idx} is invalid. Skipping...")
            return

        # update metric(s)
        self.train_loss(loss.detach())

        return {"loss": loss, "preds": preds, "labels": labels}

    def training_epoch_end(self, outputs: List[Any]):
        # log metric(s)
        self.log(f"{self.train_phase}/loss", self.train_loss, prog_bar=False)

    def on_validation_start(self):
        # ensure directory for storing refinement outputs is defined
        if not getattr(self, "refinement_output_dir", None):
            self.refinement_output_dir = Path(self.trainer.default_root_dir)

    def validation_step(self, batch: Batch, batch_idx: int):
        try:
            loss, preds, labels = self.step(batch)
        except RuntimeError as e:
            if "CUDA out of memory" not in str(e):
                raise(e)
            torch.cuda.empty_cache()
            log.info(f"Skipping batch with index {batch_idx} due to OOM error...")
            return

        # update metric(s)
        self.val_loss(loss.detach())

        return {"loss": loss, "preds": preds, "labels": labels}

    def validation_epoch_end(self, outputs: List[Any]):
        # log metric(s)
        self.log(f"{self.val_phase}/loss", self.val_loss, prog_bar=False)

    def on_test_start(self):
        # ensure directory for storing refinement outputs is defined
        if not getattr(self, "refinement_output_dir", None):
            self.refinement_output_dir = Path(self.trainer.default_root_dir)

    @typechecked
    def combine_individual_batch_inputs(self, batch: List[Batch]) -> Batch:
        for batch_ in batch:
            batch_.x = batch_.x[batch_.overlap_true_start_atom_index:batch_.overlap_true_end_atom_index]
            batch_.ca_x = batch_.ca_x[batch_.overlap_true_start_residue_index:batch_.overlap_true_end_residue_index]
            batch_.label = batch_.label[batch_.overlap_true_start_atom_index:batch_.overlap_true_end_atom_index] if hasattr(batch_, "label") else []
            batch_.num_atoms_per_residue = batch_.num_atoms_per_residue[batch_.overlap_true_start_residue_index:batch_.overlap_true_end_residue_index]
            batch_.batch = batch_.batch[batch_.overlap_true_start_atom_index:batch_.overlap_true_end_atom_index]
            residue_to_atom_names_keys = list(batch_.residue_to_atom_names_mapping[0][0])[batch_.overlap_true_start_residue_index:batch_.overlap_true_end_residue_index]
            batch_.residue_to_atom_names_mapping[0] = defaultdict(
                list,
                {key: batch_.residue_to_atom_names_mapping[0][0][key] for key in residue_to_atom_names_keys}
            )
        combined_batch = Batch.from_data_list(batch)
        combined_batch.batch[:] = combined_batch.batch.unique()[0]
        combined_batch.residue_to_atom_names_mapping = [[
            defaultdict(
                list,
                {k: v for d in combined_batch.residue_to_atom_names_mapping for k, v in d[0].items()}
            )
        ]]
        combined_batch.initial_pdb_filepath = combined_batch.initial_pdb_filepath[0]
        combined_batch.true_pdb_filepath = combined_batch.true_pdb_filepath[0] if hasattr(combined_batch, "true_pdb_filepath") else None
        combined_batch._num_nodes = len(combined_batch.x)
        combined_batch._num_graphs = 1
        return combined_batch

    @typechecked
    def test_step(self, batch: Union[Batch, List[Batch]], batch_idx: int, dataloader_idx: int = 0):
        try:
            loss, preds, labels = self.step(batch)
        except RuntimeError as e:
            if "CUDA out of memory" not in str(e):
                raise(e)
            torch.cuda.empty_cache()
            log.info(f"Skipping test batch with index {batch_idx} due to OOM error...")
            return

        # update metric(s)
        self.test_loss(loss.detach())

        # as necessary, combine individual batch inputs into a single object
        if isinstance(batch, list):
            batch = self.combine_individual_batch_inputs(batch)

        # score using external refinement metrics
        refinement_metrics = self.score_refinement_preds(batch, preds, labels)

        return refinement_metrics

    def test_epoch_end(self, refinement_metrics_list: List[Any]):
        # log metric(s)
        self.log(f"{self.test_phase}/loss", self.test_loss, prog_bar=False)

        # tag current refinement outputs according to the phase in which they were generated
        refinement_metrics_list_ = [metrics for metrics_list in refinement_metrics_list for metrics in metrics_list]
        refinement_metrics_list = [{f"{self.test_phase}/{key}": value for key, value in metrics.items()} for metrics in refinement_metrics_list_]

        # compile refinement metrics collected by the current device (e.g., rank zero)
        refinement_metrics_csv_path = os.path.join(
            self.refinement_output_dir,
            f"{self.test_phase}_epoch_{self.current_epoch}_rank_{self.global_rank}_refinement_metrics.csv"
        )
        refinement_metrics_df = pd.DataFrame(refinement_metrics_list)
        refinement_metrics_df.to_csv(refinement_metrics_csv_path, index=False)

        refinement_logs = {
            f"{self.test_phase}/{metric_name}": refinement_metrics_df[f"{self.test_phase}/{metric_name}"].mean()
            for metric_name in self.all_refinement_test_metrics
        }

        if getattr(self, "logger", None) is not None and getattr(self.logger, "experiment", None) is not None:
            # use WandB as our experiment logger
            wandb_run = self.logger.experiment

            refinement_table = wandb.Table(
                columns=refinement_metrics_df.columns.to_list() + [
                    f"{self.test_phase}/effective_initial_structure",
                    f"{self.test_phase}/effective_predicted_structure",
                    f"{self.test_phase}/effective_relaxed_predicted_structure",
                    f"{self.test_phase}/effective_true_structure",
                    f"{self.test_phase}/initial_structure",
                    f"{self.test_phase}/true_structure",
                    f"{self.test_phase}/effective_true_pdb_filepath",
                ]
            )
            for _, row in refinement_metrics_df.iterrows():
                row_metrics = row.to_list() + [
                    wandb.Molecule(row[f"{self.test_phase}/effective_initial_pdb_filepath"]),
                    wandb.Molecule(row[f"{self.test_phase}/effective_predicted_pdb_filepath"]),
                    wandb.Molecule(row[f"{self.test_phase}/effective_relaxed_predicted_pdb_filepath"]),
                    wandb.Molecule(row[f"{self.test_phase}/effective_true_pdb_filepath"]),
                    wandb.Molecule(row[f"{self.test_phase}/initial_pdb_filepath"]),
                    wandb.Molecule(row[f"{self.test_phase}/true_pdb_filepath"]),
                    row[f"{self.test_phase}/effective_true_pdb_filepath"]
                ]
                refinement_table.add_data(*row_metrics)
            refinement_logs[f"{self.test_phase}/refinement_metrics"] = refinement_table

            wandb_run.log(refinement_logs)

        # also log sampling metrics directly
        for metric_name in self.all_refinement_test_metrics:
            self.log(f"{self.test_phase}/{metric_name}", refinement_logs[f"{self.test_phase}/{metric_name}"], sync_dist=True)

    @torch.inference_mode()
    @typechecked
    def score_refinement_preds(self, batch: Batch, preds: torch.Tensor, labels: torch.Tensor) -> List[Dict[str, Any]]:
        # create temporary output PDB files for predictions and labels
        batch_metrics = []
        initial_pos = batch.x.detach().cpu().numpy()
        pred_pos = preds.detach().cpu().numpy()
        label_pos = labels.detach().cpu().numpy()
        batch_index = batch.batch.cpu().numpy()
        for b_index in range(batch.num_graphs):
            metrics = {}
            temp_pdb_dir = tempfile._get_default_tempdir()
            temp_pdb_code = next(tempfile._get_candidate_names())
            initial_path = str(temp_pdb_dir / Path(f"init_{temp_pdb_code}").with_suffix(".pdb"))
            prediction_path = str(temp_pdb_dir / Path(f"pred_{temp_pdb_code}").with_suffix(".pdb"))
            relaxed_prediction_path = str(temp_pdb_dir / Path(f"relaxed_pred_{temp_pdb_code}").with_suffix(".pdb"))
            reference_path = str(temp_pdb_dir / Path(f"ref_{temp_pdb_code}").with_suffix(".pdb"))
            # isolate each individual example within the current batch
            initial_pos_ = initial_pos[batch_index == b_index]
            pred_pos_ = pred_pos[batch_index == b_index]
            label_pos_ = label_pos[batch_index == b_index]
            residue_to_atom_names_mapping_ = batch.residue_to_atom_names_mapping[b_index][0]
            write_residue_atom_positions_as_pdb(initial_path, initial_pos_, residue_to_atom_names_mapping_)
            write_residue_atom_positions_as_pdb(prediction_path, pred_pos_, residue_to_atom_names_mapping_)
            write_residue_atom_positions_as_pdb(reference_path, label_pos_, residue_to_atom_names_mapping_)
            amber_relax(prediction_path, relaxed_prediction_path)  # use AMBER to relax the predicted positions
            # score initial as well as refined structure using TM-score
            init_tmscore_metrics = calculate_tmscore_metrics(initial_path, reference_path, self.hparams.path_cfg.tmscore_exec_path)
            pred_tmscore_metrics = calculate_tmscore_metrics(prediction_path, reference_path, self.hparams.path_cfg.tmscore_exec_path)
            relaxed_pred_tmscore_metrics = calculate_tmscore_metrics(relaxed_prediction_path, reference_path, self.hparams.path_cfg.tmscore_exec_path)
            # score initial as well as refined structure using lDDT
            init_lddt_metric = generate_lddt_score(initial_path, reference_path, self.hparams.path_cfg.lddt_exec_path)
            pred_lddt_metric = generate_lddt_score(prediction_path, reference_path, self.hparams.path_cfg.lddt_exec_path)
            relaxed_pred_lddt_metric = generate_lddt_score(relaxed_prediction_path, reference_path, self.hparams.path_cfg.lddt_exec_path)
            # score initial as well as refined structure using MolProbity
            init_molp_metrics = calculate_molprobity_metrics(initial_path, self.hparams.path_cfg.molprobity_exec_path)
            pred_molp_metrics = calculate_molprobity_metrics(prediction_path, self.hparams.path_cfg.molprobity_exec_path)
            relaxed_pred_molp_metrics = calculate_molprobity_metrics(relaxed_prediction_path, self.hparams.path_cfg.molprobity_exec_path)
            # keep track of improvements to GDT-HA scores
            pred_improvement_score = 1 if pred_tmscore_metrics["GDT-HA"] > init_tmscore_metrics["GDT-HA"] else 0
            relaxed_pred_improvement_score = 1 if relaxed_pred_tmscore_metrics["GDT-HA"] > init_tmscore_metrics["GDT-HA"] else 0
            # combine metrics
            for init_metric in init_tmscore_metrics:
                metrics[f"init_{init_metric}"] = init_tmscore_metrics[init_metric]
            for pred_metric in pred_tmscore_metrics:
                metrics[f"pred_{pred_metric}"] = pred_tmscore_metrics[pred_metric]
            for relaxed_pred_metric in relaxed_pred_tmscore_metrics:
                metrics[f"relaxed_pred_{relaxed_pred_metric}"] = relaxed_pred_tmscore_metrics[relaxed_pred_metric]
            for init_metric in init_molp_metrics:
                metrics[f"init_{init_metric}"] = init_molp_metrics[init_metric]
            for pred_metric in pred_molp_metrics:
                metrics[f"pred_{pred_metric}"] = pred_molp_metrics[pred_metric]
            for relaxed_pred_metric in relaxed_pred_molp_metrics:
                metrics[f"relaxed_pred_{relaxed_pred_metric}"] = relaxed_pred_molp_metrics[relaxed_pred_metric]
            metrics["init_lddt_score"] = init_lddt_metric.mean().item()
            metrics["pred_lddt_score"] = pred_lddt_metric.mean().item()
            metrics["relaxed_pred_lddt_score"] = relaxed_pred_lddt_metric.mean().item()
            metrics["pred_improvement_score"] = pred_improvement_score
            metrics["relaxed_pred_improvement_score"] = relaxed_pred_improvement_score
            metrics["effective_initial_pdb_filepath"] = initial_path
            metrics["effective_predicted_pdb_filepath"] = prediction_path
            metrics["effective_relaxed_predicted_pdb_filepath"] = relaxed_prediction_path
            metrics["effective_true_pdb_filepath"] = reference_path
            metrics["initial_pdb_filepath"] = batch.initial_pdb_filepath[b_index]
            metrics["true_pdb_filepath"] = batch.true_pdb_filepath[b_index]
            batch_metrics.append(metrics)
        return batch_metrics
    
    def on_predict_epoch_start(self):
        # define where the final predictions should be recorded
        self.predictions_csv_path = os.path.join(
            self.trainer.default_root_dir,
            f"{self.predict_phase}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_rank_{self.global_rank}_predictions.csv",
        )
    
    @torch.inference_mode()
    @typechecked
    def predict_step(self, batch: Union[Batch, List[Batch]], batch_idx: int, dataloader_idx: int = 0):
        if isinstance(batch, list):
            preds = torch.cat([self.forward(batch_)[-1] for batch_ in batch], dim=0)
        else:
            _, preds = self.forward(batch)
    
        # as necessary, combine individual batch inputs into a single object
        if isinstance(batch, list):
            batch = self.combine_individual_batch_inputs(batch)

        step_outputs = self.record_refinement_preds(batch, preds)
        return step_outputs
    
    @torch.inference_mode()
    @typechecked
    def on_predict_epoch_end(self, outputs: List[Any]):
        prediction_outputs = [
            output for output_ in outputs for output__ in output_ for output in output__
        ]
        # compile predictions collected by the current device (e.g., rank zero)
        predictions_csv_df = pd.DataFrame(prediction_outputs)
        predictions_csv_df.to_csv(self.predictions_csv_path, index=False)
    
    @torch.inference_mode()
    @typechecked
    def record_refinement_preds(self, batch: Batch, preds: torch.Tensor) -> List[Dict[str, Any]]:
        # create temporary output PDB files for predictions
        batch_metrics = []
        initial_pos = batch.x.detach().cpu().numpy()
        pred_pos = preds.detach().cpu().numpy()
        batch_index = batch.batch.cpu().numpy()
        for b_index in range(batch.num_graphs):
            metrics = {}
            temp_pdb_dir = tempfile._get_default_tempdir()
            temp_pdb_code = next(tempfile._get_candidate_names())
            initial_path = str(temp_pdb_dir / Path(f"init_{temp_pdb_code}").with_suffix(".pdb"))
            prediction_path = str(temp_pdb_dir / Path(f"pred_{temp_pdb_code}").with_suffix(".pdb"))
            relaxed_prediction_path = str(temp_pdb_dir / Path(f"relaxed_pred_{temp_pdb_code}").with_suffix(".pdb"))
            # isolate each individual example within the current batch
            initial_pos_ = initial_pos[batch_index == b_index]
            pred_pos_ = pred_pos[batch_index == b_index]
            residue_to_atom_names_mapping_ = batch.residue_to_atom_names_mapping[b_index][0]
            write_residue_atom_positions_as_pdb(initial_path, initial_pos_, residue_to_atom_names_mapping_)
            write_residue_atom_positions_as_pdb(prediction_path, pred_pos_, residue_to_atom_names_mapping_)
            amber_relax(prediction_path, relaxed_prediction_path)  # use AMBER to relax the predicted positions
            # score initial as well as refined structure using MolProbity
            init_molp_metrics = calculate_molprobity_metrics(initial_path, self.hparams.path_cfg.molprobity_exec_path)
            pred_molp_metrics = calculate_molprobity_metrics(prediction_path, self.hparams.path_cfg.molprobity_exec_path)
            relaxed_pred_molp_metrics = calculate_molprobity_metrics(relaxed_prediction_path, self.hparams.path_cfg.molprobity_exec_path)
            # combine metrics
            for init_metric in init_molp_metrics:
                metrics[f"init_{init_metric}"] = init_molp_metrics[init_metric]
            for pred_metric in pred_molp_metrics:
                metrics[f"pred_{pred_metric}"] = pred_molp_metrics[pred_metric]
            for relaxed_pred_metric in relaxed_pred_molp_metrics:
                metrics[f"relaxed_pred_{relaxed_pred_metric}"] = relaxed_pred_molp_metrics[relaxed_pred_metric]
            metrics["effective_initial_pdb_filepath"] = initial_path
            metrics["effective_predicted_pdb_filepath"] = prediction_path
            metrics["effective_relaxed_predicted_pdb_filepath"] = relaxed_prediction_path
            metrics["initial_pdb_filepath"] = batch.initial_pdb_filepath[b_index]
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
    cfg = omegaconf.OmegaConf.load(root / "configs" / "model" / "gcpnet_ar.yaml")
    _ = hydra.utils.instantiate(cfg)
