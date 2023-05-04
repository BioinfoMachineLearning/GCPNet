# -------------------------------------------------------------------------------------------------------------------------------------
# Following code curated for GCPNet (https://github.com/BioinfoMachineLearning/GCPNet):
# -------------------------------------------------------------------------------------------------------------------------------------

import copy
import os
import torch
import torchmetrics

import torch.nn as nn
import torch.nn.functional as F

from pytorch_lightning import LightningModule
from typing import Any, Dict, List, Optional, Tuple, Union
from torch_geometric.data import Data, Batch
from torch_geometric.utils import unbatch
from omegaconf import DictConfig
from torch.distributions import Categorical

from src.models import HALT_FILE_EXTENSION
from src.models.components import centralize, localize
from src.models.components.gcpnet import GCPEmbedding, GCPMLPDecoder, ScalarVector

from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked

patch_typeguard()  # use before @typechecked


class GCPNetCPDLitModule(LightningModule):
    """LightningModule for computational protein design (CPD) using GCPNet.

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
        node_input_dims: List[int],
        edge_input_dims: List[int],
        model_cfg: DictConfig,
        module_cfg: DictConfig,
        layer_cfg: DictConfig,
        path_cfg: DictConfig = None,
        dropout: float = 0.1,
        autoregressive_decoder: bool = False,
        **kwargs
    ):
        super().__init__()

        # this line allows to access init params with `self.hparams` attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False, ignore=["layer_class"])

        # feature dimensionalities
        self.node_dims = ScalarVector(model_cfg.h_hidden_dim, model_cfg.chi_hidden_dim)
        self.edge_dims = ScalarVector(model_cfg.e_hidden_dim, model_cfg.xi_hidden_dim)
        edge_hidden_dims = (self.edge_dims[0] + 20, self.edge_dims[1])

        # PyTorch modules #

        # input embeddings
        self.gcp_embedding = GCPEmbedding(
            edge_input_dims,
            node_input_dims,
            self.edge_dims,
            self.node_dims,
            num_atom_types=0,
            cfg=module_cfg,
            pre_norm=False
        )

        # message-passing encoder layers
        self.encoder_layers = nn.ModuleList(
            layer_class(
                self.node_dims,
                self.edge_dims,
                cfg=module_cfg,
                layer_cfg=layer_cfg,
                dropout=dropout
            ) for _ in range(model_cfg.num_encoder_layers)
        )

        if autoregressive_decoder:
            # replace frame gate with vector gate subsequently if frame gate was initially requested
            module_cfg.vector_gate = module_cfg.frame_gate
            module_cfg.frame_gate = False
            module_cfg.ablate_frame_updates = True

            # embed residue types for autoregressive decoding
            self.atom_embedding = nn.Embedding(model_cfg.output_dim, model_cfg.output_dim)

            # message-passing decoder layers (where frame gate updates are effectively replaced with vector gate updates)
            self.decoder_layers = nn.ModuleList(
                layer_class(
                    self.node_dims,
                    edge_hidden_dims,
                    cfg=module_cfg,
                    layer_cfg=layer_cfg,
                    dropout=dropout,
                    autoregressive=True
                ) for _ in range(model_cfg.num_decoder_layers)
            )

        # GCP to coalesce scalar and vector-valued node features into scalar node features
        invariant_node_projection_dim = model_cfg.output_dim if autoregressive_decoder else self.node_dims[0]
        self.invariant_node_projection = module_cfg.selected_GCP(
            self.node_dims,
            (invariant_node_projection_dim, 0),
            nonlinearities=(None, None),
            scalar_gate=module_cfg.scalar_gate,
            vector_gate=module_cfg.vector_gate,
            frame_gate=module_cfg.frame_gate,
            sigma_frame_gate=module_cfg.sigma_frame_gate,
            vector_frame_residual=module_cfg.vector_frame_residual,
            ablate_frame_updates=module_cfg.ablate_frame_updates,
            ablate_scalars=module_cfg.ablate_scalars,
            ablate_vectors=module_cfg.ablate_vectors,
            enable_e3_equivariance=module_cfg.enable_e3_equivariance
        )

        if not autoregressive_decoder:
            # MLP-based decoder to predict amino acid type probabilites for each node
            self.decoder = GCPMLPDecoder(
                invariant_node_projection_dim,
                vocab_size=model_cfg.output_dim,
                num_layers=model_cfg.num_decoder_layers,
                residual_updates=model_cfg.decoder_residual_updates
            )

        # loss function and metrics #
        self.criterion = torch.nn.CrossEntropyLoss()
        # note: for averaging loss across batches
        self.train_loss = torchmetrics.MeanMetric()
        self.val_loss = torchmetrics.MeanMetric()
        self.test_loss = torchmetrics.MeanMetric()

        self.train_phase, self.val_phase, self.test_phase = "train", "val", "test"

        # note: for averaging perplexity across batches
        self.train_perplexity = torchmetrics.MeanMetric()
        self.val_perplexity = torchmetrics.MeanMetric()

    @typechecked
    def forward(self, batch: Any) -> Tuple[Any, Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]]:
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

        # embed node and edge input features
        (h, chi), (e, xi) = self.gcp_embedding(batch)

        # encode graph features using a series of geometric message-passing layers
        for layer in self.encoder_layers:
            (h, chi) = layer(
                (h, chi),
                (e, xi),
                batch.edge_index,
                batch.f_ij,
                node_mask=batch.mask
            )

        if self.hparams.autoregressive_decoder:
            # embed and mask input sequence
            encoder_embedding = (h, chi)

            sequence_embedding = self.atom_embedding(batch.seq)
            sequence_embedding = sequence_embedding[batch.edge_index[0]]
            sequence_embedding[batch.edge_index[0] >= batch.edge_index[1]] = 0

            # inject invariant input sequence into current invariant edge representations
            (e, xi) = (torch.cat((e, sequence_embedding), dim=-1), xi)

            # decode graph features using a series of geometric message-passing layers
            for layer in self.decoder_layers:
                (h, chi) = layer(
                    (h, chi),
                    (e, xi),
                    batch.edge_index,
                    batch.f_ij,
                    node_rep_regressive=encoder_embedding,
                    node_mask=batch.mask
                )

        # record final version of each feature in `Batch` object
        batch.h, batch.chi, batch.e, batch.xi = h, chi, e, xi

        # summarize intermediate node representations as final predictions
        out = self.invariant_node_projection(
            (batch.h, batch.chi),
            batch.edge_index,
            batch.f_ij,
            node_inputs=True,
            node_mask=batch.mask
        )  # e.g., GCP((h, chi)) -> h'

        if not self.hparams.autoregressive_decoder:
            # perform a direct-shot prediction of amino acid type probabilities
            out = self.decoder(out)

        return batch, out

    def training_step(self, batch: Any, batch_idx: int):
        _, (preds, _) = self.forward(batch)
        preds, labels = preds[batch.mask], batch.seq[batch.mask]
        loss = self.criterion(preds, labels)

        # update metrics
        self.train_loss(loss.detach())
        self.train_perplexity(torch.exp(loss.detach()))

        return {"loss": loss, "preds": preds, "labels": labels}

    def training_epoch_end(self, outputs: List[Any]):
        # log metrics
        self.log(f"{self.train_phase}/loss", self.train_loss, prog_bar=False)
        self.log(f"{self.train_phase}/perplexity", self.train_perplexity, prog_bar=False)

    def validation_step(self, batch: Any, batch_idx: int):
        _, (preds, _) = self.forward(batch)
        preds, labels = preds[batch.mask], batch.seq[batch.mask]
        loss = self.criterion(preds, labels)

        # update metrics
        self.val_loss(loss.detach())
        self.val_perplexity(torch.exp(loss.detach()))

        return {"loss": loss, "preds": preds, "labels": labels}

    def validation_epoch_end(self, outputs: List[Any]):
        # log metrics
        self.log(f"{self.val_phase}/loss", self.val_loss, prog_bar=True)
        self.log(f"{self.val_phase}/perplexity", self.val_perplexity, prog_bar=True)

    def load_splits(self):
        if getattr(self.trainer.datamodule, "custom_splits", None) is not None:
            self.subsets = self.trainer.datamodule.custom_splits

    def on_test_start(self):
        """Lightning calls this prior to running `test_step`."""
        self.load_splits()
        metric_keys = list(self.subsets.keys()) + ["all"]
        self.loss_metrics = nn.ModuleDict(
            {key: torchmetrics.CatMetric() for key in metric_keys}
        )
        self.num_nodes_metrics = nn.ModuleDict(
            {key: torchmetrics.CatMetric() for key in metric_keys}
        )
        self.recovery_metrics = nn.ModuleDict(
            {key: torchmetrics.CatMetric() for key in metric_keys}
        )
        super().on_test_start()

    @typechecked
    def autoregressively_generate_samples(
        self,
        node_rep: ScalarVector,
        edge_rep: ScalarVector,
        edge_index: TensorType[2, "num_edges"],
        frames: TensorType["num_edges", 3, 3],
        encoder_node_mask: TensorType["num_nodes"],
        num_samples: int,
        temperature: float = 0.1
    ):
        num_nodes = node_rep[0].shape[0]
        with torch.no_grad():
            edge_rep = self.gcp_embedding.edge_embedding(
                edge_rep, edge_index, frames, node_inputs=False, node_mask=encoder_node_mask
            )
            edge_rep = self.gcp_embedding.edge_normalization(edge_rep)
            node_rep = self.gcp_embedding.node_embedding(
                node_rep, edge_index, frames, node_inputs=True, node_mask=encoder_node_mask
            )
            node_rep = self.gcp_embedding.node_normalization(node_rep)

            for layer in self.encoder_layers:
                node_rep = layer(
                    node_rep,
                    edge_rep,
                    edge_index,
                    frames,
                    node_mask=encoder_node_mask
                )

            node_rep = node_rep.repeat(num_samples, 1, 1)
            edge_rep = edge_rep.repeat(num_samples, 1, 1)

            edge_index = edge_index.expand(num_samples, -1, -1)
            frames = frames.expand(num_samples, -1, -1, -1)
            offset = num_nodes * torch.arange(num_samples, device=node_rep.scalar.device).reshape(-1, 1, 1)
            edge_index = torch.cat(tuple(edge_index + offset), dim=-1)
            frames = torch.cat(tuple(frames), dim=0)

            residue_sequence = torch.zeros(num_samples * num_nodes, device=node_rep.scalar.device, dtype=torch.int)
            sequence_embedding = torch.zeros(num_samples * num_nodes, 20, device=node_rep.scalar.device)

            node_rep_cache = [node_rep.clone() for _ in self.decoder_layers]

            encoder_node_mask_ = encoder_node_mask.repeat(num_samples)

            for i in range(num_nodes):
                sequence_embedding_ = sequence_embedding[edge_index[0]]
                sequence_embedding_[edge_index[0] >= edge_index[1]] = 0
                edge_rep_masked = ScalarVector(torch.cat((edge_rep[0], sequence_embedding_), dim=-1), edge_rep[1])

                edge_mask = (edge_index[1] % num_nodes) == i
                edge_index_ = edge_index[:, edge_mask]
                edge_rep_masked = edge_rep_masked.idx(edge_mask)
                frames_ = frames[edge_mask]
                node_mask = torch.zeros(num_samples * num_nodes, device=node_rep.scalar.device, dtype=torch.bool)
                node_mask[i::num_nodes] = True

                # ensure that nodes with missing coordinates are also masked out
                node_mask = node_mask & encoder_node_mask_

                for j, layer in enumerate(self.decoder_layers):
                    out = layer(
                        node_rep_cache[j],
                        edge_rep_masked,
                        edge_index_,
                        frames_,
                        node_rep_regressive=node_rep_cache[0],
                        node_mask=node_mask
                    )

                    out = out.idx(node_mask)

                    if j < len(self.decoder_layers) - 1:
                        node_rep_cache[j + 1].scalar[i::num_nodes] = out.scalar
                        node_rep_cache[j + 1].vector[i::num_nodes] = out.vector

                logits = self.invariant_node_projection(
                    out,
                    edge_index_,
                    frames_,
                    node_inputs=True,
                    node_mask=node_mask
                )
                residue_sequence[i::num_nodes] = Categorical(logits=logits / temperature).sample()
                sequence_embedding[i::num_nodes] = self.atom_embedding(residue_sequence[i::num_nodes])

            return residue_sequence.reshape(num_samples, num_nodes)

    @typechecked
    def predict_datum_sequence(
        self,
        datum: Data
    ) -> TensorType["num_nodes", "vocab_size"]:
        batch = Batch.from_data_list([datum])
        with torch.no_grad():
            _, (_, log_probs) = self.forward(batch)
        return log_probs

    @typechecked
    def calculate_loss_for_datum(
        self,
        datum: Data,
        log_probs: Optional[TensorType["num_nodes", "vocab_size"]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        log_probs = (
            log_probs
            if log_probs is not None
            else self.predict_datum_sequence(datum)
        )
        log_probs, seq = log_probs[datum.mask], datum.seq[datum.mask]
        loss = F.nll_loss(log_probs, seq, reduction="sum")
        effective_num_nodes = torch.tensor([log_probs.shape[0]], dtype=torch.float, device=loss.device)
        return loss, effective_num_nodes

    @typechecked
    def calculate_recovery_for_datum(
        self,
        datum: Data,
        log_probs: Optional[TensorType["num_nodes", "vocab_size"]] = None
    ) -> torch.Tensor:
        if self.hparams.autoregressive_decoder:
            samples = self.autoregressively_generate_samples(
                ScalarVector(datum.h, datum.chi),
                ScalarVector(datum.e, datum.xi),
                datum.edge_index,
                datum.f_ij,
                encoder_node_mask=datum.mask,
                num_samples=100,
                temperature=0.1
            )
            return samples.eq(datum.seq).float().mean()
        else:
            log_probs = (
                log_probs
                if log_probs is not None
                else self.predict_datum_sequence(datum)
            )
            seq_pred = torch.argmax(log_probs, dim=-1)  # note: no need to take exp() here
            recovery = (seq_pred == datum.seq).float().mean()
            return recovery

    @typechecked
    def compute_test_examples_metrics(
        self,
        data: List[Data],
        x_list: List[torch.Tensor],
        frames_list: List[torch.Tensor]
    ):
        for datum, x, f_ij in zip(data, x_list, frames_list):
            # recover centralized positions and equivariant frames for each datum
            datum.x = x
            datum.f_ij = f_ij
            # perform forward pass with datum
            log_probs = self.predict_datum_sequence(datum)
            # update loss
            loss, num_nodes = self.calculate_loss_for_datum(datum, log_probs=log_probs)
            self.loss_metrics["all"].update(loss.detach())
            for subset in self.subsets:
                if datum.name in self.subsets[subset]:
                    self.loss_metrics[subset].update(loss.detach())
            # update number of nodes
            self.num_nodes_metrics["all"].update(num_nodes.detach())
            for subset in self.subsets:
                if datum.name in self.subsets[subset]:
                    self.num_nodes_metrics[subset].update(num_nodes.detach())
            # update recovery
            recovery = self.calculate_recovery_for_datum(datum, log_probs=log_probs)
            self.recovery_metrics["all"].update(recovery.detach())
            for subset in self.subsets:
                if datum.name in self.subsets[subset]:
                    self.recovery_metrics[subset].update(recovery.detach())

    def test_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0):
        updated_batch, (preds, _) = self.forward(copy.deepcopy(batch))  # break reference to `batch`
        preds, labels = preds[batch.mask], batch.seq[batch.mask]
        loss = self.criterion(preds, labels)

        # update metrics
        self.test_loss(loss.detach())

        # update sequence recovery metric
        if dataloader_idx == 0:
            row, col = updated_batch.edge_index
            # reference: https://github.com/pyg-team/pytorch_geometric/issues/1827#issuecomment-727196996
            edge_batch = updated_batch.batch[row]
            self.compute_test_examples_metrics(
                batch.to_data_list(),
                list(unbatch(updated_batch.x, updated_batch.batch)),
                list(unbatch(updated_batch.f_ij, edge_batch))
            )

        return {"loss": loss, "preds": preds, "labels": labels}

    def compute_test_metrics(self) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        metric_keys = list(self.subsets.keys()) + ["all"]
        perplexity_output, recovery_output = {}, {}
        for key in metric_keys:
            perplexity_output[key] = torch.exp(self.loss_metrics[key].compute().sum() / self.num_nodes_metrics[key].compute().sum())
            recovery_output[key] = torch.median(self.recovery_metrics[key].compute())
        return perplexity_output, recovery_output

    def test_epoch_end(self, outputs: List[Any]):
        # log metrics
        self.log(f"{self.test_phase}/loss", self.test_loss, prog_bar=False)

        # compute and log test metrics
        perplexity_output, recovery_output = self.compute_test_metrics()
        for key in perplexity_output:
            self.log(f"{self.test_phase}/perplexity/" + key, perplexity_output[key], prog_bar=False)
        for key in recovery_output:
            self.log(f"{self.test_phase}/recovery/" + key, recovery_output[key], prog_bar=False)

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
    cfg = omegaconf.OmegaConf.load(root / "configs" / "model" / "gcpnet_cpd.yaml")
    _ = hydra.utils.instantiate(cfg)
