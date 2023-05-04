# -------------------------------------------------------------------------------------------------------------------------------------
# Following code curated for GCPNet (https://github.com/BioinfoMachineLearning/GCPNet):
# -------------------------------------------------------------------------------------------------------------------------------------

from copy import copy
from functools import partial
from typing import Any, Optional, Tuple, Union
import torch
from torch import nn
import torch.nn.functional as F


from torch_geometric.data import Batch
from torch_geometric.utils import subgraph
from torch_scatter import scatter

from omegaconf import OmegaConf, DictConfig

from src.datamodules.components.atom3d_dataset import NUM_ATOM_TYPES

from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked

from src.models import get_nonlinearity
from src.models.components import GCPDropout, GCPLayerNorm, ScalarVector, is_identity, safe_norm, scalarize, vectorize

patch_typeguard()  # use before @typechecked


class GCP(nn.Module):
    def __init__(
            self,
            input_dims: ScalarVector,
            output_dims: ScalarVector,
            nonlinearities: Tuple[Optional[str]] = ("relu", "sigmoid"),
            scalar_gate: int = 0,
            vector_gate: bool = True,
            frame_gate: bool = False,
            sigma_frame_gate: bool = False,
            bottleneck: int = 1,
            vector_residual: bool = False,
            vector_frame_residual: bool = False,
            ablate_frame_updates: bool = False,
            ablate_scalars: bool = False,
            ablate_vectors: bool = False,
            enable_e3_equivariance: bool = False,
            scalarization_vectorization_output_dim: int = 3,
            **kwargs
    ):
        super(GCP, self).__init__()
        if nonlinearities is None:
            nonlinearities = (None, None)

        self.scalar_input_dim, self.vector_input_dim = input_dims
        self.scalar_output_dim, self.vector_output_dim = output_dims
        self.scalar_nonlinearity, self.vector_nonlinearity = (
            get_nonlinearity(nonlinearities[0], return_functional=True),
            get_nonlinearity(nonlinearities[1], return_functional=True)
        )
        self.scalar_gate, self.vector_gate, self.frame_gate, self.sigma_frame_gate = (
            scalar_gate, vector_gate, frame_gate, sigma_frame_gate
        )
        self.vector_residual, self.vector_frame_residual = vector_residual, vector_frame_residual
        self.ablate_frame_updates = ablate_frame_updates
        self.ablate_scalars, self.ablate_vectors = ablate_scalars, ablate_vectors
        self.enable_e3_equivariance = enable_e3_equivariance

        if self.scalar_gate > 0:
            self.norm = nn.LayerNorm(self.scalar_output_dim)

        if self.vector_input_dim:
            assert (
                self.vector_input_dim % bottleneck == 0
            ), f"Input channel of vector ({self.vector_input_dim}) must be divisible with bottleneck factor ({bottleneck})"

            self.hidden_dim = self.vector_input_dim // bottleneck if bottleneck > 1 else max(self.vector_input_dim,
                                                                                             self.vector_output_dim)

            self.vector_down = nn.Linear(self.vector_input_dim, self.hidden_dim, bias=False)
            self.scalar_out = nn.Linear(self.hidden_dim + self.scalar_input_dim, self.scalar_output_dim)

            if self.vector_output_dim:
                self.vector_up = nn.Linear(self.hidden_dim, self.vector_output_dim, bias=False)
                if self.vector_gate:
                    self.vector_out_scale = nn.Linear(self.scalar_output_dim, self.vector_output_dim)

            if not self.ablate_frame_updates:
                vector_down_frames_input_dim = self.hidden_dim if not self.vector_output_dim else self.vector_output_dim
                self.vector_down_frames = nn.Linear(vector_down_frames_input_dim,
                                                    scalarization_vectorization_output_dim, bias=False)
                self.scalar_out_frames = nn.Linear(
                    self.scalar_output_dim + scalarization_vectorization_output_dim * 3, self.scalar_output_dim)

                if self.vector_output_dim and self.sigma_frame_gate:
                    self.vector_out_scale_sigma_frames = nn.Linear(self.scalar_output_dim, self.vector_output_dim)
                elif self.vector_output_dim and self.frame_gate:
                    self.vector_out_scale_frames = nn.Linear(
                        self.scalar_output_dim, scalarization_vectorization_output_dim * 3)
                    self.vector_up_frames = nn.Linear(
                        scalarization_vectorization_output_dim, self.vector_output_dim, bias=False)
        else:
            self.scalar_out = nn.Linear(self.scalar_input_dim, self.scalar_output_dim)

    @typechecked
    def process_vector(
        self,
        scalar_rep: TensorType["batch_num_entities", "merged_scalar_dim"],
        v_pre: TensorType["batch_num_entities", 3, "m"],
        vector_hidden_rep: TensorType["batch_num_entities", 3, "n"]
    ) -> TensorType["batch_num_entities", "o", 3]:
        vector_rep = self.vector_up(vector_hidden_rep)
        if self.vector_residual:
            vector_rep = vector_rep + v_pre
        vector_rep = vector_rep.transpose(-1, -2)
        if self.vector_gate:
            gate = self.vector_out_scale(self.vector_nonlinearity(scalar_rep))
            vector_rep = vector_rep * torch.sigmoid(gate).unsqueeze(-1)
        elif not is_identity(self.vector_nonlinearity):
            vector_rep = vector_rep * self.vector_nonlinearity(safe_norm(vector_rep, dim=-1, keepdim=True))

        return vector_rep

    @typechecked
    def create_zero_vector(
        self,
        scalar_rep: TensorType["batch_num_entities", "merged_scalar_dim"]
    ) -> TensorType["batch_num_entities", "o", 3]:
        return torch.zeros(scalar_rep.shape[0], self.vector_output_dim, 3, device=scalar_rep.device)

    @typechecked
    def process_vector_frames(
        self,
        scalar_rep: TensorType["batch_num_entities", "merged_scalar_dim"],
        v_pre: TensorType["batch_num_entities", 3, "o"],
        edge_index: TensorType[2, "batch_num_edges"],
        frames: TensorType["batch_num_edges", 3, 3],
        node_inputs: bool,
        node_mask: Optional[TensorType["batch_num_nodes"]] = None
    ) -> TensorType["batch_num_entities", "p", 3]:
        vector_rep = v_pre.transpose(-1, -2)
        if self.sigma_frame_gate:
            # bypass vectorization in favor of row-wise gating
            gate = self.vector_out_scale_sigma_frames(self.vector_nonlinearity(scalar_rep))
            vector_rep = vector_rep * torch.sigmoid(gate).unsqueeze(-1)
        elif self.frame_gate:
            # apply elementwise gating between localized frame vectors and vector residuals
            gate = self.vector_out_scale_frames(self.vector_nonlinearity(scalar_rep))
            # perform frame-gating, where edges must be present
            gate_vector = vectorize(
                gate,
                edge_index,
                frames,
                node_inputs=node_inputs,
                dim_size=scalar_rep.shape[0],
                node_mask=node_mask
            )
            # ensure the channels for `coordinates` are being left-multiplied
            gate_vector_rep = self.vector_up_frames(gate_vector.transpose(-1, -2)).transpose(-1, -2)
            vector_rep = vector_rep * self.vector_nonlinearity(safe_norm(gate_vector_rep, dim=-1, keepdim=True))
            if self.vector_frame_residual:
                vector_rep = vector_rep + v_pre.transpose(-1, -2)
        elif not is_identity(self.vector_nonlinearity):
            vector_rep = vector_rep * self.vector_nonlinearity(safe_norm(vector_rep, dim=-1, keepdim=True))

        return vector_rep

    @typechecked
    def forward(
        self,
        s_maybe_v: Union[
            Tuple[
                TensorType["batch_num_entities", "scalar_dim"],
                TensorType["batch_num_entities", "m", "vector_dim"]
            ],
            TensorType["batch_num_entities", "merged_scalar_dim"]
        ],
        edge_index: TensorType[2, "batch_num_edges"],
        frames: TensorType["batch_num_edges", 3, 3],
        node_inputs: bool = False,
        node_mask: Optional[TensorType["batch_num_nodes"]] = None
    ) -> Union[
        Tuple[
            TensorType["batch_num_entities", "new_scalar_dim"],
            TensorType["batch_num_entities", "n", "vector_dim"]
        ],
        TensorType["batch_num_entities", "new_scalar_dim"]
    ]:
        if self.vector_input_dim:
            scalar_rep, vector_rep = s_maybe_v
            scalar_rep = torch.zeros_like(scalar_rep) if self.ablate_scalars else scalar_rep
            vector_rep = torch.zeros_like(vector_rep) if self.ablate_vectors else vector_rep
            v_pre = vector_rep.transpose(-1, -2)

            vector_hidden_rep = self.vector_down(v_pre)
            vector_norm = safe_norm(vector_hidden_rep, dim=-2)
            merged = torch.cat((scalar_rep, vector_norm), dim=-1)
        else:
            merged = s_maybe_v
            merged = torch.zeros_like(merged) if self.ablate_scalars else merged

        scalar_rep = self.scalar_out(merged)

        if self.vector_input_dim and self.vector_output_dim:
            vector_rep = self.process_vector(scalar_rep, v_pre, vector_hidden_rep)

        scalar_rep = self.scalar_nonlinearity(scalar_rep)
        vector_rep = self.create_zero_vector(
            scalar_rep
        ) if self.vector_output_dim and not self.vector_input_dim else vector_rep

        if self.ablate_frame_updates:
            return ScalarVector(scalar_rep, vector_rep) if self.vector_output_dim else scalar_rep

        # GCP: update scalar features using complete local frames
        v_pre = vector_rep.transpose(-1, -2)
        vector_hidden_rep = self.vector_down_frames(v_pre)
        scalar_hidden_rep = scalarize(
            vector_hidden_rep.transpose(-1, -2),
            edge_index,
            frames,
            node_inputs=node_inputs,
            enable_e3_equivariance=self.enable_e3_equivariance,
            dim_size=vector_hidden_rep.shape[0],
            node_mask=node_mask
        )
        merged = torch.cat((scalar_rep, scalar_hidden_rep), dim=-1)

        scalar_rep = self.scalar_out_frames(merged)

        if not self.vector_output_dim:
            # bypass updating vector features using complete local frames (e.g., in the case of a final layer)
            scalar_rep = torch.zeros_like(scalar_rep) if self.ablate_scalars else scalar_rep
            return self.scalar_nonlinearity(scalar_rep)

        # GCP: update vector features using complete local frames
        if self.vector_input_dim and self.vector_output_dim:
            vector_rep = self.process_vector_frames(
                scalar_rep,
                v_pre,
                edge_index,
                frames,
                node_inputs=node_inputs,
                node_mask=node_mask
            )

        scalar_rep = self.scalar_nonlinearity(scalar_rep)
        scalar_rep = torch.zeros_like(scalar_rep) if self.ablate_scalars else scalar_rep
        vector_rep = torch.zeros_like(vector_rep) if self.ablate_vectors else vector_rep
        return ScalarVector(scalar_rep, vector_rep)


class GCP2(nn.Module):
    def __init__(
            self,
            input_dims: ScalarVector,
            output_dims: ScalarVector,
            nonlinearities: Tuple[Optional[str]] = ("relu", "sigmoid"),
            scalar_gate: int = 0,
            vector_gate: bool = True,
            frame_gate: bool = False,
            sigma_frame_gate: bool = False,
            bottleneck: int = 1,
            vector_residual: bool = False,
            vector_frame_residual: bool = False,
            ablate_frame_updates: bool = False,
            ablate_scalars: bool = False,
            ablate_vectors: bool = False,
            enable_e3_equivariance: bool = False,
            scalarization_vectorization_output_dim: int = 3,
            **kwargs
    ):
        super(GCP2, self).__init__()
        if nonlinearities is None:
            nonlinearities = (None, None)

        self.scalar_input_dim, self.vector_input_dim = input_dims
        self.scalar_output_dim, self.vector_output_dim = output_dims
        self.scalar_nonlinearity, self.vector_nonlinearity = (
            get_nonlinearity(nonlinearities[0], return_functional=True),
            get_nonlinearity(nonlinearities[1], return_functional=True)
        )
        self.scalar_gate, self.vector_gate, self.frame_gate, self.sigma_frame_gate = (
            scalar_gate, vector_gate, frame_gate, sigma_frame_gate
        )
        self.vector_residual, self.vector_frame_residual = vector_residual, vector_frame_residual
        self.ablate_frame_updates = ablate_frame_updates
        self.ablate_scalars, self.ablate_vectors = ablate_scalars, ablate_vectors
        self.enable_e3_equivariance = enable_e3_equivariance

        if self.scalar_gate > 0:
            self.norm = nn.LayerNorm(self.scalar_output_dim)

        if self.vector_input_dim:
            assert (
                self.vector_input_dim % bottleneck == 0
            ), f"Input channel of vector ({self.vector_input_dim}) must be divisible with bottleneck factor ({bottleneck})"

            self.hidden_dim = self.vector_input_dim // bottleneck if bottleneck > 1 else max(self.vector_input_dim,
                                                                                             self.vector_output_dim)

            scalar_vector_frame_dim = (scalarization_vectorization_output_dim *
                                       3) if not self.ablate_frame_updates else 0
            self.vector_down = nn.Linear(self.vector_input_dim, self.hidden_dim, bias=False)
            self.scalar_out = nn.Linear(self.hidden_dim + self.scalar_input_dim +
                                        scalar_vector_frame_dim, self.scalar_output_dim)

            if not self.ablate_frame_updates:
                self.vector_down_frames = nn.Linear(
                    self.vector_input_dim, scalarization_vectorization_output_dim, bias=False)

            if self.vector_output_dim:
                self.vector_up = nn.Linear(self.hidden_dim, self.vector_output_dim, bias=False)
                if not self.ablate_frame_updates:
                    if self.frame_gate:
                        self.vector_out_scale_frames = nn.Linear(
                            self.scalar_output_dim, scalarization_vectorization_output_dim * 3)
                        self.vector_up_frames = nn.Linear(
                            scalarization_vectorization_output_dim, self.vector_output_dim, bias=False)
                    elif self.vector_gate:
                        self.vector_out_scale = nn.Linear(self.scalar_output_dim, self.vector_output_dim)
                elif self.vector_gate:
                    self.vector_out_scale = nn.Linear(self.scalar_output_dim, self.vector_output_dim)
        else:
            self.scalar_out = nn.Linear(self.scalar_input_dim, self.scalar_output_dim)

    @typechecked
    def create_zero_vector(
        self,
        scalar_rep: TensorType["batch_num_entities", "merged_scalar_dim"]
    ) -> TensorType["batch_num_entities", "o", 3]:
        return torch.zeros(scalar_rep.shape[0], self.vector_output_dim, 3, device=scalar_rep.device)

    @typechecked
    def process_vector_without_frames(
        self,
        scalar_rep: TensorType["batch_num_entities", "merged_scalar_dim"],
        v_pre: TensorType["batch_num_entities", 3, "m"],
        vector_hidden_rep: TensorType["batch_num_entities", 3, "n"]
    ) -> TensorType["batch_num_entities", "o", 3]:
        vector_rep = self.vector_up(vector_hidden_rep)
        if self.vector_residual:
            vector_rep = vector_rep + v_pre
        vector_rep = vector_rep.transpose(-1, -2)

        if self.vector_gate:
            gate = self.vector_out_scale(self.vector_nonlinearity(scalar_rep))
            vector_rep = vector_rep * torch.sigmoid(gate).unsqueeze(-1)
        elif not is_identity(self.vector_nonlinearity):
            vector_rep = vector_rep * self.vector_nonlinearity(safe_norm(vector_rep, dim=-1, keepdim=True))

        return vector_rep

    @typechecked
    def process_vector_with_frames(
        self,
        scalar_rep: TensorType["batch_num_entities", "merged_scalar_dim"],
        v_pre: TensorType["batch_num_entities", 3, "m"],
        vector_hidden_rep: TensorType["batch_num_entities", 3, "n"],
        edge_index: TensorType[2, "batch_num_edges"],
        frames: TensorType["batch_num_edges", 3, 3],
        node_inputs: bool,
        node_mask: Optional[TensorType["batch_num_nodes"]] = None
    ) -> TensorType["batch_num_entities", "o", 3]:
        vector_rep = self.vector_up(vector_hidden_rep)
        if self.vector_residual:
            vector_rep = vector_rep + v_pre
        vector_rep = vector_rep.transpose(-1, -2)

        if self.frame_gate:
            # derive vector features from direction-robust frames
            gate = self.vector_out_scale_frames(self.vector_nonlinearity(scalar_rep))
            # perform frame-gating, where edges must be present
            gate_vector = vectorize(
                gate,
                edge_index,
                frames,
                node_inputs=node_inputs,
                dim_size=scalar_rep.shape[0],
                node_mask=node_mask
            )
            # ensure frame vector channels for `coordinates` are being left-multiplied
            gate_vector_rep = self.vector_up_frames(gate_vector.transpose(-1, -2)).transpose(-1, -2)
            # apply row-wise scalar gating with frame vector
            vector_rep = vector_rep * self.vector_nonlinearity(safe_norm(gate_vector_rep, dim=-1, keepdim=True))
        elif self.vector_gate:
            gate = self.vector_out_scale(self.vector_nonlinearity(scalar_rep))
            vector_rep = vector_rep * torch.sigmoid(gate).unsqueeze(-1)
        elif not is_identity(self.vector_nonlinearity):
            vector_rep = vector_rep * self.vector_nonlinearity(safe_norm(vector_rep, dim=-1, keepdim=True))

        return vector_rep

    @typechecked
    def forward(
        self,
        s_maybe_v: Union[
            Tuple[
                TensorType["batch_num_entities", "scalar_dim"],
                TensorType["batch_num_entities", "m", "vector_dim"]
            ],
            TensorType["batch_num_entities", "merged_scalar_dim"]
        ],
        edge_index: TensorType[2, "batch_num_edges"],
        frames: TensorType["batch_num_edges", 3, 3],
        node_inputs: bool = False,
        node_mask: Optional[TensorType["batch_num_nodes"]] = None
    ) -> Union[
        Tuple[
            TensorType["batch_num_entities", "new_scalar_dim"],
            TensorType["batch_num_entities", "n", "vector_dim"]
        ],
        TensorType["batch_num_entities", "new_scalar_dim"]
    ]:
        if self.vector_input_dim:
            scalar_rep, vector_rep = s_maybe_v
            scalar_rep = torch.zeros_like(scalar_rep) if self.ablate_scalars else scalar_rep
            vector_rep = torch.zeros_like(vector_rep) if self.ablate_vectors else vector_rep
            v_pre = vector_rep.transpose(-1, -2)

            vector_hidden_rep = self.vector_down(v_pre)
            vector_norm = safe_norm(vector_hidden_rep, dim=-2)
            merged = torch.cat((scalar_rep, vector_norm), dim=-1)

            if not self.ablate_frame_updates:
                # GCP2: curate direction-robust scalar geometric features
                vector_down_frames_hidden_rep = self.vector_down_frames(v_pre)
                scalar_hidden_rep = scalarize(
                    vector_down_frames_hidden_rep.transpose(-1, -2),
                    edge_index,
                    frames,
                    node_inputs=node_inputs,
                    enable_e3_equivariance=self.enable_e3_equivariance,
                    dim_size=vector_down_frames_hidden_rep.shape[0],
                    node_mask=node_mask
                )
                merged = torch.cat((merged, scalar_hidden_rep), dim=-1)
        else:
            # bypass updating scalar features using vector information
            merged = s_maybe_v

        scalar_rep = self.scalar_out(merged)

        if not self.vector_output_dim:
            # bypass updating vector features using scalar information
            scalar_rep = torch.zeros_like(scalar_rep) if self.ablate_scalars else scalar_rep
            return self.scalar_nonlinearity(scalar_rep)
        elif self.vector_output_dim and not self.vector_input_dim:
            # instantiate vector features that are learnable in proceeding GCP layers
            vector_rep = self.create_zero_vector(scalar_rep)
        elif self.ablate_frame_updates:
            # GCP-Baseline: update vector features using row-wise scalar gating
            vector_rep = self.process_vector_without_frames(scalar_rep, v_pre, vector_hidden_rep)
        else:
            # GCP2: update vector features using either row-wise scalar gating with complete local frames or row-wise self-scalar gating
            vector_rep = self.process_vector_with_frames(
                scalar_rep,
                v_pre,
                vector_hidden_rep,
                edge_index,
                frames,
                node_inputs=node_inputs,
                node_mask=node_mask
            )

        scalar_rep = self.scalar_nonlinearity(scalar_rep)
        scalar_rep = torch.zeros_like(scalar_rep) if self.ablate_scalars else scalar_rep
        vector_rep = torch.zeros_like(vector_rep) if self.ablate_vectors else vector_rep
        return ScalarVector(scalar_rep, vector_rep)


class GCPEmbedding(nn.Module):
    def __init__(
        self,
        edge_input_dims: ScalarVector,
        node_input_dims: ScalarVector,
        edge_hidden_dims: ScalarVector,
        node_hidden_dims: ScalarVector,
        num_atom_types: int = NUM_ATOM_TYPES,
        num_lig_flags: int = 2,
        cfg: DictConfig = None,
        pre_norm: bool = True
    ):
        super(GCPEmbedding, self).__init__()
        if num_atom_types > 0:
            self.atom_embedding = nn.Embedding(num_atom_types, num_atom_types)
        else:
            self.atom_embedding = None

        self.concatenate_lig_flag = getattr(cfg, "concatenate_lig_flag", None)
        if self.concatenate_lig_flag:
            node_input_dims += ScalarVector(num_lig_flags, 0)
            self.lig_flag_embedding = nn.Embedding(num_lig_flags, num_lig_flags)

        self.pre_norm = pre_norm
        if pre_norm:
            self.edge_normalization = GCPLayerNorm(edge_input_dims)
            self.node_normalization = GCPLayerNorm(node_input_dims)
        else:
            self.edge_normalization = GCPLayerNorm(edge_hidden_dims)
            self.node_normalization = GCPLayerNorm(node_hidden_dims)

        self.edge_embedding = cfg.selected_GCP(
            edge_input_dims,
            edge_hidden_dims,
            nonlinearities=(None, None),
            scalar_gate=cfg.scalar_gate,
            vector_gate=cfg.vector_gate,
            frame_gate=cfg.frame_gate,
            sigma_frame_gate=cfg.sigma_frame_gate,
            vector_frame_residual=cfg.vector_frame_residual,
            ablate_frame_updates=cfg.ablate_frame_updates,
            ablate_scalars=cfg.ablate_scalars,
            ablate_vectors=cfg.ablate_vectors,
            enable_e3_equivariance=cfg.enable_e3_equivariance
        )

        self.node_embedding = cfg.selected_GCP(
            node_input_dims,
            node_hidden_dims,
            nonlinearities=(None, None),
            scalar_gate=cfg.scalar_gate,
            vector_gate=cfg.vector_gate,
            frame_gate=cfg.frame_gate,
            sigma_frame_gate=cfg.sigma_frame_gate,
            vector_frame_residual=cfg.vector_frame_residual,
            ablate_frame_updates=cfg.ablate_frame_updates,
            ablate_scalars=cfg.ablate_scalars,
            ablate_vectors=cfg.ablate_vectors,
            enable_e3_equivariance=cfg.enable_e3_equivariance
        )

    @typechecked
    def forward(
        self,
        batch: Batch
    ) -> Tuple[
        Tuple[
            TensorType["batch_num_nodes", "h_hidden_dim"],
            TensorType["batch_num_nodes", "m", "chi_hidden_dim"]
        ],
        Tuple[
            TensorType["batch_num_edges", "e_hidden_dim"],
            TensorType["batch_num_edges", "x", "xi_hidden_dim"]
        ]
    ]:
        if self.atom_embedding is not None:
            node_rep = ScalarVector(self.atom_embedding(batch.h), batch.chi)
        else:
            node_rep = ScalarVector(batch.h, batch.chi)

        if self.concatenate_lig_flag:
            lig_flag_embedding = self.lig_flag_embedding(batch.lig_flag.long())
            new_scalar_node_rep = torch.cat((node_rep[0], lig_flag_embedding), dim=-1)
            node_rep = ScalarVector(new_scalar_node_rep, batch.chi)

        edge_rep = ScalarVector(batch.e, batch.xi)
        if self.pre_norm:
            edge_rep = self.edge_normalization(edge_rep)
            node_rep = self.node_normalization(node_rep)

        edge_rep = self.edge_embedding(
            edge_rep,
            batch.edge_index,
            batch.f_ij,
            node_inputs=False,
            node_mask=getattr(batch, "mask", None)
        )
        node_rep = self.node_embedding(
            node_rep,
            batch.edge_index,
            batch.f_ij,
            node_inputs=True,
            node_mask=getattr(batch, "mask", None)
        )

        if not self.pre_norm:
            edge_rep = self.edge_normalization(edge_rep)
            node_rep = self.node_normalization(node_rep)

        return node_rep, edge_rep


def get_GCP_with_custom_cfg(input_dims, output_dims, cfg: DictConfig, **kwargs):
    cfg_dict = copy(OmegaConf.to_container(cfg, throw_on_missing=True))
    cfg_dict["nonlinearities"] = cfg.nonlinearities
    del cfg_dict["scalar_nonlinearity"]
    del cfg_dict["vector_nonlinearity"]

    for key in kwargs:
        cfg_dict[key] = kwargs[key]

    return cfg.selected_GCP(input_dims, output_dims, **cfg_dict)


class GCPMessagePassing(nn.Module):
    def __init__(
        self,
        input_dims: ScalarVector,
        output_dims: ScalarVector,
        edge_dims: ScalarVector,
        cfg: DictConfig,
        mp_cfg: DictConfig,
        reduce_function: str = "mean"
    ):
        super().__init__()

        # hyperparameters
        self.scalar_input_dim, self.vector_input_dim = input_dims
        self.scalar_output_dim, self.vector_output_dim = output_dims
        self.edge_scalar_dim, self.edge_vector_dim = edge_dims
        self.conv_cfg = mp_cfg
        self.self_message = self.conv_cfg.self_message
        self.use_residual_message_gcp = self.conv_cfg.use_residual_message_gcp
        self.reduce_function = reduce_function

        scalars_in_dim = 2 * self.scalar_input_dim + self.edge_scalar_dim
        vectors_in_dim = 2 * self.vector_input_dim + self.edge_vector_dim

        # config instantiations
        soft_cfg = copy(cfg)
        soft_cfg.bottleneck, soft_cfg.vector_residual = cfg.default_bottleneck, cfg.default_vector_residual

        primary_cfg_GCP = partial(get_GCP_with_custom_cfg, cfg=soft_cfg)
        secondary_cfg_GCP = partial(get_GCP_with_custom_cfg, cfg=cfg)

        # PyTorch modules #
        module_list = [
            primary_cfg_GCP(
                (scalars_in_dim, vectors_in_dim),
                output_dims,
                nonlinearities=cfg.nonlinearities if self.conv_cfg.num_message_layers > 1 else None,
                enable_e3_equivariance=cfg.enable_e3_equivariance
            )
        ]

        for _ in range(self.conv_cfg.num_message_layers - 2):
            module_list.append(secondary_cfg_GCP(output_dims, output_dims, enable_e3_equivariance=cfg.enable_e3_equivariance))

        if self.conv_cfg.num_message_layers > 1:
            module_list.append(primary_cfg_GCP(output_dims, output_dims, nonlinearities=(None, None), enable_e3_equivariance=cfg.enable_e3_equivariance))

        self.message_fusion = nn.ModuleList(module_list)

    @typechecked
    def message(
        self,
        node_rep: ScalarVector,
        edge_rep: ScalarVector,
        edge_index: TensorType[2, "batch_num_edges"],
        frames: TensorType["batch_num_edges", 3, 3],
        node_mask: Optional[TensorType["batch_num_nodes"]] = None
    ) -> TensorType["batch_num_edges", "message_dim"]:
        row, col = edge_index
        vector = node_rep.vector.reshape(node_rep.vector.shape[0], node_rep.vector.shape[1] * node_rep.vector.shape[2])
        vector_reshaped = ScalarVector(node_rep.scalar, vector)

        s_row, v_row = vector_reshaped.idx(row)
        s_col, v_col = vector_reshaped.idx(col)

        v_row = v_row.reshape(v_row.shape[0], v_row.shape[1] // 3, 3)
        v_col = v_col.reshape(v_col.shape[0], v_col.shape[1] // 3, 3)

        message = ScalarVector(s_row, v_row).concat((edge_rep, ScalarVector(s_col, v_col)))

        if self.use_residual_message_gcp:
            message_residual = self.message_fusion[0](message, edge_index, frames, node_inputs=False, node_mask=node_mask)
            for module in self.message_fusion[1:]:
                # ResGCP: exchange geometric messages while maintaining residual connection to original message
                new_message = module(message_residual, edge_index, frames, node_inputs=False, node_mask=node_mask)
                message_residual = message_residual + new_message
        else:
            message_residual = message
            for module in self.message_fusion:
                # ablate ResGCP: exchange geometric messages without maintaining residual connection to original message
                message_residual = module(message_residual, edge_index, frames, node_inputs=False, node_mask=node_mask)

        return message_residual.flatten()

    @typechecked
    def aggregate(
        self,
        message: TensorType["batch_num_edges", "message_dim"],
        edge_index: TensorType[2, "batch_num_edges"],
        dim_size: int
    ) -> TensorType["batch_num_nodes", "aggregate_dim"]:
        row, col = edge_index
        aggregate = scatter(message, col, dim=0, dim_size=dim_size, reduce=self.reduce_function)
        return aggregate

    @typechecked
    def forward(
        self,
        node_rep: ScalarVector,
        edge_rep: ScalarVector,
        edge_index: TensorType[2, "batch_num_edges"],
        frames: TensorType["batch_num_edges", 3, 3],
        node_mask: Optional[TensorType["batch_num_nodes"]] = None
    ) -> ScalarVector:
        message = self.message(node_rep, edge_rep, edge_index, frames, node_mask=node_mask)
        aggregate = self.aggregate(message, edge_index, dim_size=node_rep.scalar.shape[0])
        return ScalarVector.recover(aggregate, self.vector_output_dim)


class GCPInteractions(nn.Module):
    def __init__(
        self,
        node_dims: ScalarVector,
        edge_dims: ScalarVector,
        cfg: DictConfig,
        layer_cfg: DictConfig,
        dropout: float = 0.1,
        autoregressive: bool = False,
        nonlinearities: Optional[Tuple[Any, Any]] = None,
        updating_node_positions: bool = False
    ):
        super().__init__()

        # hyperparameters #
        if nonlinearities is None:
            nonlinearities = cfg.nonlinearities
        self.pre_norm = layer_cfg.pre_norm
        self.updating_node_positions = updating_node_positions
        self.ablate_x_force_update = getattr(cfg, "ablate_x_force_update", True)
        self.node_positions_weight = getattr(cfg, "node_positions_weight", 1.0)
        reduce_function = "add" if autoregressive else "mean"

        # PyTorch modules #

        # geometry-complete message-passing neural network
        message_function = GCPMessagePassing

        self.interaction = message_function(
            node_dims,
            node_dims,
            edge_dims,
            reduce_function=reduce_function,
            cfg=cfg,
            mp_cfg=layer_cfg.mp_cfg
        )

        # config instantiations
        ff_cfg = copy(cfg)
        ff_cfg.nonlinearities = nonlinearities
        ff_without_res_cfg = copy(cfg)
        ff_without_res_cfg.vector_residual = False

        ff_GCP = partial(get_GCP_with_custom_cfg, cfg=ff_cfg)
        ff_without_res_GCP = partial(get_GCP_with_custom_cfg, cfg=ff_without_res_cfg)

        self.gcp_norm = nn.ModuleList([GCPLayerNorm(node_dims) for _ in range(2)])
        self.gcp_dropout = nn.ModuleList([GCPDropout(dropout) for _ in range(2)])

        # build out feedforward (FF) network modules
        ff_interaction_layers = []
        hidden_dims = node_dims if layer_cfg.num_feedforward_layers == 1 else 4 * node_dims.scalar, 2 * node_dims.vector
        ff_interaction_layers.append(
            ff_without_res_GCP(
                node_dims, hidden_dims,
                nonlinearities=None if layer_cfg.num_feedforward_layers == 1 else cfg.nonlinearities,
                enable_e3_equivariance=cfg.enable_e3_equivariance
            )
        )

        interaction_layers = [
            ff_GCP(hidden_dims, hidden_dims, enable_e3_equivariance=cfg.enable_e3_equivariance)
            for _ in range(layer_cfg.num_feedforward_layers - 2)
        ]
        ff_interaction_layers.extend(interaction_layers)

        if layer_cfg.num_feedforward_layers > 1:
            ff_interaction_layers.append(
                ff_without_res_GCP(
                    hidden_dims, node_dims,
                    nonlinearities=(None, None),
                    enable_e3_equivariance=cfg.enable_e3_equivariance
                )
            )

        self.feedforward_network = nn.ModuleList(ff_interaction_layers)

        # potentially build out node position update modules
        if updating_node_positions:
            # node position update GCPs
            node_position_update_gcps = [
                ff_without_res_GCP(
                    node_dims, (node_dims.scalar, 1),
                    nonlinearities=cfg.nonlinearities,
                    enable_e3_equivariance=cfg.enable_e3_equivariance
                )
            ]
            self.node_position_update_network = nn.ModuleList(node_position_update_gcps)

            # node position force-update layers
            scalar_hidden_dim = node_dims.scalar
            scalar_nonlinearity = cfg.nonlinearities[0]
            self.phi_force_i = None if self.ablate_x_force_update else nn.Linear(scalar_hidden_dim, scalar_hidden_dim)
            self.phi_force_j = None if self.ablate_x_force_update else nn.Linear(scalar_hidden_dim, scalar_hidden_dim)
            phi_x_force_ij_layer = None if self.ablate_x_force_update else nn.Linear(scalar_hidden_dim, 3, bias=False)
            None if self.ablate_x_force_update else torch.nn.init.xavier_uniform_(
                phi_x_force_ij_layer.weight, gain=0.001)
            self.phi_force_ij = None if self.ablate_x_force_update else nn.Sequential(
                get_nonlinearity(scalar_nonlinearity, layer_cfg.nonlinearity_slope),
                phi_x_force_ij_layer
            )

    @typechecked
    def autoregressive_forward(
        self,
        node_rep: ScalarVector,
        edge_rep: ScalarVector,
        edge_index: TensorType[2, "batch_num_edges"],
        frames: TensorType["batch_num_edges", 3, 3],
        autoregressive_node_rep: ScalarVector,
        node_mask: Optional[TensorType["batch_num_nodes"]] = None
    ) -> ScalarVector:
        # derive edge mask
        row, col = edge_index
        edge_mask = row < col

        # mask out edges and their features
        edge_rep_forward = edge_rep.idx(edge_mask)
        edge_rep_backward = edge_rep.idx(~edge_mask)
        edge_index_forward = edge_index[:, edge_mask]
        edge_index_backward = edge_index[:, ~edge_mask]
        frames_forward = frames[edge_mask, :, :]
        frames_backward = frames[~edge_mask, :, :]

        # perform forward and backward passes
        autoregressive_node_rep = ScalarVector(autoregressive_node_rep[0], autoregressive_node_rep[1])
        forward_interaction = self.interaction(
            node_rep,
            edge_rep_forward,
            edge_index_forward,
            frames_forward,
            node_mask=node_mask
        )
        backward_interaction = self.interaction(
            autoregressive_node_rep,
            edge_rep_backward,
            edge_index_backward,
            frames_backward,
            node_mask=node_mask
        )
        autoregressive_mp = forward_interaction + backward_interaction

        count = scatter(
            torch.ones_like(col),
            col,
            dim_size=autoregressive_mp[0].shape[0],
            reduce="sum"
        ).clamp(min=1).unsqueeze(-1)
        autoregressive_mp = ScalarVector(
            autoregressive_mp[0] / count,
            autoregressive_mp[1] / count.unsqueeze(-1)
        )

        return autoregressive_mp

    @typechecked
    def derive_x_update(
        self,
        node_rep: ScalarVector,
        edge_index: TensorType[2, "batch_num_edges"],
        f_ij: TensorType["batch_num_edges", 3, 3],
        node_mask: Optional[TensorType["batch_num_nodes"]] = None
    ) -> TensorType["batch_num_nodes", 3]:
        row, col = edge_index

        # VectorUpdate: use vector-valued features to derive node position updates
        (h_v, chi_v) = node_rep
        for position_update_gcp in self.node_position_update_network:
            (h_v, chi_v) = position_update_gcp(
                (h_v, chi_v),
                edge_index,
                f_ij,
                node_inputs=True,
                node_mask=node_mask
            )

        # ForceUpdate: use inter-atom forces to derive node position updates from each neighboring node
        if self.ablate_x_force_update:
            x_force_update = torch.zeros((h_v.shape[0], 3), device=h_v.device)
        else:
            f_ij = f_ij.reshape(f_ij.shape[0], 1, -1)
            x_diff, x_cross, x_vertical = f_ij[:, :, :3].squeeze(), f_ij[:, :, 3:6].squeeze(), f_ij[:, :, 6:].squeeze()
            
            h_i, h_j = h_v[row], h_v[col]
            x_force_coef = self.phi_force_ij(self.phi_force_i(h_i) + self.phi_force_j(h_j))
            x_force_update = (
                x_force_coef[:, :1] * x_diff + x_force_coef[:, 1:2] * x_cross + x_force_coef[:, 2:3] * x_vertical
            )

            # summarize node position updates across neighboring nodes
            x_force_update = scatter(x_force_update, col, dim=0, reduce="mean")

        # combine scalar and vector-valued features to curate a single positional update for each node
        x_update = (chi_v.squeeze(1) + x_force_update) * self.node_positions_weight  # (up/down)weight position updates

        return x_update.clamp(min=-100, max=100)  # note: not used but may save training

    @typechecked
    def forward(
        self,
        node_rep: Tuple[TensorType["batch_num_nodes", "node_hidden_dim"], TensorType["batch_num_nodes", "m", 3]],
        edge_rep: Tuple[TensorType["batch_num_edges", "edge_hidden_dim"], TensorType["batch_num_edges", "x", 3]],
        edge_index: TensorType[2, "batch_num_edges"],
        frames: TensorType["batch_num_edges", 3, 3],
        node_rep_regressive: Optional[Tuple[TensorType["batch_num_nodes",
                                                       "hidden_dim"], TensorType["batch_num_nodes", "m", 3]]] = None,
        node_mask: Optional[TensorType["batch_num_nodes"]] = None,
        node_pos: Optional[TensorType["batch_num_nodes", 3]] = None
    ) -> Union[
        Tuple[
            TensorType["batch_num_nodes", "hidden_dim"],
            TensorType["batch_num_nodes", "n", 3]
        ],
        Tuple[
            Tuple[
                TensorType["batch_num_nodes", "hidden_dim"],
                TensorType["batch_num_nodes", "n", 3]
            ],
            TensorType["batch_num_nodes", 3]
        ]
    ]:
        node_rep = ScalarVector(node_rep[0], node_rep[1])
        edge_rep = ScalarVector(edge_rep[0], edge_rep[1])

        # apply GCP normalization (1)
        if self.pre_norm:
            node_rep = self.gcp_norm[0](node_rep)

        # forward propagate with interaction module
        if node_rep_regressive is not None:
            hidden_residual = self.autoregressive_forward(
                node_rep, edge_rep, edge_index, frames, node_rep_regressive, node_mask=node_mask
            )
        else:
            hidden_residual = self.interaction(
                node_rep, edge_rep, edge_index, frames, node_mask=node_mask
            )

        # mask out nodes
        if node_mask is not None:
            node_rep_residual = node_rep
            node_rep, hidden_residual = node_rep.idx(node_mask), hidden_residual.idx(node_mask)
            creating_subgraph = not node_mask.all()
            if creating_subgraph:
                if edge_index.shape[1] == 0:
                    # handle for (sub)graphs corresponding to nodes with no edges
                    subgraph_edge_index, subgraph_frames = (edge_index, frames)
                else:
                    nodes_subset = torch.where(node_mask)[0]
                    subgraph_edge_index, subgraph_frames = subgraph(
                        nodes_subset,
                        edge_index=edge_index,
                        edge_attr=frames,
                        relabel_nodes=True
                    )

        # apply GCP dropout (1)
        node_rep = node_rep + self.gcp_dropout[0](hidden_residual)

        # apply GCP normalization (2)
        if self.pre_norm:
            node_rep = self.gcp_norm[1](node_rep)
        else:
            node_rep = self.gcp_norm[0](node_rep)

        # propagate with feedforward layers
        hidden_residual = node_rep
        ff_edge_index = subgraph_edge_index if node_mask is not None and creating_subgraph else edge_index
        ff_frames = subgraph_frames if node_mask is not None and creating_subgraph else frames
        for module in self.feedforward_network:
            hidden_residual = module(
                hidden_residual,
                ff_edge_index,
                ff_frames,
                node_inputs=True,
                node_mask=node_mask
            )

        # apply GCP dropout (2)
        node_rep = node_rep + self.gcp_dropout[1](hidden_residual)

        # apply GCP normalization (3)
        if not self.pre_norm:
            node_rep = self.gcp_norm[1](node_rep)

        # update only unmasked node representations and residuals
        if node_mask is not None:
            node_rep_residual[0][node_mask], node_rep_residual[1][node_mask] = node_rep[0], node_rep[1]
            node_rep = node_rep_residual

        # bypass updating node positions
        if not self.updating_node_positions:
            return node_rep

        # update node positions
        node_pos = node_pos + self.derive_x_update(
            node_rep, edge_index, frames, node_mask=node_mask
        )

        return node_rep, node_pos


class GCPMLPDecoder(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        vocab_size: int = 20,
        num_layers: int = 1,
        residual_updates: bool = False
    ):
        super().__init__()
        self.residual_updates = residual_updates

        readout_layers = [
            nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers - 1)
        ] + [nn.Linear(hidden_dim, vocab_size)]
        self.readout = nn.ModuleList(readout_layers) if residual_updates else nn.Sequential(*readout_layers)

    @typechecked
    def residual_forward(
        self,
        h: TensorType["batch_num_nodes", "h_hidden_dim"]
    ) -> TensorType["batch_num_nodes", "vocab_size"]:
        h_readout = h
        for layer in self.readout[:-1]:
            h_readout = h_readout + layer(h_readout)
        logits = self.readout[-1](h_readout)
        return logits

    @typechecked
    def forward(
        self,
        h: TensorType["batch_num_nodes", "h_hidden_dim"]
    ) -> Tuple[
        TensorType["batch_num_nodes", "vocab_size"],
        TensorType["batch_num_nodes", "vocab_size"]
    ]:
        logits = self.residual_forward(h) if self.residual_updates else self.readout(h)
        log_probs = F.log_softmax(logits, dim=-1)
        return logits, log_probs


if __name__ == "__main__":
    _ = GCPInteractions()
