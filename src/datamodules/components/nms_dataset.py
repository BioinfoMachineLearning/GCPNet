# -------------------------------------------------------------------------------------------------------------------------------------
# Following code curated for GCPNet (https://github.com/BioinfoMachineLearning/GCPNet):
# -------------------------------------------------------------------------------------------------------------------------------------

import os
from typing import Literal, Optional, Tuple, Union
import torch
import numpy as np

from torch.utils import data as data
from torch_geometric.data import Data
from src.datamodules.components.atom3d_dataset import _edge_features
from src.datamodules.components.helper import _normalize, _rbf
from src.datamodules.components.protein_graph_dataset import ProteinGraphDataset

from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked

patch_typeguard()  # use before @typechecked


@typechecked
def _edge_features(
    coords: TensorType["num_nodes", 3],
    edge_index: TensorType[2, "num_edges"],
    edge_attr: TensorType["num_edges", "num_edge_scalar_features"],
    D_max: float = 4.5,
    num_rbf: int = 16,
    device: Union[torch.device, str] = "cpu"
) -> Tuple[
    TensorType["num_edges", "num_cat_edge_scalar_features"],
    TensorType["num_edges", "num_edge_vector_features", 3]
]:
    E_vectors = coords[edge_index[0]] - coords[edge_index[1]]
    rbf = _rbf(E_vectors.norm(dim=-1),
               D_max=D_max,
               D_count=num_rbf,
               device=device)

    edge_s = torch.cat((edge_attr, rbf), dim=-1)
    edge_v = _normalize(E_vectors).unsqueeze(-2)

    edge_s, edge_v = map(torch.nan_to_num, (edge_s, edge_v))

    return edge_s, edge_v


@typechecked
def _node_features(
    vel: TensorType["num_nodes", 3],
    coords: TensorType["num_nodes", 3]
) -> Tuple[
    TensorType["num_nodes", "num_node_scalar_features"],
    TensorType["num_nodes", "num_cat_node_vector_features", 3]
]:
    orientations = ProteinGraphDataset._orientations(coords)

    node_s = torch.sqrt(torch.sum(vel ** 2, dim=-1)).unsqueeze(-1)
    node_v = torch.cat((vel.unsqueeze(1), orientations), dim=1)

    return node_s, node_v


class NMSDataset(data.Dataset):
    """
    From https://github.com/mouthful/ClofNet/blob/master/newtonian

    Newtonian Many-Body System (NMS) Dataset:
    {
        small: ES
        static: G+ES
        dynamic: L+ES
    }
    """

    def __init__(self,
                 partition: Literal["train", "valid", "test"] = "train",
                 data_root: Optional[str] = None,
                 data_mode: Literal["small", "small_20body", "static", "dynamic"] = "small",
                 rbf_edge_dist_cutoff: float = 4.5,
                 num_rbf: int = 16,
                 device: Union[torch.device, str] = "cpu",
                 max_samples: int = 1e8,
                 frame_0: int = 30,
                 frame_T: int = 40):
        super(NMSDataset, self).__init__()
        
        # hyperparameters
        self.partition = partition
        self.suffix = partition
        self.data_root = data_root
        self.rbf_edge_dist_cutoff = rbf_edge_dist_cutoff
        self.num_rbf = num_rbf
        self.device = device
        self.max_samples = int(max_samples)

        # secure data filename
        if data_mode == "small":
            self.suffix += "_charged5_initvel1small"
        elif (data_mode == "static") or (data_mode == "dynamic"):
            self.suffix += f"_{data_mode}20_initvel1{data_mode}_20body"
        elif data_mode == "small_20body":
            self.suffix += f"_charged20_initvel1{data_mode}"
        else:
            self.suffix += f"_{data_mode[:-7]}20_initvel1{data_mode}"

        # load graphs
        self.data, self.edges = self.load_graphs()
        self.node_counts = [self.data[0].shape[1] for _ in range(self.data[0].shape[0])]
        self.edge_counts = [self.data[2].shape[1] for _ in range(self.data[0].shape[0])]

        # select time span
        self.frame_0, self.frame_T = frame_0, frame_T

    def load_graphs(self) -> Tuple[
        Tuple[
            TensorType["num_samples", "num_nodes", 1],
            TensorType["num_samples", "num_timesteps", "num_nodes", 3],
            TensorType["num_samples", "num_edges", 1],
            TensorType["num_samples", "num_timesteps", "num_nodes", 3]
        ],
        TensorType[2, "num_edges"]
    ]:
        charges = np.load(os.path.join(self.data_root, "charges_" + self.suffix + ".npy"))
        vel = np.load(os.path.join(self.data_root, "vel_" + self.suffix + ".npy"))
        loc = np.load(os.path.join(self.data_root, "loc_" + self.suffix + ".npy"))
        edges = np.load(os.path.join(self.data_root, "edges_" + self.suffix + ".npy"))

        charges, vel, edge_attr, loc, edges = self.preprocess_graphs(charges, vel, loc, edges)
        return (charges, vel, edge_attr, loc), edges

    @typechecked
    def preprocess_graphs(
        self,
        charges: np.ndarray,
        vel: np.ndarray,
        loc: np.ndarray,
        edges: np.ndarray
    ) -> Tuple[
        TensorType["num_samples", "num_nodes", 1],
        TensorType["num_samples", "num_timesteps", "num_nodes", 3],
        TensorType["num_samples", "num_edges", 1],
        TensorType["num_samples", "num_timesteps", "num_nodes", 3],
        TensorType[2, "num_edges"]
    ]:
        # limit number of scalar node features
        charges = charges[0:self.max_samples]
        # cast to torch and swap num_nodes <--> num_features dimensions
        vel = torch.Tensor(vel).transpose(2, 3)
        loc = torch.Tensor(loc).transpose(2, 3)
        num_nodes = loc.shape[2]  # cache number of nodes
        loc = loc[0:self.max_samples, :, :, :]  # limit number of vector-valued node positions
        # limit number of speeds (i.e., vector-valued node features) when starting the trajectory
        vel = vel[0:self.max_samples, :, :, :]

        # initialize edges and edge attributes
        edge_attr = []
        rows, cols = [], []
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j:
                    edge_attr.append(edges[:, i, j])
                    rows.append(i)
                    cols.append(j)
        edges = [rows, cols]
        # swap num_nodes <--> batch_size and add num_features dimension
        edge_attr = torch.Tensor(np.array(edge_attr)).transpose(0, 1).unsqueeze(2)

        return torch.Tensor(charges), torch.Tensor(vel), torch.Tensor(edge_attr), torch.Tensor(loc), torch.LongTensor(edges)

    @typechecked
    def set_max_samples(self, max_samples: int):
        self.max_samples = max_samples
        self.data, self.edges = self.load_graphs()

    def get_num_nodes(self) -> int:
        return self.data[0].shape[1]

    @typechecked
    def _featurize_as_graph(
        self,
        vel: TensorType["num_nodes", 3],
        edge_attr: TensorType["num_edges", 1],
        loc_O: TensorType["num_nodes", 3],
        loc_T: TensorType["num_nodes", 3],
        edge_index: TensorType[2, "num_edges"]
    ) -> Data:
        edge_s, edge_v = _edge_features(loc_O,
                                        edge_index,
                                        edge_attr,
                                        D_max=self.rbf_edge_dist_cutoff,
                                        num_rbf=self.num_rbf,
                                        device=self.device)

        node_s, node_v = _node_features(vel, loc_O)

        data = Data(
            h=node_s,
            chi=node_v,
            e=edge_s,
            xi=edge_v,
            x=loc_O,
            label=loc_T,
            edge_index=edge_index
        )
        return data

    def __len__(self):
        return len(self.data[0])

    def __getitem__(self, idx: int):
        (_, vel, edge_attr, loc), edge_index = self.data, self.edges
        vel, edge_attr, loc = vel[idx], edge_attr[idx], loc[idx]
        return self._featurize_as_graph(
            vel[self.frame_0], edge_attr, loc[self.frame_0], loc[self.frame_T], edge_index
        )


if __name__ == "__main__":
    NMSDataset()
