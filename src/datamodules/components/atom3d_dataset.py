# -------------------------------------------------------------------------------------------------------------------------------------
# Following code curated for GCPNet (https://github.com/BioinfoMachineLearning/GCPNet):
# -------------------------------------------------------------------------------------------------------------------------------------

import torch
import torch_cluster
import torch_geometric
import pandas as pd

from torch_geometric.data import Data
from typing import Any, Dict, Optional, Tuple, Union

from src.datamodules.components.helper import _normalize, _rbf, _orientations

from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked

patch_typeguard()  # use before @typechecked


_atom_types_dict: Dict[str, int] = {
    "H": 0,
    "C": 1,
    "N": 2,
    "O": 3,
    "F": 4,
    "S": 5,
    "Cl": 6,
    "CL": 6,
    "P": 7
}

NUM_ATOM_TYPES = len(_atom_types_dict)


@typechecked
def _element_mapping(x: str) -> int:
    return _atom_types_dict.get(x, 8)


@typechecked
def _edge_features(
    coords: TensorType["num_nodes", 3],
    edge_index: TensorType[2, "num_edges"],
    D_max: float = 4.5,
    num_rbf: int = 16,
    device: Union[torch.device, str] = "cpu"
) -> Tuple[
    TensorType["num_edges", "num_edge_scalar_features"],
    TensorType["num_edges", "num_edge_vector_features", 3]
]:
    E_vectors = coords[edge_index[0]] - coords[edge_index[1]]
    rbf = _rbf(E_vectors.norm(dim=-1),
               D_max=D_max,
               D_count=num_rbf,
               device=device)

    edge_s = rbf
    edge_v = _normalize(E_vectors).unsqueeze(-2)

    edge_s, edge_v = map(torch.nan_to_num, (edge_s, edge_v))

    return edge_s, edge_v


@typechecked
def _node_features(
    df: pd.DataFrame,
    coords: TensorType["num_nodes", 3],
    device: Union[torch.device, str] = "cpu"
) -> Tuple[
    TensorType["num_nodes"],
    TensorType["num_nodes", "num_node_vector_features", 3]
]:
    atoms = torch.as_tensor(
        list(map(_element_mapping, df.element)), dtype=torch.long, device=device
    )
    orientations = _orientations(coords)

    node_s = atoms
    node_v = orientations

    return node_s, node_v


class BaseTransform:
    """
    From https://github.com/drorlab/gvp-pytorch
    """

    def __init__(
        self,
        edge_cutoff: float = 4.5,
        num_rbf: int = 16,
        max_num_neighbors: int = 32,
        device: Union[torch.device, str] = "cpu"
    ):
        self.edge_cutoff = edge_cutoff
        self.num_rbf = num_rbf
        self.max_num_neighbors = max_num_neighbors
        self.device = device

    def __call__(self, df: pd.DataFrame, edge_index: Optional[Any] = None) -> Data:
        with torch.no_grad():
            coords = df[["x", "y", "z"]].to_numpy()
            coords = torch.as_tensor(coords,
                                     dtype=torch.float32,
                                     device=self.device)

            edge_index = torch_cluster.radius_graph(coords,
                                                    r=self.edge_cutoff,
                                                    max_num_neighbors=self.max_num_neighbors)

            edge_s, edge_v = _edge_features(coords,
                                            edge_index,
                                            D_max=self.edge_cutoff,
                                            num_rbf=self.num_rbf,
                                            device=self.device)

            node_s, node_v = _node_features(df,
                                            coords,
                                            device=self.device)

            return torch_geometric.data.Data(h=node_s,
                                             chi=node_v,
                                             e=edge_s,
                                             xi=edge_v,
                                             x=coords,
                                             edge_index=edge_index)


########################################################################

class LBATransform(BaseTransform):
    """
    From https://github.com/drorlab/gvp-pytorch
    """

    def __call__(self, elem: Any, index: int = -1):
        pocket, ligand = elem["atoms_pocket"], elem["atoms_ligand"]
        df = pd.concat([pocket, ligand], ignore_index=True)

        data = super().__call__(df)
        with torch.no_grad():
            data.label = elem["scores"]["neglog_aff"]
            lig_flag = torch.zeros(df.shape[0], device=self.device, dtype=torch.bool)
            lig_flag[-len(ligand):] = 1
            data.lig_flag = lig_flag
        return data


class PSRTransform(BaseTransform):
    """
    From https://github.com/drorlab/gvp-pytorch
    """

    def __call__(self, elem: Any, index: int = -1):
        df = elem["atoms"]
        df = df[df.element != "H"].reset_index(drop=True)
        data = super().__call__(df, elem.get("edge_index", None))
        data.label = elem["scores"]["gdt_ts"]
        data.id = eval(elem["id"])[0]
        return data
