# -------------------------------------------------------------------------------------------------------------------------------------
# Following code curated for GCPNet (https://github.com/BioinfoMachineLearning/GCPNet):
# -------------------------------------------------------------------------------------------------------------------------------------

import copy
import itertools
import math
import random
import rdkit
import torch
import torch_geometric

import networkx as nx
import numpy as np

from rdkit.Chem import rdMolTransforms
from typing import Tuple

from src.datamodules.components.helper import _normalize, _rbf
from src.datamodules.components.protein_graph_dataset import ProteinGraphDataset


ATOM_TYPES = ["H", "C", "B", "N", "O", "F", "Si", "P", "S", "Cl", "Br", "I"]
FORMAL_CHARGE = [-1, -2, 1, 2, 0]
DEGREE = [0, 1, 2, 3, 4, 5, 6]
NUM_HS = [0, 1, 2, 3, 4]
LOCAL_CHIRAL_TAGS = [0, 1, 2, 3] 
HYBRIDIZATION = [
    rdkit.Chem.rdchem.HybridizationType.S,
    rdkit.Chem.rdchem.HybridizationType.SP,
    rdkit.Chem.rdchem.HybridizationType.SP2,
    rdkit.Chem.rdchem.HybridizationType.SP3,
    rdkit.Chem.rdchem.HybridizationType.SP3D,
    rdkit.Chem.rdchem.HybridizationType.SP3D2,
    rdkit.Chem.rdchem.HybridizationType.UNSPECIFIED
]
BOND_TYPES = ["SINGLE", "DOUBLE", "TRIPLE", "AROMATIC"]


def one_hot_embedding(value, options):
    embedding = [0] * (len(options) + 1)
    index = options.index(value) if value in options else -1
    embedding[index] = 1
    return embedding


def adjacency_to_undirected_edge_index(adj):
    adj = np.triu(np.array(adj, dtype=int))  # keep just upper triangular entries from sym matrix
    array_adj = np.array(np.nonzero(adj), dtype=int)  # use indices of non-zero values in adj matrix
    edge_index = np.zeros((2, 2 * array_adj.shape[1]), dtype=int)  # make placeholder for undirected edge list
    edge_index[:, ::2] = array_adj
    edge_index[:, 1::2] = np.flipud(array_adj)
    return edge_index


def get_edge_features(list_rdkit_bonds):
    F_e = (len(BOND_TYPES) + 1) + 2 + (6 + 1)  # note: 14
    
    edge_features = np.zeros((len(list_rdkit_bonds) * 2, F_e))
    for edge_index, edge in enumerate(list_rdkit_bonds):
        features = one_hot_embedding(str(edge.GetBondType()), BOND_TYPES)  # note: `dim=4 + 1`
        features += [int(edge.GetIsConjugated())]  # note: `dim=1`
        features += [int(edge.IsInRing())]  # note: `dim=1`
        features += one_hot_embedding(edge.GetStereo(), list(range(6)))  # note: `dim=6 + 1`

        # Encode both directed edges to get undirected edge
        edge_features[2 * edge_index: 2 * edge_index + 2, :] = features
        
    return np.array(edge_features, dtype=np.float32)


def get_node_features(list_rdkit_atoms, owning_mol):
    F_v = (len(ATOM_TYPES) + 1) + \
        (len(DEGREE)+1) + \
        (len(FORMAL_CHARGE) + 1) + \
        (len(NUM_HS)+1) + \
        (len(HYBRIDIZATION) + 1) + \
        2 + 4 + 5  # note: 52
    
    global_tags = dict(rdkit.Chem.FindMolChiralCenters(owning_mol, force=True, includeUnassigned=True, useLegacyImplementation=False))
    
    node_features = np.zeros((len(list_rdkit_atoms), F_v))
    for node_index, node in enumerate(list_rdkit_atoms):
        features = one_hot_embedding(node.GetSymbol(), ATOM_TYPES)  # note: atom symbol, `dim=12 + 1` 
        features += one_hot_embedding(node.GetTotalDegree(), DEGREE)  # note: total number of bonds, H included, `dim=7 + 1`
        features += one_hot_embedding(node.GetFormalCharge(), FORMAL_CHARGE)  # note: formal charge, `dim=5 + 1` 
        features += one_hot_embedding(node.GetTotalNumHs(), NUM_HS)  # note: total number of bonded hydrogens, `dim=5 + 1`
        features += one_hot_embedding(node.GetHybridization(), HYBRIDIZATION)  # note: hybridization state, `dim=7 + 1`
        features += [int(node.GetIsAromatic())]  # note: whether atom is part of aromatic system, `dim = 1`
        features += [node.GetMass()  * 0.01]  # note: `atomic mass` / 100, `dim=1`
        
        ### note: chiral tags go last ###
        # build global chiral tag
        idx = node.GetIdx()
        global_chiral_tag = 0
        if idx in global_tags:
            if global_tags[idx] == "R":
                global_chiral_tag = 1
            elif global_tags[idx] == "S":
                global_chiral_tag = 2
            else:
                global_chiral_tag = -1
        
        features += one_hot_embedding(global_chiral_tag, [0,1,2])  # note: chiral tag of atom, `dim=3 + 1` (global chiral features)
        
        # build local chiral tag
        features += one_hot_embedding(node.GetChiralTag(), LOCAL_CHIRAL_TAGS)  # chiral tag of atom, `dim=4 + 1` (local chiral features)
        
        node_features[node_index,:] = features
        
    return np.array(node_features, dtype = np.float32)


def get_all_paths(G, N=3):
    # adapted from: https://stackoverflow.com/questions/28095646/finding-all-paths-walks-of-given-length-in-a-networkx-graph
    def findPaths(G, u, n):
        if n == 0:
            return [[u]]
        paths = [[u] + path for neighbor in G.neighbors(u) for path in findPaths(G, neighbor, n - 1) if u not in path]
        return paths
    
    allpaths = []
    for node in G:
        allpaths.extend(findPaths(G, node, N))
    
    return allpaths


def get_internal_coordinates_from_all_paths(mol, adj, repeats=False): 
    if isinstance(mol, rdkit.Chem.rdchem.Conformer):
        conformer = mol
    if isinstance(mol, rdkit.Chem.rdchem.Mol):
        conformer = mol.GetConformer()
        
    graph = nx.from_numpy_matrix(adj, parallel_edges=False, create_using=None)
    
    distance_paths, angle_paths, dihedral_paths = get_all_paths(graph, N=1), get_all_paths(graph, N=2), get_all_paths(graph, N=3)
    
    if len(dihedral_paths) == 0:
        raise Exception(f"Error: No dihedral angle detected for mol {mol}")
    
    bond_distance_indices = np.array(distance_paths, dtype=int)
    bond_angle_indices = np.array(angle_paths, dtype=int)
    dihedral_angle_indices = np.array(dihedral_paths, dtype=int)
    
    if not repeats:  # note: only taking (0, 1) vs. (1, 0); (1, 2, 3) vs. (3, 2, 1); and (1, 3, 6, 7) vs. (7, 6, 3, 1)
        bond_distance_indices = bond_distance_indices[bond_distance_indices[:, 0] < bond_distance_indices[:, 1]]
        bond_angle_indices = bond_angle_indices[bond_angle_indices[:, 0] < bond_angle_indices[:, 2]]
        dihedral_angle_indices = dihedral_angle_indices[dihedral_angle_indices[:, 1] < dihedral_angle_indices[:, 2]]

    bond_distances = np.array([rdMolTransforms.GetBondLength(conformer, int(index[0]), int(index[1])) for index in bond_distance_indices], dtype=np.float32)
    bond_angles = np.array([rdMolTransforms.GetAngleRad(conformer, int(index[0]), int(index[1]), int(index[2])) for index in bond_angle_indices], dtype=np.float32)
    dihedral_angles = np.array([rdMolTransforms.GetDihedralRad(conformer, int(index[0]), int(index[1]), int(index[2]), int(index[3])) for index in dihedral_angle_indices], dtype=np.float32)
   
    return bond_distances, bond_distance_indices, bond_angles, bond_angle_indices, dihedral_angles, dihedral_angle_indices


def get_geometric_edge_features(
    coords: torch.Tensor,
    edge_index: np.ndarray,
    D_max: float = 4.5,
    num_rbf: int = 16
) -> Tuple[torch.Tensor, torch.Tensor]:
    E_vectors = coords[edge_index[0]] - coords[edge_index[1]]
    rbf = _rbf(E_vectors.norm(dim=-1),
               D_max=D_max,
               D_count=num_rbf)

    edge_s = rbf
    edge_v = _normalize(E_vectors).unsqueeze(-2)

    edge_s, edge_v = map(torch.nan_to_num, (edge_s, edge_v))

    return edge_s, edge_v


def get_geometric_node_features(coords: torch.Tensor) -> torch.Tensor:
    orientations = ProteinGraphDataset._orientations(coords)
    node_v = orientations
    return node_v


def embed_conformer_with_all_paths(rdkit_mol3D, repeats=False, D_max=4.5, num_rbf=16):
    if isinstance(rdkit_mol3D, rdkit.Chem.rdchem.Conformer):
        mol = rdkit_mol3D.GetOwningMol()
        conformer = rdkit_mol3D
    elif isinstance(rdkit_mol3D, rdkit.Chem.rdchem.Mol):
        mol = rdkit_mol3D
        conformer = mol.GetConformer()

    # derive coordinates
    x = torch.from_numpy(conformer.GetPositions()).float()

    # build edges
    adj = rdkit.Chem.GetAdjacencyMatrix(mol)
    edge_index = adjacency_to_undirected_edge_index(adj)

    # build edge features
    bonds = []
    for b in range(int(edge_index.shape[1] / 2)):
        bond_index = edge_index[:,::2][:, b]
        bond = mol.GetBondBetweenAtoms(int(bond_index[0]), int(bond_index[1]))
        bonds.append(bond)
    edge_features = get_edge_features(bonds)
    edge_e, edge_xi = get_geometric_edge_features(
        coords=x,
        edge_index=edge_index,
        D_max=D_max,
        num_rbf=num_rbf
    )

    # build node features 
    atoms = rdkit.Chem.rdchem.Mol.GetAtoms(mol)
    atom_symbols = [atom.GetSymbol() for atom in atoms]
    node_features = get_node_features(atoms, mol)
    node_chi = get_geometric_node_features(x)
    mask = torch.isfinite(x.sum(dim=-1))
    
    bond_distances, bond_distance_indices, bond_angles, bond_angle_indices, dihedral_angles, dihedral_angle_indices = get_internal_coordinates_from_all_paths(conformer, adj, repeats=repeats)

    return atom_symbols, edge_index, edge_features, node_features, bond_distances, bond_distance_indices, bond_angles, bond_angle_indices, dihedral_angles, dihedral_angle_indices, edge_e, edge_xi, node_chi, x, mask


class SampleMapToPositives:
    def __init__(self, dataframe, is_sorted=True, include_anchor=False):
        # note: `is_sorted` vastly speeds up processing but requires that the DataFrame be sorted by `SMILES_nostereo`
        self.mapping = {}
        self.include_anchor = include_anchor
        
        for row_index, row in dataframe.iterrows():
            if is_sorted:
                subset_df = dataframe.iloc[max(row_index - 50, 0): row_index + 50, :]
                
                if self.include_anchor == False:
                    positives = set(subset_df[(subset_df.ID == row.ID) & (subset_df.index.values != row_index)].index)
                else:
                    positives = set(subset_df[(subset_df.ID == row.ID)].index)
                
                self.mapping[row_index] = positives
                
    def sample(self, i, N=1, without_replacement=True):  # sample positives
        if without_replacement:
            samples = random.sample(self.mapping[i], min(N, len(self.mapping[i])))
        else:
            samples = [random.choice(list(self.mapping[i])) for _ in range(N)]
        
        return samples
    

class SampleMapToNegatives:
    def __init__(self, dataframe, is_sorted=True):
        # note: `is_sorted` vastly speeds up processing but requires that the DataFrame be sorted by `SMILES_nostereo`
        self.mapping = {}
        for row_index, row in dataframe.iterrows():
            if is_sorted:
                negative_classes = []
                subset_df = dataframe.iloc[max(row_index - 200, 0) : row_index + 200, :]
                grouped_negatives = subset_df[(subset_df.SMILES_nostereo == row.SMILES_nostereo) & (subset_df.ID != row.ID)].groupby(by="ID", sort=False).groups.values()
                negative_classes = [set(list(group)) for group in grouped_negatives]
                self.mapping[row_index] = negative_classes
        
    def sample(self, i, N=1, without_replacement=True, stratified=True):  # sample negatives
        if without_replacement:
            if stratified:
                samples = [random.sample(self.mapping[i][j], min(len(self.mapping[i][j]), N)) for j in range(len(self.mapping[i]))]
                samples = list(itertools.chain(*samples))
            else:
                population = list(itertools.chain(*[list(self.mapping[i][j]) for j in range(len(self.mapping[i]))]))
                samples = random.sample(population, min(len(population), N))
                
        else:
            if stratified:
                samples = [[random.choice(list(population)) for _ in range(N)] for population in self.mapping[i]]
                samples = list(itertools.chain(*samples))

            else:
                population = list(itertools.chain(*[list(self.mapping[i][j]) for j in range(len(self.mapping[i]))]))
                samples = [random.choice(population) for _ in range(N)]
            
        return samples


class SingleConformerBatchSampler(torch.utils.data.sampler.BatchSampler):
    # Sample positives and negatives for each anchor, where the positives include the anchor.
    # Note: Must be used with `SampleMapToPositives` with `include_anchor=True`.
    
    # Here, `single_conformer_data_source` is a `pd.DataFrame` consisting of just 1 conformer per stereoisomer.
    # `full_data_source` is a `pd.DataFrame` consisting of all conformers for each stereoisomer.
    # Importantly, `single_conformer_data_source` must be a subset of `full_data_source`, with the original indices.
    
    def __init__(self, single_conformer_data_source, full_data_source, batch_size, drop_last, num_pos=0, num_neg=1, without_replacement=True, stratified=True):
        self.single_conformer_data_source = single_conformer_data_source
        
        self.positive_sampler = SampleMapToPositives(full_data_source, include_anchor=True)
        self.negative_sampler = SampleMapToNegatives(full_data_source)
        
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.without_replacement = without_replacement
        self.stratified = stratified
        self.num_pos = num_pos
        self.num_neg = num_neg
                
    def __iter__(self):
        groups = [[*self.positive_sampler.sample(i, N=(1 + self.num_pos), without_replacement=self.without_replacement), *self.negative_sampler.sample(i, N=self.num_neg, without_replacement=self.without_replacement, stratified=self.stratified)] for i in self.single_conformer_data_source.index.values]
        
        np.random.shuffle(groups)
        batches = [list(itertools.chain(*groups[self.batch_size * i:self.batch_size * i + self.batch_size])) for i in range(math.floor(len(groups) / self.batch_size))]
        return iter(batches)

    def __len__(self):  # note: number of batches
        return math.floor(len(self.single_conformer_data_source) / self.batch_size)  # drop the last batch if it does not contain `batch_size` anchors


class NegativeBatchSampler(torch.utils.data.sampler.BatchSampler):
    def __init__(self, data_source, batch_size, drop_last, num_neg, without_replacement=True, stratified=True):
        self.data_source = data_source
        self.negative_sampler = SampleMapToNegatives(data_source)
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.without_replacement = without_replacement
        self.stratified = stratified
        self.num_neg = num_neg
                
    def __iter__(self):
        groups = [[i, *self.negative_sampler.sample(i, N=self.num_neg, without_replacement=self.without_replacement, stratified=self.stratified)] for i in range(0, len(self.data_source))]
        np.random.shuffle(groups)
        batches = [list(itertools.chain(*groups[self.batch_size * i:self.batch_size * i + self.batch_size])) for i in range(math.floor(len(groups) / self.batch_size))]
        return iter(batches)

    def __len__(self):  # note: the number of batches
        return math.floor(len(self.data_source) / self.batch_size)  # note: drops the last batch if it does not contain `batch_size` anchors


class MaskedGraphDataset(torch_geometric.data.Dataset):
    def __init__(self, df, regression="", stereo_mask=True, mask_coordinates=False, D_max=4.5, num_rbf=16):
        super(MaskedGraphDataset, self).__init__()
        self.df = df
        self.stereo_mask = stereo_mask
        self.mask_coordinates = mask_coordinates
        self.regression = regression
        self.D_max = D_max
        self.num_rbf = num_rbf
        
    def get_all_paths(self, G, N=3):
        # adapted from: https://stackoverflow.com/questions/28095646/finding-all-paths-walks-of-given-length-in-a-networkx-graph
        def findPaths(G, u, n):
            if n==0:
                return [[u]]
            paths = [[u] + path for neighbor in G.neighbors(u) for path in findPaths(G, neighbor, n - 1) if u not in path]
            return paths
    
        allpaths = []
        for node in G:
            allpaths.extend(findPaths(G, node, N))
        return allpaths
    
    def process_mol(self, mol):
        # Get internal coordinates for the conformer, using all possible (forward) paths of length 2, 3, and 4.
        # Reverse paths (i.e., (1, 2) and (2, 1) or (1, 2, 3, 4) and (4, 3, 2, 1)) are not included when `repeats=False`.
        
        atom_symbols, edge_index, edge_features, node_features, bond_distances, bond_distance_index, bond_angles, bond_angle_index, dihedral_angles, dihedral_angle_index, edge_e, edge_xi, node_chi, x, mask = embed_conformer_with_all_paths(mol, repeats=False, D_max=self.D_max, num_rbf=self.num_rbf)
        
        bond_angles = bond_angles % (2 * np.pi)
        dihedral_angles = dihedral_angles % (2 * np.pi)
        
        data = torch_geometric.data.Data(
            x=x,
            mask=mask,
            edge_index=torch.as_tensor(edge_index, dtype=torch.long),
            h=torch.as_tensor(node_features),
            chi=node_chi,
            e=torch.cat((torch.as_tensor(edge_features), edge_e), dim=1),
            xi=edge_xi
        )
        data.bond_distances = torch.as_tensor(bond_distances)
        data.bond_distance_index = torch.as_tensor(bond_distance_index, dtype=torch.long).T
        data.bond_angles = torch.as_tensor(bond_angles)
        data.bond_angle_index = torch.as_tensor(bond_angle_index, dtype=torch.long).T
        data.dihedral_angles = torch.as_tensor(dihedral_angles)
        data.dihedral_angle_index = torch.as_tensor(dihedral_angle_index, dtype=torch.long).T

        assert data.x.shape[0] == data.h.shape[0], "Number of atom positions must match number of node features."
        
        return data
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, key):
        mol = copy.deepcopy(self.df.iloc[key].rdkit_mol_cistrans_stereo)
        
        data = self.process_mol(mol)
        
        if self.regression != "":
            # note: `self.regression` is the variable name of the supervised target in `self.df`
            data.label = torch.tensor(copy.deepcopy(self.df.iloc[key][self.regression]))

        if self.stereo_mask:
            data.h[:, -9:] = 0.0
            data.e[:, (-7 - self.num_rbf):-self.num_rbf] = 0.0

        if self.mask_coordinates:
            data.bond_distances[:] = 0.0
            data.bond_angles[:] = 0.0
            data.dihedral_angles[:] = 0.0

        return data
