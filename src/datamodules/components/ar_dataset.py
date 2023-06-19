# -------------------------------------------------------------------------------------------------------------------------------------
# Following code curated for GCPNet (https://github.com/BioinfoMachineLearning/GCPNet):
# -------------------------------------------------------------------------------------------------------------------------------------

import os
import random
import subprocess
import torch

import numpy as np

from pathlib import Path
from torch.utils.data import Dataset
from torch_geometric.data import Data
from typing import Any, Dict, List, Optional, Tuple, Union

from src.datamodules.components.covalent_helper import compute_covalent_bond_matrix
from src.datamodules.components.helper import _normalize, _rbf
from src.datamodules.components.protein_graph_dataset import ProteinGraphDataset
from src.utils.ar_utils import get_atom_features, get_residue_indices, get_seq_onehot, get_seq_from_pdb, derive_residue_local_frames
from src.utils.pylogger import get_pylogger
from src.utils.utils import TimeoutException, time_limit

from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked

patch_typeguard()  # use before @typechecked

log = get_pylogger(__name__)


SEQUENCE_CROP_LENGTH = 250  # note: a cropping constant used through the AR dataset


@typechecked
def _edge_features(
    coords: TensorType["num_nodes", 3],
    edge_index: TensorType[2, "num_edges"],
    scalar_edge_feats: TensorType["num_edges", "num_edge_scalar_features"],
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

    edge_s = torch.cat((scalar_edge_feats, rbf), dim=-1)
    edge_v = _normalize(E_vectors).unsqueeze(-2)

    edge_s, edge_v = map(torch.nan_to_num, (edge_s, edge_v))

    return edge_s, edge_v


@typechecked
def _node_features(
    coords: TensorType["num_nodes", 3]
) -> TensorType["num_nodes", "num_cat_node_vector_features", 3]:
    node_v = ProteinGraphDataset._orientations(coords)
    return node_v


class ARDataset(Dataset):
    def __init__(
        self,
        initial_pdbs: List[Dict[str, Any]],
        model_data_cache_dir: str,
        rbf_edge_dist_cutoff: float,
        num_rbf: int,
        k_min: int,
        k_max: int,
        esm_model: Optional[Any] = None,
        esm_batch_converter: Optional[Any] = None,
        python_exec_path: Optional[str] = None,
        pdbtools_dir: Optional[str] = None,
        force_process_data: bool = False,
        load_only_unprocessed_examples: bool = False,
        is_test_dataset: bool = False
    ):
        self.initial_pdbs = initial_pdbs
        self.model_data_cache_dir = model_data_cache_dir
        self.rbf_edge_dist_cutoff = rbf_edge_dist_cutoff
        self.num_rbf = num_rbf
        self.k_min = k_min
        self.k_max = k_max
        self.esm_model = esm_model
        self.esm_batch_converter = esm_batch_converter
        self.python_exec_path = python_exec_path
        self.pdbtools_dir = pdbtools_dir
        self.force_process_data = force_process_data
        self.load_only_unprocessed_examples = load_only_unprocessed_examples
        self.is_test_dataset = is_test_dataset
        self.num_pdbs = len(self.initial_pdbs)

        os.makedirs(self.model_data_cache_dir, exist_ok=True)

        # filter the dataset down to only unprocessed examples
        if load_only_unprocessed_examples:
            unprocessed_initial_pdbs = []
            for idx in range(self.num_pdbs):
                initial_pdb_filepath = self.initial_pdbs[idx]["initial_pdb"]
                initial_data_filepath = Path(self.model_data_cache_dir) / Path(Path(initial_pdb_filepath).stem + ".pt")
                if not os.path.exists(str(initial_data_filepath)):
                    unprocessed_initial_pdbs.append(self.initial_pdbs[idx])
            self.initial_pdbs = unprocessed_initial_pdbs
            self.num_pdbs = len(self.initial_pdbs)

    def __len__(self):
        return self.num_pdbs

    @typechecked
    def _get_sequence_features(self, pdb_filepath: str, cropped_sequence: Optional[str] = None) -> Tuple[Dict[str, Any], int]:
        sequence = cropped_sequence if cropped_sequence is not None else get_seq_from_pdb(pdb_filepath)
        node_feats = {
            "residue_sequence": sequence,
            "residue_onehot": get_seq_onehot(sequence),
            "residue_indices": get_residue_indices(sequence)
        }
        return node_feats, len(sequence)

    @typechecked
    def _get_structural_features(self, pdb_filepath: str, true_pdb: str, seq_len: int) -> Dict[str, Dict[str, Any]]:
        atom_embeddings, atom_pos_list, num_atoms_per_residue_list, ca_atom_pos_list, residue_to_atom_names_mapping_list = get_atom_features(
            pdb_filepath, true_pdb, (1, seq_len)
        )
        node_feats = {
            "atom_data": {
                "atom_embeddings": atom_embeddings,
                "atom_pos_list": atom_pos_list,
                "num_atoms_per_residue_list": num_atoms_per_residue_list,
                "ca_atom_pos_list": ca_atom_pos_list,
                "residue_to_atom_names_mapping_list": residue_to_atom_names_mapping_list
            }
        }
        return node_feats

    @staticmethod
    @typechecked
    def compute_protein_atom_edge_index_and_features(
        atom_pos: TensorType["batch_size", "num_atoms", 1, 3],
        atom_pair_feats: TensorType["batch_size", "num_atoms", "num_atoms", "num_pair_feats"],
        atom_indices: TensorType["batch_size", "num_atoms"],
        k_min: int = 12,
        k_max: int = 18,
        upper_eps: float = 999.9
    ) -> Tuple[
        TensorType[2, "batch_num_edges"],
        TensorType["batch_num_edges", "num_pair_feats"]
    ]:
        assert 0 < k_min <= k_max, "Atoms' minimum number of edges must be greater than zero and not greater than atoms' maximum number of edges."
        batch_size, num_atoms = atom_pos.shape[:2]

        # compute distance map using Ca atoms' current 3D positions
        dist_mat = torch.cdist(atom_pos[:, :, 0, :], atom_pos[:, :, 0, :]) + torch.eye(num_atoms).unsqueeze(0) * upper_eps  # shape: [`batch_size`, `num_atoms`, `num_atoms`]

        # consider sequence separation as a distinct edge connectivity criterion
        sep = atom_indices[:, None, :] - atom_indices[:, :, None]
        sep = sep.abs() + torch.eye(num_atoms).unsqueeze(0) * upper_eps

        # select `k_max` neighbors for each atom
        _, edge_index = torch.topk(dist_mat, min(k_max, num_atoms), largest=False)  # note: shape: [`batch_size`, `num_atoms`, `k_max`]
        topk_matrix = torch.zeros((batch_size, num_atoms, num_atoms))
        topk_matrix.scatter_(2, edge_index, 1.0)
        cond = torch.logical_or(topk_matrix > 0.0, sep < k_min)
        b, i, j = torch.where(cond)

        src = b * num_atoms + i
        dst = b * num_atoms + j

        # finalize sparse edge features
        edge_index = torch.stack((src, dst))
        sparse_atom_pair_feats = atom_pair_feats[b, i, j]

        return edge_index, sparse_atom_pair_feats
    
    @staticmethod
    @typechecked
    def finalize_graph_features_within_data(data: Data, rbf_edge_dist_cutoff: float, num_rbf: int) -> Data:
        # collate atom features from residue features to reduce storage requirements
        residue_atom_onehot = []
        residue_atom_esm = []
        for residue_idx, num_atoms_in_residue_i in enumerate(data.num_atoms_per_residue.tolist()):
            residue_onehot = torch.tile(data.residue_onehot[residue_idx], (num_atoms_in_residue_i, 1))
            residue_esm = torch.tile(data.residue_esm[residue_idx], (num_atoms_in_residue_i, 1))
            residue_atom_onehot.append(residue_onehot)
            residue_atom_esm.append(residue_esm)
        residue_atom_onehot = torch.cat((torch.vstack(residue_atom_onehot), data.atom_onehot), dim=-1)
        residue_atom_esm = torch.vstack(residue_atom_esm)
        assert residue_atom_esm.shape[0] == data.initial_atom_pos_disp.shape[0], "Number of atoms must match between ESM embeddings and 3D positions."

        # build geometric graph features
        h = torch.cat((residue_atom_onehot, residue_atom_esm), dim=-1)
        chi = _node_features(data.initial_atom_pos_disp)
        e, xi = _edge_features(
            data.initial_atom_pos_disp,
            data.edge_index,
            scalar_edge_feats=data.initial_atom_pair_feats,
            D_max=rbf_edge_dist_cutoff,
            num_rbf=num_rbf
        )

        # standardize graph features
        standardized_data = Data(
            h=h,
            chi=chi,
            e=e,
            xi=xi,
            x=data.initial_atom_pos,
            ca_x=data.initial_ca_atom_pos,
            label=data.true_atom_pos,
            edge_index=data.edge_index,
            num_atoms_per_residue=data.num_atoms_per_residue,
            residue_to_atom_names_mapping=[data.residue_to_atom_names_mapping],
            initial_pdb_filepath=data.initial_pdb_filepath,
            true_pdb_filepath=data.true_pdb_filepath
        )

        return standardized_data
    
    @staticmethod
    @typechecked
    def crop_pdb_file(
        input_pdb_filepath: str,
        output_pdb_filepath: str,
        crop_index: int,
        full_sequence_length: int,
        new_pdb_chain_id: str = "A",
        new_start_index: int = 1,
        sequence_crop_length: int = SEQUENCE_CROP_LENGTH,
        sequence_range: Optional[Tuple[int, int]] = None,
        python_exec_path: Optional[str] = None,
        pdbtools_dir: Optional[str] = None
    ) -> Tuple[int, int]:
        """
        Crop the input PDB file to its output file counterpart according to a crop index,
        full PDB file sequence length, and a maximum sequence length for a given crop index.
        Return the start and end indices of the (new) PDB file's sequence length.
        """
        if sequence_range is not None:
            # e.g., reuse the start and end indices when creating a random crop of the initial PDB input file
            start_index, end_index = sequence_range[0], sequence_range[1]
        else:
            start_index = crop_index * sequence_crop_length
            end_index = start_index + sequence_crop_length
            if end_index > full_sequence_length:
                # introduce a random, sequence-contiguous crop of sequence length `sequence_crop_length` to ensure all crops are of the same length
                start_index = random.choice([i for i in range(0, full_sequence_length - sequence_crop_length)])
                end_index = start_index + sequence_crop_length
        assert (end_index - start_index) == sequence_crop_length, "Each crop must be of sequence length `sequence_crop_length`."
        assert (start_index < end_index <= full_sequence_length), "Sequence range indices must be consecutive and within the maximum sequence length."
        if pdbtools_dir is not None:
            assert python_exec_path is not None, "Path to a Python executable (i.e., binary) file must be provided to run individual scripts within `pdb_tools`."
        pdb_selres_cmd = f"{python_exec_path} {os.path.join(pdbtools_dir, 'pdb_selres.py')}" if pdbtools_dir is not None else "pdb_selres"
        pdb_chain_cmd = f"{python_exec_path} {os.path.join(pdbtools_dir, 'pdb_chain.py')}" if pdbtools_dir is not None else "pdb_chain"
        pdb_reres_cmd = f"{python_exec_path} {os.path.join(pdbtools_dir, 'pdb_reres.py')}" if pdbtools_dir is not None else "pdb_reres"
        crop_cmd = f"{pdb_selres_cmd} -{start_index + 1}:{end_index} {input_pdb_filepath} | {pdb_chain_cmd} -{new_pdb_chain_id} | {pdb_reres_cmd} -{new_start_index} > {output_pdb_filepath}"
        try:
            subprocess.run(args=crop_cmd, shell=True)
        except Exception as e:
            log.error(f"Unable to crop PDB file {input_pdb_filepath} via pdb-tools. Please investigate the validity of this input file.")
            raise e
        return (start_index, end_index)
    
    @typechecked
    def _get_data(self, idx: int) -> Data:
        initial_pdb_filepath = self.initial_pdbs[idx]["initial_pdb"]
        initial_pdb_is_cropped = "_" in Path(initial_pdb_filepath).stem
        initial_data_filepath = Path(self.model_data_cache_dir) / Path(Path(initial_pdb_filepath).stem + ".pt")

        # directly load preprocessed graph topology and features (if available)
        initial_data_exists = os.path.exists(str(initial_data_filepath)) and not self.force_process_data
        if initial_data_exists:
            try:
                data = torch.load(str(initial_data_filepath))
            except:
                # note: this means we need to re-process the input data
                initial_data_exists = False

        if not initial_data_exists:
            # otherwise, start by identifying filepath of true PDB
            if self.is_test_dataset:
                true_pdb_filepath = initial_pdb_filepath
            else:
                true_pdb_filepath = self.initial_pdbs[idx]["true_pdb"]
            true_pdb_is_cropped = "_" in Path(true_pdb_filepath).stem

            # handle sequence cropping of input PDB files (up to a length of `SEQUENCE_CROP_LENGTH=250`)
            cropped_sequence = None
            if initial_pdb_is_cropped:
                assert true_pdb_is_cropped, "Both initial and true PDB files must be cropped in accordance with one another."
                input_initial_pdb_filepath = str(Path(initial_pdb_filepath).parent / Path(Path(initial_pdb_filepath).stem.split("_")[0] + ".pdb"))
                input_true_pdb_filepath = str(Path(true_pdb_filepath).parent / Path(Path(true_pdb_filepath).stem.split("_")[0] + ".pdb"))
                crop_index = int(Path(initial_pdb_filepath).stem.split("_")[1])
                # work around the fact that `pdb-tools` does not preserve the integrity of an input PDB file's sequence entry
                seq_node_feats, seq_len = self._get_sequence_features(input_initial_pdb_filepath)
                sequence_range = self.crop_pdb_file(
                    input_pdb_filepath=input_initial_pdb_filepath,
                    output_pdb_filepath=initial_pdb_filepath,
                    crop_index=crop_index,
                    full_sequence_length=seq_len,
                    sequence_crop_length=SEQUENCE_CROP_LENGTH,
                    python_exec_path=self.python_exec_path,
                    pdbtools_dir=self.pdbtools_dir
                )
                _, _ = self.crop_pdb_file(
                    input_pdb_filepath=input_true_pdb_filepath,
                    output_pdb_filepath=true_pdb_filepath,
                    crop_index=crop_index,
                    full_sequence_length=seq_len,
                    sequence_crop_length=SEQUENCE_CROP_LENGTH,
                    sequence_range=sequence_range,
                    python_exec_path=self.python_exec_path,
                    pdbtools_dir=self.pdbtools_dir
                )
                cropped_sequence = seq_node_feats["residue_sequence"][sequence_range[0]:sequence_range[1]]  # prepare to skip conventional sequence parsing
                assert os.path.exists(initial_pdb_filepath) and os.path.exists(true_pdb_filepath), "Both initial and true PDB files must exist after cropping."

            # otherwise, collect all sequence-based, structural, and chemical features
            seq_node_feats, seq_len = self._get_sequence_features(initial_pdb_filepath, cropped_sequence=cropped_sequence)
            initial_covalent_bond_mat = compute_covalent_bond_matrix(initial_pdb_filepath)
            initial_struct_node_feats = self._get_structural_features(initial_pdb_filepath, true_pdb_filepath, seq_len)
            true_struct_node_feats = self._get_structural_features(true_pdb_filepath, true_pdb_filepath, seq_len)

            # add ESMFold sequence embeddings as scalar atom features that are shared between atoms of the same residue
            # note: assumes only a single chain's sequence is available
            residue_sequences = [("", seq_node_feats["residue_sequence"])]
            batch_tokens = self.esm_batch_converter(residue_sequences)[2]
            with torch.inference_mode():
                results = self.esm_model(batch_tokens, repr_layers=[self.esm_model.num_layers])
            token_representations = results["representations"][self.esm_model.num_layers].cpu()
            esm_residue_representations = []
            for i, (_, protein_sequence) in enumerate(residue_sequences):
                representations = token_representations[i, 1: len(protein_sequence) + 1]
                esm_residue_representations.append(representations)
            esm_residue_representations = torch.cat(esm_residue_representations, dim=0)

            residue_onehot = []
            atom_onehot = []
            atom_pos_displ_list = []
            num_residues = seq_node_feats["residue_onehot"].shape[0]
            for i in range(num_residues):
                atom_embeddings = initial_struct_node_feats["atom_data"]["atom_embeddings"][i]
                residue_onehot.append(seq_node_feats["residue_onehot"][i])
                atom_onehot.append(atom_embeddings[:, 0:37])
                atom_pos_displ_list.append(atom_embeddings[:, 37:40])
            residue_onehot = np.vstack(residue_onehot)
            atom_onehot = np.vstack(atom_onehot)
            initial_atom_pos_disp = np.vstack(atom_pos_displ_list)

            initial_atom_pos_list = initial_struct_node_feats["atom_data"]["atom_pos_list"]
            initial_atom_pos = np.vstack(initial_atom_pos_list)
            true_atom_pos_list = true_struct_node_feats["atom_data"]["atom_pos_list"]
            true_atom_pos = np.vstack(true_atom_pos_list)

            num_atoms_per_residue = initial_struct_node_feats["atom_data"]["num_atoms_per_residue_list"]
            residue_to_atom_names_mapping = initial_struct_node_feats["atom_data"]["residue_to_atom_names_mapping_list"]
            ca_atom_pos_list = initial_struct_node_feats["atom_data"]["ca_atom_pos_list"]
            initial_ca_atom_pos = np.vstack(ca_atom_pos_list)

            p, q, k, t = derive_residue_local_frames(initial_pdb_filepath, initial_atom_pos_disp, num_atoms_per_residue)
            initial_atom_frame_pairs = np.concatenate([p, q, k, t], axis=-1)

            # build `data` with initial features
            data = Data(
                residue_onehot=torch.from_numpy(residue_onehot).float(),
                atom_onehot=torch.from_numpy(atom_onehot).float(),
                residue_esm=esm_residue_representations,
                initial_atom_pos=torch.from_numpy(initial_atom_pos).float(),
                initial_atom_pos_disp=torch.from_numpy(initial_atom_pos_disp).float(),
                initial_ca_atom_pos=torch.from_numpy(initial_ca_atom_pos),
                initial_pdb_filepath=initial_pdb_filepath,
                true_pdb_filepath=true_pdb_filepath,
                true_atom_pos=torch.from_numpy(true_atom_pos).float(),
                num_atoms_per_residue=torch.tensor(num_atoms_per_residue),
                residue_to_atom_names_mapping=residue_to_atom_names_mapping
            )

            # install graph topology and "sparsify" dense edge features according to it
            num_atoms = data.initial_atom_pos_disp.shape[0]
            atom_pos = data.initial_atom_pos_disp.view(1, num_atoms, 1, 3)
            initial_atom_frame_pairs = (
                # try to maintain parity with AR dataloading implementation via the following permutations
                _normalize(torch.from_numpy(initial_atom_frame_pairs).float()).view(1, num_atoms, num_atoms, 12).permute(0, 3, 1, 2).permute(0, 2, 3, 1)
            )
            initial_covalent_bond_mat = (
                torch.from_numpy(initial_covalent_bond_mat).float().view(1, num_atoms, num_atoms, 1)
            )
            atom_pair_feats = torch.cat((initial_atom_frame_pairs, initial_covalent_bond_mat), dim=-1)
            atom_indices = torch.arange(num_atoms).long().unsqueeze(0)
            edge_index, atom_pair_feats = self.compute_protein_atom_edge_index_and_features(
                atom_pos=atom_pos,
                atom_pair_feats=atom_pair_feats,
                atom_indices=atom_indices,
                k_min=self.k_min,
                k_max=self.k_max
            )
            data["edge_index"] = edge_index
            data["initial_atom_pair_feats"] = atom_pair_feats

            # cache processed topology and features
            torch.save(data, str(initial_data_filepath))

        # finalize graph features according to current graph topology and model specifications
        data = self.finalize_graph_features_within_data(data, rbf_edge_dist_cutoff=self.rbf_edge_dist_cutoff, num_rbf=self.num_rbf)
        return data

    @typechecked
    def __getitem__(self, idx: int, retrieval_time_limit_in_seconds: int = 1000) -> Data:
        if self.load_only_unprocessed_examples:
            try:
                with time_limit(retrieval_time_limit_in_seconds):
                    return self._get_data(idx)
            except TimeoutException:
                pdb_filepath = self.initial_pdbs[idx]["initial_pdb"]
                log.info(f"Cannot retrieve (or process) protein {pdb_filepath}. Removing...")
                del self.initial_pdbs[idx]
                return self._get_data(idx)
        else:
            return self._get_data(idx)
