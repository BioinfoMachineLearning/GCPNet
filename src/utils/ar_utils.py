# -------------------------------------------------------------------------------------------------------------------------------------
# Following code curated for GCPNet (https://github.com/BioinfoMachineLearning/GCPNet):
# -------------------------------------------------------------------------------------------------------------------------------------

import Bio.PDB
import warnings

import numpy as np

from Bio import SeqIO
from collections import defaultdict
from typing import DefaultDict, List, Optional, Tuple

from torchtyping import patch_typeguard
from typeguard import typechecked

patch_typeguard()  # use before @typechecked

RESIDUE_TYPES = ["A", "C", "D", "E", "F", "G", "H", "I", "K", "L",
                 "M", "N", "P", "Q", "R", "S", "T", "V", "W", "Y", "X"]

warnings.filterwarnings("ignore")

pdb_parser = Bio.PDB.PDBParser(QUIET=True)


RESIDUE_NAME_TO_ATOM_NAMES_MAPPING = {
    "ALA": {"atoms": ["N", "CA", "C", "O", "CB"]},
    "ARG": {"atoms": ["N", "CA", "C", "O", "CB", "CG", "CD", "NE", "CZ", "NH1", "NH2"]},
    "ASN": {"atoms": ["N", "CA", "C", "O", "CB", "CG", "OD1", "ND2"]},
    "ASP": {"atoms": ["N", "CA", "C", "O", "CB", "CG", "OD1", "OD2"]},
    "CYS": {"atoms": ["N", "CA", "C", "O", "CB", "SG"]},
    "GLN": {"atoms": ["N", "CA", "C", "O", "CB", "CG", "CD", "OE1", "NE2"]},
    "GLU": {"atoms": ["N", "CA", "C", "O", "CB", "CG", "CD", "OE1", "OE2"]},
    "GLY": {"atoms": ["N", "CA", "C", "O"]},
    "HIS": {"atoms": ["N", "CA", "C", "O", "CB", "CG", "ND1", "CD2", "CE1", "NE2"]},
    "ILE": {"atoms": ["N", "CA", "C", "O", "CB", "CG1", "CG2", "CD1"]},
    "LEU": {"atoms": ["N", "CA", "C", "O", "CB", "CG", "CD1", "CD2"]},
    "LYS": {"atoms": ["N", "CA", "C", "O", "CB", "CG", "CD", "CE", "NZ"]},
    "MET": {"atoms": ["N", "CA", "C", "O", "CB", "CG", "SD", "CE"]},
    "PHE": {"atoms": ["N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ"]},
    "PRO": {"atoms": ["N", "CA", "C", "O", "CB", "CG", "CD"]},
    "SER": {"atoms": ["N", "CA", "C", "O", "CB", "OG"]},
    "THR": {"atoms": ["N", "CA", "C", "O", "CB", "OG1", "CG2"]},
    "TRP": {"atoms": ["N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "NE1", "CE2", "CE3", "CZ2", "CZ3", "CH2"]},
    "TYR": {"atoms": ["N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ", "OH"]},
    "VAL": {"atoms": ["N", "CA", "C", "O", "CB", "CG1", "CG2"]}
}
#ATOM_SYMBOL_TO_INDEX_MAPPING = {"C":0, "N":1, "O":2, "S":3}
ATOM_SYMBOL_TO_INDEX_MAPPING = {"N": 0, "CA": 1, "C": 2, "O": 3, "CB": 4, "OG": 5, "CG": 6, "CD1": 7, "CD2": 8, "CE1": 9, "CE2": 10, "CZ": 11, "OD1": 12, "ND2": 13, "CG1": 14, "CG2": 15, "CD": 16, "CE": 17, "NZ": 18,
                                "OD2": 19, "OE1": 20, "NE2": 21, "OE2": 22, "OH": 23, "NE": 24, "NH1": 25, "NH2": 26, "OG1": 27, "SD": 28, "ND1": 29, "SG": 30, "NE1": 31, "CE3": 32, "CZ2": 33, "CZ3": 34, "CH2": 35, "OXT": 36}


@typechecked
def get_seq_from_pdb(pdb_filepath: str) -> str:
    for record in SeqIO.parse(pdb_filepath, "pdb-atom"):
        return str(record.seq).upper()


@typechecked
def get_seq_onehot(seq: str) -> np.ndarray:
    seq_onehot = np.zeros((len(seq), len(RESIDUE_TYPES)))
    for i, res in enumerate(seq.upper()):
        if res not in RESIDUE_TYPES:
            res = "X"
        seq_onehot[i, RESIDUE_TYPES.index(res)] = 1
    return seq_onehot


@typechecked
def get_residue_indices(seq: str) -> np.ndarray:
    seq_len = len(seq)
    res_idx = np.linspace(0, 1, num=seq_len).reshape(seq_len, -1)
    return res_idx


@typechecked
def get_one_hot(targets: np.ndarray, num_classes: int) -> np.ndarray:
    res = np.eye(num_classes)[np.array(targets).reshape(-1)]
    return res.reshape(list(targets.shape) + [num_classes])


@typechecked
def align_lists_by_biopython_residue_id(list1: list, list2: list) -> Tuple[list, list, list, list]:
    # Create dictionaries to store objects by their id properties
    dict1 = {obj.id[1]: obj for obj in list1}
    dict2 = {obj.id[1]: obj for obj in list2}

    # Get the common ids from both lists
    common_ids = set(dict1.keys()).intersection(dict2.keys())

    # Create aligned lists using the common ids
    aligned_list1 = [dict1[id] for id in common_ids]
    aligned_list2 = [dict2[id] for id in common_ids]

    # Get the ids that are not in both lists
    non_intersected_ids_list1 = set(dict1.keys()) - common_ids
    non_intersected_ids_list2 = set(dict2.keys()) - common_ids

    # Create lists of non-intersected elements using the non_intersected_ids
    non_intersected_list1 = [dict1[id] for id in non_intersected_ids_list1]
    non_intersected_list2 = [dict2[id] for id in non_intersected_ids_list2]

    assert len(aligned_list1) == len(aligned_list2), "After alignment, both residue lists must contain the same number of residues."
    return aligned_list1, aligned_list2, non_intersected_list1, non_intersected_list2


@typechecked
def get_atom_features(
    initial_pdb_filepath: str,
    true_pdb_filepath: str,
    res_range: Tuple[int, int],
    model_id: int = 0,
    chain_id: int = 0,
    testing: bool = False,
    unaligned_residue_indices_to_drop: Optional[List[int]] = None
) -> Tuple[List[np.ndarray], List[np.ndarray], List[int], List[np.ndarray], DefaultDict[str, List[str]], List[int]]:
    """Generate atom embeddings using the coordinates and type associated with each amino acid residue."""
    structure_true = pdb_parser.get_structure("native", true_pdb_filepath)
    model_true = structure_true.get_list()[model_id]
    chain_true = model_true.get_list()[chain_id]
    residue_list_true = chain_true.get_list()

    structure = pdb_parser.get_structure("tmp", initial_pdb_filepath)
    model = structure.get_list()[model_id]
    chain = model.get_list()[chain_id]
    residue_list = chain.get_list()

    num_atoms_per_residue_list = []
    ca_atom_pos_list = []
    residue_to_atom_names_mapping_list = defaultdict(list)

    res_num = res_range[1] - res_range[0] + 1
    unaligned_residue_list, unaligned_residue_list_true = [], []
    unaligned_residue_indices_to_drop_ = unaligned_residue_indices_to_drop if unaligned_residue_indices_to_drop is not None else []
    if testing:
        # note: assumes the decoy and ground-truth structures may have their residues out of sequence
        # alignment with each other, so we need to correct for this using BioPython's IDs for each residue
        residue_list, residue_list_true, unaligned_residue_list, unaligned_residue_list_true = align_lists_by_biopython_residue_id(residue_list, residue_list_true)
        # re-sort by `resseq`
        residue_list.sort(key=lambda x: x.id[1]), residue_list_true.sort(key=lambda x: x.id[1])
        unaligned_residue_list.sort(key=lambda x: x.id[1]), unaligned_residue_list_true.sort(key=lambda x: x.id[1])
        res_range = (residue_list[0].id[1], residue_list[-1].id[1])
        res_num = len(residue_list)
        unaligned_residue_indices_to_drop_ = unaligned_residue_indices_to_drop if unaligned_residue_indices_to_drop is not None else [res.id[1] - 1 for res in unaligned_residue_list]

    atom_embeddings = [-1 for _ in range(res_num)]
    atom_pos_list = [-1 for _ in range(res_num)]

    for residue, residue_true in zip(residue_list, residue_list_true):
        if residue_true.id[1] < res_range[0] or residue_true.id[1] > res_range[1]:
            # note: assumes the ground-truth structure contains more residues than the decoy structure
            continue
        atom_pos, onehot = [], []
        _resname = residue_true.get_resname() if residue_true.get_resname() in RESIDUE_NAME_TO_ATOM_NAMES_MAPPING else "GLY"
        # account for the fact that certain CASP ground-truth structures in the test datasets
        # may have gapped (missing) residues in the middle of their sequences
        num_decoy_gap_residues_passed = len([res for res in unaligned_residue_list if res.id[1] < residue.id[1] and res.id[1] > res_range[0]])
        num_true_gap_residues_passed = len([res for res in unaligned_residue_list_true if res.id[1] < residue.id[1] and res.id[1] > res_range[0]])
        if unaligned_residue_indices_to_drop is not None:
            for res_index in unaligned_residue_indices_to_drop:
                res_id = res_index + 1
                if res_id < residue.id[1] and res_id > res_range[0]:
                    num_decoy_gap_residues_passed += 1
        num_gap_residues_passed = max(num_decoy_gap_residues_passed, num_true_gap_residues_passed)
        for _atom in RESIDUE_NAME_TO_ATOM_NAMES_MAPPING[_resname]["atoms"]:
            if residue_true.has_id(_atom):
                residue_to_atom_names_mapping_list[_resname + str(residue.id[1] - res_range[0] - num_gap_residues_passed)].append(_atom)
                atom_pos.append(residue[_atom].coord)
                _onehot = np.zeros(len(ATOM_SYMBOL_TO_INDEX_MAPPING))
                _onehot[ATOM_SYMBOL_TO_INDEX_MAPPING[_atom]] = 1
                onehot.append(_onehot)
        num_atoms_per_residue_list.append(len(atom_pos))
        ca_atom_pos = residue["CA"].coord
        ca_atom_pos_list.append(ca_atom_pos)
        atom_embedding = np.concatenate((np.array(onehot), np.array(atom_pos) - ca_atom_pos[None, :]), axis=1)
        atom_embeddings[residue.id[1] - res_range[0] - num_gap_residues_passed] = atom_embedding.astype(np.float16)
        atom_pos_list[residue.id[1] - res_range[0] - num_gap_residues_passed] = np.array(atom_pos).astype(np.float16)

    atom_nums = np.zeros((res_num))
    for i, _item in enumerate(atom_embeddings):
        if not np.isscalar(_item):
            atom_nums[i] = _item.shape[0]

    unaligned_residue_indices_to_drop = unaligned_residue_indices_to_drop_

    return atom_embeddings, atom_pos_list, num_atoms_per_residue_list, ca_atom_pos_list, residue_to_atom_names_mapping_list, unaligned_residue_indices_to_drop


@typechecked
def derive_residue_local_frames(
    pdb_filepath: str,
    atom_pos: np.ndarray,
    num_atoms_per_residue: List[int],
    unaligned_residue_indices_to_drop: Optional[List[int]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # load pdb file
    structure = pdb_parser.get_structure("tmp_struct", pdb_filepath)
    residues = [_ for _ in structure.get_residues()]

    if unaligned_residue_indices_to_drop is not None:
        residues = [_ for idx, _ in enumerate(residues) if idx not in unaligned_residue_indices_to_drop]

    # construct a local frame for each PDB residue
    pdict = {
        "N": np.stack([np.array(residue["N"].coord) for residue in residues]),
        "Ca": np.stack([np.array(residue["CA"].coord) for residue in residues]),
        "C": np.stack([np.array(residue["C"].coord) for residue in residues])
    }

    # recreate Cb atoms' positions given the positions of residues' N, Ca, and C atoms
    ca = -0.58273431
    cb = 0.56802827
    cc = -0.54067466

    b = pdict["Ca"] - pdict["N"]
    c = pdict["C"] - pdict["Ca"]
    a = np.cross(b, c)
    pdict["Cb"] = ca * a + cb * b + cc * c

    # build each residue's local frame
    z = pdict["Cb"] - pdict["Ca"]
    z /= np.linalg.norm(z, axis=-1)[:, None]
    x = np.cross(pdict["Ca"] - pdict["N"], z)
    x /= np.linalg.norm(x, axis=-1)[:, None]
    y = np.cross(z, x)
    y /= np.linalg.norm(y, axis=-1)[:, None]

    xyz = np.stack([x, y, z])

    pdict["local_frame"] = np.transpose(xyz, [1, 0, 2])

    start, end, j = 0, 0, 0
    atom_idx = [-1 for _ in range(atom_pos.shape[0])]
    for i in range(len(num_atoms_per_residue)):
        start = end
        end += num_atoms_per_residue[i]
        atom_idx[start:end] = [j] * num_atoms_per_residue[i]
        j = j + 1

    p = np.zeros((atom_pos.shape[0], atom_pos.shape[0], 3))
    q = np.zeros((atom_pos.shape[0], atom_pos.shape[0], 3))
    k = np.zeros((atom_pos.shape[0], atom_pos.shape[0], 3))
    t = np.zeros((atom_pos.shape[0], atom_pos.shape[0], 3))
    for i in range(atom_pos.shape[0]):
        res_idx = atom_idx[i]
        for j in range(atom_pos.shape[0]):
            p[i, j, :] = np.matmul(pdict["local_frame"][res_idx], atom_pos[j] - atom_pos[i])
            q[i, j, :] = np.matmul(pdict["local_frame"][atom_idx[i]], pdict["local_frame"][atom_idx[j]][0])
            k[i, j, :] = np.matmul(pdict["local_frame"][atom_idx[i]], pdict["local_frame"][atom_idx[j]][1])
            t[i, j, :] = np.matmul(pdict["local_frame"][atom_idx[i]], pdict["local_frame"][atom_idx[j]][2])

    return p, q, k, t
