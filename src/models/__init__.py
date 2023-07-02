# -------------------------------------------------------------------------------------------------------------------------------------
# Following code curated for GCPNet (https://github.com/BioinfoMachineLearning/GCPNet):
# -------------------------------------------------------------------------------------------------------------------------------------


import subprocess
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from biopandas.pdb import PandasPdb
from torch_scatter import scatter
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

from atom3d.util import metrics
from collections import defaultdict
from functools import partial
from torchtyping import patch_typeguard
from typeguard import typechecked

from src import utils

from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked

patch_typeguard()  # use before @typechecked

HALT_FILE_EXTENSION = "done"

RELAX_MAX_ITERATIONS = 1
RELAX_ENERGY_TOLERANCE = 2.39
RELAX_STIFFNESS = 10.0
RELAX_EXCLUDE_RESIDUES = []
RELAX_MAX_OUTER_ITERATIONS = 1


log = utils.get_pylogger(__name__)


@typechecked
def get_nonlinearity(nonlinearity: Optional[str] = None, slope: float = 1e-2, return_functional: bool = False) -> Any:
    nonlinearity = nonlinearity if nonlinearity is None else nonlinearity.lower().strip()
    if nonlinearity == "relu":
        return F.relu if return_functional else nn.ReLU()
    elif nonlinearity == "leakyrelu":
        return partial(F.leaky_relu, negative_slope=slope) if return_functional else nn.LeakyReLU(negative_slope=slope)
    elif nonlinearity == "selu":
        return partial(F.selu) if return_functional else nn.SELU()
    elif nonlinearity == "silu":
        return partial(F.silu) if return_functional else nn.SiLU()
    elif nonlinearity == "sigmoid":
        return torch.sigmoid if return_functional else nn.Sigmoid()
    elif nonlinearity is None:
        return nn.Identity()
    else:
        raise NotImplementedError(f"The nonlinearity {nonlinearity} is currently not implemented.")


@typechecked
def get_comparison_function(comparison: Optional[str] = None, return_functional: bool = False) -> Any:
    comparison = comparison if comparison is None else comparison.lower().strip()
    if comparison == "cosine":
        return F.cosine_similarity if return_functional else nn.CosineSimilarity()
    elif comparison is None:
        return F.cosine_similarity if return_functional else nn.CosineSimilarity()
    else:
        raise NotImplementedError(f"The comparison function {comparison} is currently not implemented.")


@typechecked
def randn_invariant_features(n1: int, n2: int, dims: Tuple[int, int], device: Union[str, torch.device] = "cpu"):
    """
    Returns random invariant feature tuples (s, s) drawn elementwise from a normal distribution.

    :param n1: first number of data points
    :param n2: second number of data points
    :param dims: tuple of dimensions (n1_scalar, n2_scalar)

    :return: (s1, s2) with s1.shape = (n1, n1_scalar) and
             s2.shape = (n2, n2_scalar)
    """
    return torch.randn(n1, dims[0], device=device), \
        torch.randn(n2, dims[1], device=device)


@typechecked
def randn_equivariant_features(n1: int, n2: int, dims: Tuple[int, int], device: Union[str, torch.device] = "cpu"):
    """
    Returns random equivariant feature tuples (V, V) drawn elementwise from a normal distribution.

    :param n1: first number of data points
    :param n2: second number of data points
    :param dims: tuple of dimensions (n1_vector, n2_vector)

    :return: (V1, V2) with V1.shape = (n1, n1_vector, 3) and
             V2.shape = (n2, n2_vector, 3)
    """
    return torch.randn(n1, dims[0], 3, device=device), \
        torch.randn(n2, dims[1], 3, device=device)


@typechecked
def randn(n: int, dims: Tuple[int, int], device: Union[str, torch.device] = "cpu"):
    """
    Returns random tuples (s, V) drawn elementwise from a normal distribution.

    :param n: number of data points
    :param dims: tuple of dimensions (n_scalar, n_vector)

    :return: (s, V) with s.shape = (n, n_scalar) and
             V.shape = (n, n_vector, 3)
    """
    return torch.randn(n, dims[0], device=device), \
        torch.randn(n, dims[1], 3, device=device)


@typechecked
def get_psr_corr_coefs() -> Dict[str, Callable]:
    """
    From https://github.com/drorlab/gvp-pytorch
    """
    @typechecked
    def _corr(
        metric: Callable,
        labels: List[np.float32],
        preds: List[np.float32],
        ids: Optional[List[str]] = None,
        glob: bool = True
    ) -> np.float64:
        if glob:
            return metric(labels, preds)
        _labels, _preds = defaultdict(list), defaultdict(list)
        for _t, _p, _id in zip(labels, preds, ids):
            _labels[_id].append(_t)
            _preds[_id].append(_p)
        return np.mean([metric(_labels[_id], _preds[_id]) for _id in _labels])

    corr_coefs = {
        "PearsonCorrCoef": partial(_corr, metrics.pearson),
        "SpearmanCorrCoef": partial(_corr, metrics.spearman),
        "KendallTau": partial(_corr, metrics.kendall),
    }
    local_corr_coefs = {
        f"Local{k}": partial(v, glob=False) for k, v in corr_coefs.items()
    }
    global_corr_coefs = {
        f"Global{k}": partial(v, glob=True) for k, v in corr_coefs.items()
    }

    return {**local_corr_coefs, **global_corr_coefs}


@typechecked
def get_grad_norm(
    parameters: Union[torch.Tensor, Iterable[torch.Tensor]],
    device: Union[torch.device, str],
    norm_type: float = 2.0
) -> torch.Tensor:
    """
    Adapted from: https://pytorch.org/docs/stable/_modules/torch/nn/utils/clip_grad.html#clip_grad_norm_
    """
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]

    norm_type = float(norm_type)

    if len(parameters) == 0:
        return torch.tensor(0.0, device=device)

    total_norm = torch.norm(
        torch.stack(
            [torch.norm(p.grad.detach(), norm_type) for p in parameters]
        ),
        p=norm_type
    )
    return total_norm


@typechecked
def convert_idx_from_batch_local_to_global(
    idx: TensorType["subset_size"],
    batch_index: TensorType["full_size"],
    batch_num_graphs: int
) -> Tuple[TensorType["subset_size"], TensorType["subset_size"]]:
    # create a subset of `batch_index` corresponding to the indices specified in `idx`
    subset_batch_index = torch.cumsum(idx < torch.roll(idx, 1), dim=0).flatten() - 1

    # compute offsets for `scatter` operation
    offsets = torch.zeros(batch_num_graphs + 1, dtype=torch.long, device=idx.device)
    scatter(torch.ones_like(batch_index), batch_index, out=offsets[1:], reduce="add")
    offsets = torch.cumsum(offsets, dim=0)

    # add offsets to `edge_index` to convert batch-local indices to batch-global indices
    batch_idx = idx + offsets[subset_batch_index]

    return batch_idx, subset_batch_index


@typechecked
def write_residue_atom_positions_as_pdb(
    output_filepath: str,
    pos: np.ndarray,
    residue_to_atom_names_mapping: Dict[str, List[str]]
):
    with open(output_filepath, "w") as f:
        i, j = 1, 1
        for res in residue_to_atom_names_mapping:
            res_name = res[:3]
            for atom in residue_to_atom_names_mapping[res]:
                atom_seq = j
                res_seq = i
                x, y, z = pos[j - 1]
                line = "{:6s}{:5d} {:^4s} {:3s} {:1s}{:4d}    {:8.3f}{:8.3f}{:8.3f}{:6.2f}{:6.2f}".format("ATOM", atom_seq, atom, res_name, "A", res_seq, x, y, z, 0, 0)
                f.write(line + "\n")
                j = j + 1
            i = i + 1


@typechecked
def annotate_pdb_with_new_column_values(
    input_pdb_filepath: str,
    output_pdb_filepath: str,
    column_name: str,
    new_column_values: np.ndarray,
    pdb_df_key: str = "ATOM"
):
    pdb = PandasPdb().read_pdb(input_pdb_filepath)
    if len(pdb.df[pdb_df_key]) > 0 and column_name in pdb.df[pdb_df_key]:
        if column_name in ["b_factor"]:
            residue_indices = (pdb.df[pdb_df_key]["residue_number"].values - pdb.df[pdb_df_key]["residue_number"].values.min())
            pdb.df[pdb_df_key].loc[:, column_name] = new_column_values[residue_indices]
        else:
            raise NotImplementedError(f"PDB column {column_name} is currently not supported.")
    pdb.to_pdb(output_pdb_filepath)


def amber_relax(input_pdb_filepath: str, output_pdb_filepath: str, use_gpu: bool = False, verbose: bool = True):
    # adapted from: https://github.com/deepmind/alphafold
    from src.utils.amber import protein, relax
    
    test_config = {
        "max_iterations": RELAX_MAX_ITERATIONS,
        "tolerance": RELAX_ENERGY_TOLERANCE,
        "stiffness": RELAX_STIFFNESS,
        "exclude_residues": RELAX_EXCLUDE_RESIDUES,
        "max_outer_iterations": RELAX_MAX_OUTER_ITERATIONS,
        "use_gpu": use_gpu
    }

    amber_relax = relax.AmberRelaxation(**test_config)

    with open(input_pdb_filepath) as f:
      relaxed_prot = protein.from_pdb_string(f.read())
    try:
      relaxed_pdb_str, _, _ = amber_relax.process(prot=relaxed_prot)
      if verbose:
        log.info("`AmberRelaxation` has finished running!")
    except Exception as e:
      log.warning(f"Skipping AMBER relaxation for PDB {input_pdb_filepath} due to exception: {e}")
      with open(input_pdb_filepath) as f:
        relaxed_pdb_str = f.read()
    with open(output_pdb_filepath, "w") as f:
        f.write(relaxed_pdb_str)



def calculate_tmscore_metrics(pred_pdb_filepath: str, native_pdb_filepath: str, tmscore_exec_path: str) -> Dict[str, float]:
    """Calculates TM-score structural metrics between predicted and native protein structures.

    Args:
        pred_pdb_filepath (str): Filepath to predicted protein structure in PDB format.
        native_pdb_filepath (str): Filepath to native protein structure in PDB format.
        tmscore_exec_path (str): Path to TM-score executable.

    Returns:
        Dict[str, float]: Dictionary containing TM-score structural metrics (e.g., GDT-HA).
    """
    # run TM-score with subprocess and capture output
    cmd = [tmscore_exec_path, pred_pdb_filepath, native_pdb_filepath]
    output = subprocess.check_output(cmd, universal_newlines=True)

    # parse TM-score output to extract structural metrics
    metrics = {}
    for line in output.splitlines():
        if line.startswith("TM-score"):
            metrics["TM-score"] = float(line.split()[-3])
        elif line.startswith("MaxSub"):
            metrics["MaxSub"] = float(line.split()[-3])
        elif line.startswith("GDT-TS"):
            metrics["GDT-TS"] = float(line.split()[-5])
        elif line.startswith("RMSD"):
            metrics["RMSD"] = float(line.split()[-1])
        elif line.startswith("GDT-HA"):
            metrics["GDT-HA"] = float(line.split()[-5])

    return metrics



def calculate_molprobity_metrics(pdb_filepath: str, molprobity_exec_path: str) -> Dict[str, float]:
    """Calculates MolProbity metrics for a given protein structure.

    Args:
        pdb_filepath (str): Filepath to protein structure in PDB format.
        molprobity_exec_path (str): Path to MolProbity executable.

    Returns:
        Dict[str, float]: Dictionary containing MolProbity metrics (e.g., clashscore, rotamer outlier).
    """
    # run MolProbity with subprocess and capture output
    cmd = f"{molprobity_exec_path} {pdb_filepath}"
    stdout, _ = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE).communicate()

    # parse MolProbity output to extract structural metrics
    lines = stdout.decode("ascii").splitlines()
    metric_names = [item for item in lines[1].strip().split(":")]
    metric_values = [np.nan if item == "" else item for item in lines[2].strip().split(":")]
    if len(metric_names) != len(metric_values):
        # note: for backbone-only PDB inputs, parsing an alternative line for results may be required
        metric_values = [np.nan if item == "" else item for item in lines[4].strip().split(":")]
        if len(metric_names) != len(metric_values):
            # note: on some computing platforms, MolProbity may yield its output on the last line of standard output
            metric_values = [np.nan if item == "" else item for item in lines[-1].strip().split(":")]
    assert len(metric_names) == len(metric_values), "Number of column names must match number of column values within MolProbity's output."

    metrics = {
        "clash_score": float(metric_values[8]),
        "rotamer_outliers": float(metric_values[17]),
        "ramachandran_outliers": float(metric_values[20]),
        "molprobity_score": float(metric_values[45])
    }

    return metrics


class Queue():
    """
    Adapted from: https://github.com/arneschneuing/DiffSBDD
    """

    def __init__(self, max_len: int = 50):
        self.items = []
        self.max_len = max_len

    def __len__(self):
        return len(self.items)

    @typechecked
    def add(self, item: Any):
        self.items.insert(0, item)
        if len(self) > self.max_len:
            self.items.pop()

    @typechecked
    def mean(self) -> Any:
        return np.mean(self.items)

    @typechecked
    def std(self) -> Any:
        return np.std(self.items)
