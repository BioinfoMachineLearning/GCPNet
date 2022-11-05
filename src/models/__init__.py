# -------------------------------------------------------------------------------------------------------------------------------------
# Following code curated for GCPNet (https://github.com/BioinfoMachineLearning/GCPNet):
# -------------------------------------------------------------------------------------------------------------------------------------


import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from torchtyping import patch_typeguard
from typeguard import typechecked
from collections import defaultdict
from functools import partial
from atom3d.util import metrics

patch_typeguard()  # use before @typechecked

HALT_FILE_EXTENSION = "done"


@typechecked
def get_nonlinearity(nonlinearity: Optional[str] = None, slope: float = 1e-2, return_functional: bool = False) -> Any:
    nonlinearity = nonlinearity if nonlinearity is None else nonlinearity.lower().strip()
    if nonlinearity == "relu":
        return F.relu if return_functional else nn.ReLU()
    elif nonlinearity == "leakyrelu":
        return F.leaky_relu if return_functional else nn.LeakyReLU(negative_slope=slope)
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
